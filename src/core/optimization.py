import sqlite3
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from src.core.model_client import ModelClient
from src.core.tools import get_tools
from src.core.agent_factory import create_agent
from src.core.evaluator import Evaluator, EvaluationResult
from src.utils.format_checker import FormatChecker


class StopReason(Enum):
    SCORE_THRESHOLD = "score_threshold"
    MAX_ITERATIONS = "max_iterations"
    EARLY_STOP = "early_stop"
    PERFECT_SCORE = "perfect_score"
    USER_STOP = "user_stop"


@dataclass
class TaskResult:
    task: str
    output: str
    evaluation: EvaluationResult
    format_check: str
    error_log: Optional[str] = None


@dataclass
class IterationResult:
    iteration_number: int
    prompt: str
    task_results: List[TaskResult]
    avg_score: float
    stop_reason: Optional[StopReason] = None


class OptimizationLoop:
    def __init__(
        self,
        agent_id: int,
        db_path: str = "data/agents.db",
        on_task_start: Optional[Callable] = None,
        on_task_complete: Optional[Callable] = None,
        on_iteration_complete: Optional[Callable] = None,
        on_stream: Optional[Callable] = None
    ):
        self.agent_id = agent_id
        self.db_path = db_path
        self.on_task_start = on_task_start
        self.on_task_complete = on_task_complete
        self.on_iteration_complete = on_iteration_complete
        self.on_stream = on_stream
        
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        self._load_agent_info()
        self._init_model_clients()
        self._build_tools()
        self._init_evaluator()
        
        self._stop_requested = False
        self._score_history: List[float] = []
        self._current_results: List[TaskResult] = []
    
    def _load_agent_info(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM agents WHERE id = ?", (self.agent_id,))
        self.agent = dict(cursor.fetchone())
        
        cursor.execute(
            "SELECT * FROM model_configs WHERE agent_id = ? AND iteration_id IS NULL",
            (self.agent_id,)
        )
        configs = {c['model_type']: dict(c) for c in cursor.fetchall()}
        
        self.model_configs = configs
        
        cursor.execute(
            "SELECT task_description FROM tasks WHERE agent_id = ? AND is_active = 1",
            (self.agent_id,)
        )
        self.tasks = [row['task_description'] for row in cursor.fetchall()]
    
    def _init_model_clients(self):
        def get_client(model_type: str, default_model: str) -> ModelClient:
            config = self.model_configs.get(model_type, {})
            return ModelClient(
                source=config.get('model_source', 'ollama'),
                model_name=config.get('model_name', default_model),
                api_base=config.get('api_endpoint'),
                api_key=config.get('api_key_encrypted')
            )
        
        self.base_client = get_client('base', 'deepseek-r1:7b')
        self.eval_client = get_client('evaluator', 'deepseek-r1:7b')
        self.optimizer_client = get_client('optimizer', 'deepseek-r1:7b')
        self.task_gen_client = get_client('task_generator', 'deepseek-r1:7b')
    
    def _build_tools(self):
        search_config = self.model_configs.get('search', {})
        self.tools = get_tools(
            search_api_key=search_config.get('api_key_encrypted'),
            search_cse_id=search_config.get('cse_id')
        )
    
    def _init_evaluator(self):
        self.evaluator = Evaluator(self.eval_client, self.agent['output_format'])
        self.format_checker = FormatChecker()
    
    def generate_initial_prompt(self) -> str:
        system = "你是一个专业的Prompt工程师。根据用户需求创建详细的Agent系统提示。"
        user = f"""为{self.agent['role']}角色创建系统提示。
需求：{self.agent['user_requirement']}
输出格式必须遵循：{self.agent['output_format']}

请包含：
1. 角色描述
2. 核心能力
3. 可用工具及使用说明
4. 工作流程
5. 输出格式要求（必须严格遵守）

只输出prompt内容，不要解释。"""
        
        if self.on_stream:
            return self.base_client.chat_stream_collect(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                callback=self.on_stream
            )
        return self.base_client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ])
    
    def generate_tasks(self, num_tasks: int = 3) -> List[str]:
        if self.tasks:
            return self.tasks
        
        system = "你是一个任务生成器。根据角色生成典型的测试任务。"
        user = f"""角色：{self.agent['role']}
输出格式要求：{self.agent['output_format']}

请生成{num_tasks}个测试任务，要求：
1. 每个任务一行
2. 任务能充分检验角色能力
3. 任务能检验输出格式遵循情况
4. 包含简单、中等、复杂不同难度

只输出任务列表，每行一个任务。"""
        
        response = self.task_gen_client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ])
        
        tasks = [line.strip() for line in response.split('\n') if line.strip()]
        tasks = tasks[:num_tasks]
        
        cursor = self.conn.cursor()
        for t in tasks:
            cursor.execute(
                "INSERT INTO tasks (agent_id, task_description) VALUES (?, ?)",
                (self.agent_id, t)
            )
        self.conn.commit()
        
        self.tasks = tasks
        return tasks
    
    def run_task(self, agent_executor, task: str) -> TaskResult:
        if self.on_task_start:
            self.on_task_start(task)
        
        error_log = None
        try:
            output = agent_executor.invoke({"input": task})
            output_text = output.get('output', str(output))
        except Exception as e:
            output_text = f"任务执行失败：{str(e)}"
            error_log = str(e)
        
        format_check = self.format_checker.check(output_text, self.agent['output_format'])
        evaluation = self.evaluator.evaluate(task, output_text)
        
        result = TaskResult(
            task=task,
            output=output_text,
            evaluation=evaluation,
            format_check=format_check,
            error_log=error_log
        )
        
        if self.on_task_complete:
            self.on_task_complete(result)
        
        return result
    
    def improve_prompt(
        self,
        old_prompt: str,
        results: List[TaskResult],
        avg_score: float
    ) -> str:
        system = "你是一个Prompt优化专家。根据测试反馈修改prompt，只输出新prompt。"
        
        feedback_summary = "\n".join([
            f"任务：{r.task}\n"
            f"评分：{r.evaluation.final_score}\n"
            f"格式检查：{r.format_check}\n"
            f"反馈：{r.evaluation.feedback}"
            for r in results
        ])
        
        user = f"""当前prompt：
{old_prompt}

平均分：{avg_score}
反馈详情：
{feedback_summary}

输出格式要求：{self.agent['output_format']}

请生成改进后的新prompt，特别关注：
1. 提高内容质量
2. 确保格式遵循
3. 优化工具使用指导

只输出新prompt，不要解释。"""
        
        if self.on_stream:
            return self.optimizer_client.chat_stream_collect(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                callback=self.on_stream
            )
        return self.optimizer_client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ])
    
    def check_stop_condition(
        self,
        avg_score: float,
        iteration: int,
        max_iterations: int,
        score_threshold: float
    ) -> Optional[StopReason]:
        if self._stop_requested:
            return StopReason.USER_STOP
        
        if avg_score >= score_threshold:
            return StopReason.SCORE_THRESHOLD
        
        if iteration >= max_iterations:
            return StopReason.MAX_ITERATIONS
        
        all_perfect = all(
            r.evaluation.final_score == 5.0
            for r in self._current_results
        )
        if all_perfect and self._current_results:
            return StopReason.PERFECT_SCORE
        
        patience = self.agent.get('early_stop_patience', 3)
        threshold = self.agent.get('early_stop_threshold', 0.1)
        
        self._score_history.append(avg_score)
        if len(self._score_history) > patience:
            recent_scores = self._score_history[-patience:]
            improvements = [
                recent_scores[i] - recent_scores[i-1]
                for i in range(1, len(recent_scores))
            ]
            if all(imp < threshold for imp in improvements):
                return StopReason.EARLY_STOP
        
        return None
    
    def save_iteration(self, result: IterationResult):
        cursor = self.conn.cursor()
        
        cursor.execute(
            """INSERT INTO iterations 
               (agent_id, iteration_number, prompt, avg_score, scores_detail)
               VALUES (?, ?, ?, ?, ?)""",
            (
                self.agent_id,
                result.iteration_number,
                result.prompt,
                result.avg_score,
                json.dumps({
                    "content_quality": sum(r.evaluation.content_quality for r in result.task_results) / len(result.task_results),
                    "format_compliance": sum(r.evaluation.format_compliance for r in result.task_results) / len(result.task_results),
                    "tool_usage": sum(r.evaluation.tool_usage for r in result.task_results) / len(result.task_results),
                    "creativity": sum(r.evaluation.creativity for r in result.task_results) / len(result.task_results),
                })
            )
        )
        iter_id = cursor.lastrowid
        
        for r in result.task_results:
            cursor.execute(
                """INSERT INTO task_results
                   (iteration_id, task_description, output, scores, final_score, 
                    feedback, format_check, error_log)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    iter_id,
                    r.task,
                    r.output,
                    json.dumps({
                        "content_quality": r.evaluation.content_quality,
                        "format_compliance": r.evaluation.format_compliance,
                        "tool_usage": r.evaluation.tool_usage,
                        "creativity": r.evaluation.creativity,
                    }),
                    r.evaluation.final_score,
                    r.evaluation.feedback,
                    r.format_check,
                    r.error_log
                )
            )
        
        self.conn.commit()
    
    def request_stop(self):
        self._stop_requested = True
    
    def run_loop(
        self,
        max_iterations: int = 5,
        score_threshold: float = 4.5,
        initial_prompt: Optional[str] = None
    ) -> IterationResult:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM iterations WHERE agent_id = ?",
            (self.agent_id,)
        )
        start_iteration = cursor.fetchone()[0]
        
        if not self.tasks:
            self.generate_tasks()
        
        current_prompt = initial_prompt or self.generate_initial_prompt()
        iteration_result = None
        
        for i in range(start_iteration, start_iteration + max_iterations):
            iteration_num = i + 1
            
            agent_executor = create_agent(
                system_prompt=current_prompt,
                tools=self.tools,
                model_client=self.base_client
            )
            
            task_results = []
            for task in self.tasks:
                if self._stop_requested:
                    break
                result = self.run_task(agent_executor, task)
                task_results.append(result)
            
            self._current_results = task_results
            
            avg_score = sum(
                r.evaluation.final_score for r in task_results
            ) / len(task_results)
            
            stop_reason = self.check_stop_condition(
                avg_score, iteration_num, start_iteration + max_iterations, score_threshold
            )
            
            iteration_result = IterationResult(
                iteration_number=iteration_num,
                prompt=current_prompt,
                task_results=task_results,
                avg_score=avg_score,
                stop_reason=stop_reason
            )
            
            self.save_iteration(iteration_result)
            
            cursor.execute(
                "UPDATE agents SET status = ? WHERE id = ?",
                ('running', self.agent_id)
            )
            self.conn.commit()
            
            if self.on_iteration_complete:
                self.on_iteration_complete(iteration_result)
            
            if stop_reason:
                cursor.execute(
                    "UPDATE agents SET status = ?, final_prompt = ? WHERE id = ?",
                    ('completed' if stop_reason in [StopReason.SCORE_THRESHOLD, StopReason.PERFECT_SCORE] else 'paused',
                     current_prompt, self.agent_id)
                )
                self.conn.commit()
                return iteration_result
            
            current_prompt = self.improve_prompt(
                current_prompt, task_results, avg_score
            )
        
        return iteration_result
    
    def close(self):
        self.conn.close()
