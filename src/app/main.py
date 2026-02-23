import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import json
import threading
import time
from typing import Optional, List, Dict, Any

from src.db.database import init_db, Database
from src.config.settings import settings
from src.core.model_client import ModelClient, get_ollama_models
from src.core.tools import get_tools
from src.core.agent_factory import create_agent
from src.core.evaluator import Evaluator
from src.utils.format_checker import FormatChecker


st.set_page_config(
    page_title="Agent Prompt优化系统",
    page_icon="🤖",
    layout="wide"
)


def get_or_create_optimization_state():
    if 'optimization_states' not in st.session_state:
        st.session_state['optimization_states'] = {}
    return st.session_state['optimization_states']


class OptimizationRunner:
    def __init__(self, agent_id: int, db: Database):
        self.agent_id = agent_id
        self.db = db
        self.agent = db.get_agent(agent_id)
        self.status = "初始化"
        self.current_iteration = 0
        self.current_task = ""
        self.logs: List[str] = []
        self.prompt = ""
        self.avg_score = 0.0
        self.stop_requested = False
        self.completed = False
        self.error = None
        self.iteration_results: List[Dict[str, Any]] = []
        self.current_results: List[Dict[str, Any]] = []
        
        self._init_clients()
    
    def _init_clients(self):
        configs = self.db.get_model_configs(self.agent_id)
        
        def get_client(model_type: str, default_model: str) -> ModelClient:
            config = configs.get(model_type, {})
            return ModelClient(
                source=config.get('model_source', 'ollama'),
                model_name=config.get('model_name', default_model),
                api_base=config.get('api_endpoint'),
                api_key=config.get('api_key_encrypted')
            )
        
        self.base_client = get_client('base', settings.DEFAULT_MODEL)
        self.eval_client = get_client('evaluator', settings.DEFAULT_MODEL)
        self.optimizer_client = get_client('optimizer', settings.DEFAULT_MODEL)
        self.task_gen_client = get_client('task_generator', settings.DEFAULT_MODEL)
        
        self.tools = get_tools()
        self.evaluator = Evaluator(self.eval_client, self.agent['output_format'])
        self.format_checker = FormatChecker()
    
    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]
    
    def generate_initial_prompt(self) -> str:
        self.status = "生成初始Prompt"
        self.log("正在生成初始Prompt...")
        
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
        
        response = self.base_client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ])
        self.log("初始Prompt生成完成")
        return response
    
    def generate_tasks(self, num_tasks: int = 3) -> List[str]:
        tasks = self.db.get_active_tasks(self.agent_id)
        if tasks:
            return tasks
        
        self.status = "生成测试任务"
        self.log("正在生成测试任务...")
        
        system = """你是一个专业的任务设计专家。你需要设计能够全面测试Agent能力的具体任务。

任务设计原则：
1. 具体性：任务必须包含明确的输入数据、约束条件和预期输出
2. 可测试性：任务的输出可以被客观评估
3. 渐进难度：从简单到复杂，逐步增加难度
4. 格式检验：任务需要检验Agent是否严格遵守输出格式要求

每个任务应包含：
- 明确的场景背景
- 具体的输入要求（数据、参数等）
- 输出格式和内容要求
- 约束条件（字数、格式、必须包含的元素等）"""
        
        user = f"""角色：{self.agent['role']}
用户需求：{self.agent['user_requirement']}
输出格式要求：{self.agent['output_format']}

请生成{num_tasks}个详细的测试任务。每个任务用"【任务N】"开头，任务之间用空行分隔。

示例格式：
【任务1】
场景：某电商平台需要为新品手机撰写产品描述
输入信息：品牌为"星耀"，型号X1，主要卖点包括6.7英寸AMOLED屏幕、5000mAh电池、1亿像素主摄
输出要求：撰写150-200字的产品描述，使用Markdown格式，包含标题、核心卖点列表、购买引导语
约束条件：必须包含"旗舰体验"和"超长续航"两个关键词

请按照上述格式生成{num_tasks}个任务，难度从简单到复杂递进。"""
        
        response = self.task_gen_client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ], temperature=0.8)
        
        import re
        task_pattern = r'【任务\d+】(.*?)(?=【任务\d+】|$)'
        matches = re.findall(task_pattern, response, re.DOTALL)
        
        if matches:
            tasks = [m.strip() for m in matches if m.strip()]
        else:
            tasks = [line.strip() for line in response.split('\n') if line.strip() and len(line.strip()) > 20]
        
        tasks = tasks[:num_tasks]
        
        for t in tasks:
            self.db.add_task(self.agent_id, t)
        
        self.log(f"生成了 {len(tasks)} 个详细测试任务")
        return tasks
    
    def run_task(self, agent_executor, task: str) -> Dict[str, Any]:
        self.current_task = task
        self.log(f"执行任务: {task[:50]}...")
        
        try:
            output = agent_executor.invoke({"input": task})
            output_text = output.get('output', str(output))
        except Exception as e:
            output_text = f"任务执行失败：{str(e)}"
            self.log(f"任务执行错误: {str(e)}")
        
        format_check = self.format_checker.check(output_text, self.agent['output_format'])
        evaluation = self.evaluator.evaluate(task, output_text)
        
        self.log(f"任务评分: {evaluation.final_score}")
        
        return {
            "task": task,
            "output": output_text,
            "evaluation": evaluation,
            "format_check": format_check
        }
    
    def improve_prompt(self, old_prompt: str, results: List[Dict], avg_score: float) -> str:
        self.status = "优化Prompt"
        self.log("正在优化Prompt...")
        
        system = "你是一个Prompt优化专家。根据测试反馈修改prompt，只输出新prompt。"
        
        feedback_summary = "\n".join([
            f"任务：{r['task']}\n"
            f"评分：{r['evaluation'].final_score}\n"
            f"格式检查：{r['format_check']}\n"
            f"反馈：{r['evaluation'].feedback}"
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
        
        response = self.optimizer_client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ])
        self.log("Prompt优化完成")
        return response
    
    def run(self, max_iterations: int = 5, score_threshold: float = 4.5):
        try:
            self.db.update_agent_status(self.agent_id, 'running')
            
            tasks = self.generate_tasks()
            current_prompt = self.generate_initial_prompt()
            self.prompt = current_prompt
            
            score_history = []
            
            for iteration in range(1, max_iterations + 1):
                if self.stop_requested:
                    self.status = "已停止"
                    self.log("用户请求停止")
                    self.db.update_agent_status(self.agent_id, 'paused', current_prompt)
                    break
                
                self.current_iteration = iteration
                self.status = f"迭代 {iteration}/{max_iterations}"
                self.log(f"开始第 {iteration} 次迭代")
                
                agent_executor = create_agent(
                    system_prompt=current_prompt,
                    tools=self.tools,
                    model_client=self.base_client
                )
                
                results = []
                self.current_results = []
                for task in tasks:
                    if self.stop_requested:
                        break
                    result = self.run_task(agent_executor, task)
                    results.append(result)
                    self.current_results.append(result)
                
                if not results:
                    break
                
                avg_score = sum(r['evaluation'].final_score for r in results) / len(results)
                self.avg_score = avg_score
                score_history.append(avg_score)
                self.log(f"迭代 {iteration} 平均分: {avg_score:.2f}")
                
                self.iteration_results.append({
                    "iteration": iteration,
                    "prompt": current_prompt,
                    "results": results.copy(),
                    "avg_score": avg_score
                })
                
                self._save_iteration(iteration, current_prompt, results, avg_score)
                
                if avg_score >= score_threshold:
                    self.status = "完成 - 达到评分阈值"
                    self.log(f"达到评分阈值 {score_threshold}")
                    self.db.update_agent_status(self.agent_id, 'completed', current_prompt)
                    self.completed = True
                    break
                
                if avg_score == 5.0:
                    self.status = "完成 - 满分"
                    self.log("获得满分")
                    self.db.update_agent_status(self.agent_id, 'completed', current_prompt)
                    self.completed = True
                    break
                
                patience = self.agent.get('early_stop_patience', 3)
                threshold = self.agent.get('early_stop_threshold', 0.1)
                
                if len(score_history) > patience:
                    recent = score_history[-patience:]
                    improvements = [recent[i] - recent[i-1] for i in range(1, len(recent))]
                    if all(imp < threshold for imp in improvements):
                        self.status = "完成 - 早停"
                        self.log("触发早停条件")
                        self.db.update_agent_status(self.agent_id, 'completed', current_prompt)
                        self.completed = True
                        break
                
                current_prompt = self.improve_prompt(current_prompt, results, avg_score)
                self.prompt = current_prompt
            
            if not self.completed and not self.stop_requested:
                self.status = "完成 - 达到最大迭代次数"
                self.db.update_agent_status(self.agent_id, 'completed', current_prompt)
                self.completed = True
            
            self.log("优化完成")
            
        except Exception as e:
            self.error = str(e)
            self.status = f"错误: {str(e)}"
            self.log(f"错误: {str(e)}")
            self.db.update_agent_status(self.agent_id, 'failed')
    
    def _save_iteration(self, iteration: int, prompt: str, results: List[Dict], avg_score: float):
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            """INSERT INTO iterations 
               (agent_id, iteration_number, prompt, avg_score, scores_detail)
               VALUES (?, ?, ?, ?, ?)""",
            (self.agent_id, iteration, prompt, avg_score, json.dumps({
                "avg": avg_score
            }))
        )
        iter_id = cursor.lastrowid
        
        for r in results:
            cursor.execute(
                """INSERT INTO task_results
                   (iteration_id, task_description, output, scores, final_score, 
                    feedback, format_check)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (iter_id, r['task'], r['output'][:5000], 
                 json.dumps({"score": r['evaluation'].final_score}),
                 r['evaluation'].final_score, r['evaluation'].feedback,
                 r['format_check'])
            )
        
        conn.commit()
        conn.close()


def main():
    init_db(settings.DB_PATH)
    settings.setup_file_dir()
    
    st.sidebar.title("🤖 Agent Prompt优化系统")
    page = st.sidebar.radio(
        "导航",
        ["创建新Agent", "历史记录", "设置"]
    )
    
    if page == "创建新Agent":
        show_create_page()
    elif page == "历史记录":
        show_history_page()
    elif page == "设置":
        show_settings_page()


def show_create_page():
    st.header("创建新Agent")
    
    with st.form("agent_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            role = st.text_input("Agent角色", placeholder="如：文案策划")
            user_requirement = st.text_area(
                "需求描述",
                height=150,
                placeholder="详细描述Agent需要完成的任务和能力要求"
            )
        
        with col2:
            output_formats = st.multiselect(
                "输出格式",
                ["Markdown", "JSON", "YAML", "XML", "CSV", "纯文本"],
                default=["Markdown"]
            )
            custom_format = st.text_input(
                "自定义格式要求",
                placeholder="如有特殊格式要求，请在此描述"
            )
        
        st.subheader("参数配置")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            max_iterations = st.number_input(
                "最大迭代次数",
                min_value=1,
                max_value=20,
                value=settings.MAX_ITERATIONS
            )
            score_threshold = st.slider(
                "评分阈值",
                min_value=3.0,
                max_value=5.0,
                value=settings.SCORE_THRESHOLD,
                step=0.1
            )
        
        with col4:
            early_stop_patience = st.number_input(
                "早停耐心值",
                min_value=1,
                max_value=10,
                value=settings.EARLY_STOP_PATIENCE,
                help="连续多少次无提升时停止"
            )
            early_stop_threshold = st.slider(
                "早停阈值",
                min_value=0.0,
                max_value=1.0,
                value=settings.EARLY_STOP_THRESHOLD,
                step=0.05,
                help="提升幅度低于此值视为无提升"
            )
        
        with col5:
            if 'ollama_models' not in st.session_state:
                st.session_state['ollama_models'] = get_ollama_models(settings.OLLAMA_BASE_URL.replace('/v1', ''))
            
            ollama_models = st.session_state['ollama_models']
            
            if not ollama_models:
                st.warning("无法连接Ollama，请确保Ollama正在运行")
                base_model = st.text_input("基础模型", value=settings.DEFAULT_MODEL, key="base_model_text")
                eval_model = st.text_input("评估模型", value=settings.DEFAULT_MODEL, key="eval_model_text")
            else:
                if 'base_model_index' not in st.session_state:
                    if settings.DEFAULT_MODEL in ollama_models:
                        st.session_state['base_model_index'] = ollama_models.index(settings.DEFAULT_MODEL)
                    else:
                        st.session_state['base_model_index'] = 0
                
                if 'eval_model_index' not in st.session_state:
                    if settings.DEFAULT_MODEL in ollama_models:
                        st.session_state['eval_model_index'] = ollama_models.index(settings.DEFAULT_MODEL)
                    else:
                        st.session_state['eval_model_index'] = 0
                
                base_model = st.selectbox(
                    "基础模型",
                    options=ollama_models,
                    index=st.session_state['base_model_index'],
                    key="base_model_select"
                )
                
                eval_model = st.selectbox(
                    "评估模型",
                    options=ollama_models,
                    index=st.session_state['eval_model_index'],
                    key="eval_model_select"
                )
                
                st.session_state['base_model_index'] = ollama_models.index(base_model) if base_model in ollama_models else 0
                st.session_state['eval_model_index'] = ollama_models.index(eval_model) if eval_model in ollama_models else 0
        
        submitted = st.form_submit_button("开始生成", type="primary")
    
    col_refresh, col_spacer = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 刷新模型列表"):
            st.session_state['ollama_models'] = get_ollama_models(settings.OLLAMA_BASE_URL.replace('/v1', ''))
            st.rerun()
    
    if submitted:
        if not role or not user_requirement:
            st.error("请填写角色和需求描述")
            return
        
        output_format = ", ".join(output_formats)
        if custom_format:
            output_format += f" ({custom_format})"
        
        db = Database()
        agent_id = db.create_agent(
            role=role,
            user_requirement=user_requirement,
            output_format=output_format,
            early_stop_patience=early_stop_patience,
            early_stop_threshold=early_stop_threshold
        )
        
        for model_type, model_name in [
            ('base', base_model),
            ('evaluator', eval_model),
            ('optimizer', base_model),
            ('task_generator', base_model)
        ]:
            db.add_model_config(
                agent_id=agent_id,
                model_type=model_type,
                model_source='ollama',
                model_name=model_name
            )
        
        st.session_state['current_agent_id'] = agent_id
        st.session_state['optimization_running'] = True
        st.rerun()
    
    agent_id = st.session_state.get('current_agent_id')
    if agent_id:
        show_running_page(agent_id)


def show_running_page(agent_id: int):
    st.header("🔄 优化进行中")
    
    states = get_or_create_optimization_state()
    
    if agent_id not in states:
        db = Database()
        runner = OptimizationRunner(agent_id, db)
        states[agent_id] = runner
        
        max_iter = settings.MAX_ITERATIONS
        threshold = settings.SCORE_THRESHOLD
        
        def run_optimization():
            runner.run(max_iterations=max_iter, score_threshold=threshold)
        
        thread = threading.Thread(target=run_optimization, daemon=True)
        thread.start()
        st.session_state['optimization_thread'] = thread
    
    runner = states.get(agent_id)
    
    if not runner:
        st.error("优化状态丢失")
        return
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("状态", runner.status)
    with col2:
        st.metric("当前迭代", f"{runner.current_iteration}")
    with col3:
        st.metric("平均分", f"{runner.avg_score:.2f}")
    
    if st.button("⏹ 停止优化", type="secondary"):
        runner.stop_requested = True
        st.info("已请求停止...")
    
    st.subheader("📝 当前Prompt")
    if runner.prompt:
        with st.expander("查看Prompt", expanded=False):
            st.code(runner.prompt, language="markdown")
    
    st.subheader("� 当前迭代任务结果")
    if runner.current_results:
        for idx, result in enumerate(runner.current_results, 1):
            with st.expander(f"任务 {idx}: {result['task'][:60]}...", expanded=False):
                st.markdown("**任务描述:**")
                st.info(result['task'])
                
                st.markdown("**Agent输出:**")
                st.code(result['output'], language="markdown")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("评分", f"{result['evaluation'].final_score:.2f}")
                with col_b:
                    format_status = "✅ 通过" if result['format_check'] else "❌ 未通过"
                    st.metric("格式检查", format_status)
                
                st.markdown("---")
                st.markdown("### 📊 评分详情")
                
                eval_obj = result['evaluation']
                st.code(eval_obj.get_scoring_rules() if hasattr(eval_obj, 'get_scoring_rules') else """评分规则:
- 内容质量(40%): 评估内容与任务的相关性、准确性、完整性
- 格式符合度(25%): 评估是否严格遵循约定的输出格式
- 工具使用(20%): 评估工具选择和调用效果
- 创意性(15%): 评估输出的创新性""", language="markdown")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("内容质量", f"{eval_obj.content_quality}/5", help=eval_obj.content_quality_reason)
                with col2:
                    st.metric("深度完整性", f"{eval_obj.depth_completeness}/5", help=eval_obj.depth_completeness_reason)
                with col3:
                    st.metric("格式符合度", f"{eval_obj.format_compliance}/5", help=eval_obj.format_compliance_reason)
                with col4:
                    st.metric("工具使用", f"{eval_obj.tool_usage}/5", help=eval_obj.tool_usage_reason)
                with col5:
                    st.metric("创意性", f"{eval_obj.creativity}/5", help=eval_obj.creativity_reason)
                
                if eval_obj.content_quality_reason:
                    with st.expander("📝 各维度评分理由"):
                        st.markdown(f"**内容质量 ({eval_obj.content_quality}/5):** {eval_obj.content_quality_reason}")
                        st.markdown(f"**深度完整性 ({eval_obj.depth_completeness}/5):** {eval_obj.depth_completeness_reason}")
                        st.markdown(f"**格式符合度 ({eval_obj.format_compliance}/5):** {eval_obj.format_compliance_reason}")
                        st.markdown(f"**工具使用 ({eval_obj.tool_usage}/5):** {eval_obj.tool_usage_reason}")
                        st.markdown(f"**创意性 ({eval_obj.creativity}/5):** {eval_obj.creativity_reason}")
                
                st.markdown("**总体反馈:**")
                st.success(eval_obj.feedback)
    
    st.subheader("� 执行日志")
    log_container = st.container()
    with log_container:
        for log in runner.logs[-20:]:
            st.text(log)
    
    st.subheader("📈 评分趋势")
    db = Database()
    iterations = db.get_iterations(agent_id)
    
    if iterations:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[i['iteration_number'] for i in iterations],
            y=[i['avg_score'] for i in iterations],
            mode='lines+markers',
            name='平均分',
            line=dict(color='#FF4B4B', width=2),
            marker=dict(size=10)
        ))
        fig.update_layout(
            xaxis_title="迭代次数",
            yaxis_title="平均分",
            yaxis_range=[0, 5.5],
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📚 历史迭代详情")
    if runner.iteration_results:
        for iter_data in runner.iteration_results:
            with st.expander(f"迭代 {iter_data['iteration']} - 平均分: {iter_data['avg_score']:.2f}", expanded=False):
                st.markdown("**该迭代Prompt:**")
                st.code(iter_data['prompt'], language="markdown")
                
                st.markdown("**任务执行结果:**")
                for idx, result in enumerate(iter_data['results'], 1):
                    st.markdown(f"**任务 {idx}:**")
                    st.info(result['task'])
                    st.code(result['output'], language="markdown")
                    st.caption(f"评分: {result['evaluation'].final_score:.2f} | 格式: {'✅' if result['format_check'] else '❌'}")
    
    if runner.completed or runner.error:
        if runner.error:
            st.error(f"优化失败: {runner.error}")
        else:
            st.success(f"优化完成！最终状态: {runner.status}")
        
        if st.button("查看详情", type="primary"):
            st.session_state['optimization_running'] = False
            st.session_state['current_agent_id'] = agent_id
            st.rerun()
    
    if not runner.completed and not runner.error:
        time.sleep(2)
        st.rerun()


def show_history_page():
    st.header("历史记录")
    
    db = Database()
    agents = db.list_agents()
    
    if not agents:
        st.info("暂无历史记录")
        return
    
    for agent in agents:
        with st.expander(f"#{agent['id']} {agent['role']} - {agent['status']} ({agent['created_at']})"):
            st.markdown(f"**输出格式**: {agent['output_format']}")
            st.markdown(f"**需求**: {agent['user_requirement']}")
            
            iterations = db.get_iterations(agent['id'])
            
            if iterations:
                df = pd.DataFrame(iterations)
                st.dataframe(df[['iteration_number', 'avg_score', 'created_at']])
            
            if agent['final_prompt']:
                st.markdown("**最终Prompt**:")
                st.code(agent['final_prompt'], language="markdown")


def show_settings_page():
    st.header("设置")
    
    st.subheader("Ollama配置")
    ollama_url = st.text_input("Ollama地址", value=settings.OLLAMA_BASE_URL)
    
    st.subheader("搜索引擎配置")
    search_api_key = st.text_input("Google API Key", type="password", value=settings.GOOGLE_API_KEY or "")
    search_cse_id = st.text_input("Google CSE ID", value=settings.GOOGLE_CSE_ID or "")
    
    st.subheader("安全配置")
    file_dir = st.text_input("允许的文件目录", value=settings.AGENT_FILE_DIR)
    
    st.subheader("默认参数")
    col1, col2 = st.columns(2)
    with col1:
        default_max_iter = st.number_input("默认最大迭代次数", value=settings.MAX_ITERATIONS)
        default_threshold = st.slider("默认评分阈值", min_value=3.0, max_value=5.0, value=settings.SCORE_THRESHOLD)
    
    with col2:
        default_patience = st.number_input("默认早停耐心值", value=settings.EARLY_STOP_PATIENCE)
        default_early_threshold = st.slider("默认早停阈值", min_value=0.0, max_value=1.0, value=settings.EARLY_STOP_THRESHOLD)
    
    if st.button("保存设置"):
        st.success("设置已保存（注意：当前会话有效，永久保存请修改.env文件）")


if __name__ == "__main__":
    main()
