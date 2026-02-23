import json
import re
from typing import Optional
from dataclasses import dataclass

from src.core.model_client import ModelClient


@dataclass
class EvaluationResult:
    content_quality: int
    format_compliance: int
    tool_usage: int
    creativity: int
    final_score: float
    feedback: str
    content_quality_reason: str = ""
    format_compliance_reason: str = ""
    tool_usage_reason: str = ""
    creativity_reason: str = ""


class Evaluator:
    ERROR_PATTERNS = [
        r'Agent执行错误',
        r'任务执行失败',
        r'Error:',
        r'Exception:',
        r'Traceback',
        r'执行出错',
        r'调用失败',
        r'超时',
        r'Timeout',
    ]
    
    SCORING_RULES = """
========================================
评分规则说明
========================================

本评估系统从4个维度对Agent输出进行评分：

【内容质量】(权重40%)
- 5分: 输出内容完全符合任务要求，信息准确完整，逻辑清晰
- 4分: 内容基本符合要求，有少量遗漏或不够详细
- 3分: 内容基本满足任务要求，但有部分不准确或不完整
- 2分: 内容与任务要求有较大偏差
- 1分: 内容完全偏离任务要求

【格式符合度】(权重25%)
- 5分: 完全按照约定的输出格式要求，无任何偏差
- 4分: 格式基本正确，有轻微不符合
- 3分: 格式部分符合要求
- 2分: 格式与要求有较大偏差
- 1分: 完全不符合约定格式

【工具使用】(权重20%)
- 5分: 正确选择并使用了所有必要的工具
- 4分: 工具选择基本正确，使用效果良好
- 3分: 工具选择和使用基本合理
- 2分: 工具选择不当或使用效果不佳
- 1分: 未正确使用工具

【创意性】(权重15%)
- 5分: 输出非常有创意，超出预期
- 4分: 输出有一定创意
- 3分: 输出中规中矩
- 2分: 创意较少
- 1分: 缺乏创意

最终得分 = 内容质量×0.4 + 格式符合度×0.25 + 工具使用×0.2 + 创意性×0.15
========================================
"""
    
    def __init__(self, eval_client: ModelClient, output_format: str):
        self.eval_client = eval_client
        self.output_format = output_format
    
    def _is_error_output(self, output: str) -> tuple:
        for pattern in self.ERROR_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return True, f"检测到执行错误: {pattern}"
        return False, ""
    
    def get_scoring_rules(self) -> str:
        return self.SCORING_RULES
    
    def evaluate(self, task: str, output: str) -> EvaluationResult:
        is_error, error_msg = self._is_error_output(output)
        
        if is_error:
            return EvaluationResult(
                content_quality=1,
                format_compliance=1,
                tool_usage=1,
                creativity=1,
                final_score=1.0,
                feedback=f"Agent执行失败，{error_msg}。输出内容: {output[:200]}",
                content_quality_reason="执行错误，无法评估",
                format_compliance_reason="执行错误，无法评估",
                tool_usage_reason="执行错误，无法评估",
                creativity_reason="执行错误，无法评估"
            )
        
        system = f"""你是一个严格的评委。根据任务和输出进行多维度评分。
输出必须是JSON格式：
{{
  "content_quality": 1-5分,
  "content_quality_reason": "详细说明为什么给这个分数",
  "format_compliance": 1-5分,
  "format_compliance_reason": "详细说明为什么给这个分数",
  "tool_usage": 1-5分,
  "tool_usage_reason": "详细说明为什么给这个分数",
  "creativity": 1-5分,
  "creativity_reason": "详细说明为什么给这个分数",
  "feedback": "总体改进建议"
}}

评分必须严格按照以下标准：
- 内容质量(40%): 评估内容与任务的相关性、准确性、完整性
- 格式符合度(25%): 评估是否严格遵循约定的输出格式
- 工具使用(20%): 评估工具选择和调用效果
- 创意性(15%): 评估输出的创新性

每个维度的reason必须具体说明：
1. 哪些地方做得好
2. 哪些地方需要改进
3. 给出具体的分数依据"""
        
        user = f"""任务：{task}
输出：{output}
输出格式约定：{self.output_format}

请严格按照评分标准给出每个维度的分数和详细理由。"""
        
        response = self.eval_client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ], temperature=0.1)
        
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                
                def clamp(val, min_val=1, max_val=5):
                    return max(min_val, min(max_val, int(val)))
                
                content_quality = clamp(data.get('content_quality', 3))
                format_compliance = clamp(data.get('format_compliance', 3))
                tool_usage = clamp(data.get('tool_usage', 3))
                creativity = clamp(data.get('creativity', 3))
                
                final_score = (
                    content_quality * 0.4 +
                    format_compliance * 0.25 +
                    tool_usage * 0.2 +
                    creativity * 0.15
                )
                
                return EvaluationResult(
                    content_quality=content_quality,
                    format_compliance=format_compliance,
                    tool_usage=tool_usage,
                    creativity=creativity,
                    final_score=round(final_score, 2),
                    feedback=data.get('feedback', '评估完成'),
                    content_quality_reason=data.get('content_quality_reason', ''),
                    format_compliance_reason=data.get('format_compliance_reason', ''),
                    tool_usage_reason=data.get('tool_usage_reason', ''),
                    creativity_reason=data.get('creativity_reason', '')
                )
        except Exception as e:
            pass
        
        return EvaluationResult(
            content_quality=3,
            format_compliance=3,
            tool_usage=3,
            creativity=3,
            final_score=3.0,
            feedback="评分解析失败，使用默认评分",
            content_quality_reason="解析失败",
            format_compliance_reason="解析失败",
            tool_usage_reason="解析失败",
            creativity_reason="解析失败"
        )
