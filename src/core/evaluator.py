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
    
    def __init__(self, eval_client: ModelClient, output_format: str):
        self.eval_client = eval_client
        self.output_format = output_format
    
    def _is_error_output(self, output: str) -> tuple:
        for pattern in self.ERROR_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return True, f"检测到执行错误: {pattern}"
        return False, ""
    
    def evaluate(self, task: str, output: str) -> EvaluationResult:
        is_error, error_msg = self._is_error_output(output)
        
        if is_error:
            return EvaluationResult(
                content_quality=1,
                format_compliance=1,
                tool_usage=1,
                creativity=1,
                final_score=1.0,
                feedback=f"Agent执行失败，{error_msg}。输出内容: {output[:200]}"
            )
        
        system = """你是一个严格的评委。根据任务和输出进行多维度评分。
输出必须是JSON格式：
{
  "content_quality": 1-5分,
  "format_compliance": 1-5分,
  "tool_usage": 1-5分,
  "creativity": 1-5分,
  "feedback": "详细改进意见"
}"""
        
        user = f"""任务：{task}
输出：{output}
输出格式约定：{self.output_format}

评分标准：
- 内容质量：相关性、准确性、完整性
- 格式符合度：是否严格遵循约定格式
- 工具使用：工具选择和调用效果
- 创意性：输出的创新性

请给出评分和反馈。"""
        
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
                    feedback=data.get('feedback', '评估完成')
                )
        except Exception:
            pass
        
        return EvaluationResult(
            content_quality=3,
            format_compliance=3,
            tool_usage=3,
            creativity=3,
            final_score=3.0,
            feedback="评分解析失败，使用默认评分"
        )
