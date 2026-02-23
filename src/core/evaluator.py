import json
import re
import yaml
from typing import Optional
from dataclasses import dataclass

from src.core.model_client import ModelClient


@dataclass
class EvaluationResult:
    content_quality: int
    format_compliance: int
    tool_usage: int
    creativity: int
    depth_completeness: int
    final_score: float
    feedback: str
    content_quality_reason: str = ""
    format_compliance_reason: str = ""
    tool_usage_reason: str = ""
    creativity_reason: str = ""
    depth_completeness_reason: str = ""


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
    
    DOC_TYPE_RULES = {
        "markdown": {
            "name": "Markdown文档",
            "depth_criteria": """
【内容深度与完整性】(Markdown文档):
- 5分: >5000字，多级标题结构完整，内容极其详尽
- 4分: 3000-5000字，结构完整，内容详尽
- 3分: 1500-3000字，结构基本完整
- 2分: 800-1500字，内容偏少
- 1分: <800字，内容严重不足""",
            "analyze": lambda o: Evaluator._analyze_markdown_depth(o)
        },
        "ppt": {
            "name": "PPT幻灯片",
            "depth_criteria": """
【内容深度与完整性】(PPT幻灯片):
- 5分: >15页，每页内容充实，包含详细图表/要点/分析
- 4分: 10-15页，每页内容较充实
- 3分: 6-10页，内容基本完整
- 2分: 3-6页，内容偏少
- 1分: <3页，内容严重不足""",
            "analyze": lambda o: Evaluator._analyze_ppt_depth(o)
        },
        "word": {
            "name": "Word文档",
            "depth_criteria": """
【内容深度与完整性】(Word文档):
- 5分: >8000字，章节完整，包含目录、摘要、正文、结论
- 4分: 5000-8000字，结构完整
- 3分: 3000-5000字，内容基本完整
- 2分: 1500-3000字，内容偏少
- 1分: <1500字，内容严重不足""",
            "analyze": lambda o: Evaluator._analyze_word_depth(o)
        },
        "excel": {
            "name": "Excel表格",
            "depth_criteria": """
【内容深度与完整性】(Excel表格):
- 5分: >10个sheet，数据完整，公式/图表丰富
- 4分: 5-10个sheet，数据较完整
- 3分: 2-5个sheet，基本可用
- 2分: 1-2个sheet，数据偏少
- 1分: 无有效数据""",
            "analyze": lambda o: Evaluator._analyze_excel_depth(o)
        },
        "json": {
            "name": "JSON数据",
            "depth_criteria": """
【内容深度与完整性】(JSON数据):
- 5分: 数据结构完整，字段丰富，>50个键值对
- 4分: 结构较完整，20-50个键值对
- 3分: 基本结构，10-20个键值对
- 2分: 结构简单，<10个键值对
- 1分: 数据严重不足""",
            "analyze": lambda o: Evaluator._analyze_json_depth(o)
        },
        "yaml": {
            "name": "YAML配置",
            "depth_criteria": """
【内容深度与完整性】(YAML配置):
- 5分: 配置项完整，>30个配置项，结构清晰
- 4分: 配置较完整，15-30个配置项
- 3分: 基本配置，8-15个配置项
- 2分: 配置简单，<8个配置项
- 1分: 配置严重不足""",
            "analyze": lambda o: Evaluator._analyze_yaml_depth(o)
        },
        "csv": {
            "name": "CSV数据",
            "depth_criteria": """
【内容深度与完整性】(CSV数据):
- 5分: >1000行数据，字段完整
- 4分: 500-1000行数据
- 3分: 100-500行数据
- 2分: 20-100行数据
- 1分: <20行数据""",
            "analyze": lambda o: Evaluator._analyze_csv_depth(o)
        },
        "纯文本": {
            "name": "纯文本",
            "depth_criteria": """
【内容深度与完整性】(纯文本):
- 5分: >5000字，内容极其详尽
- 4分: 3000-5000字，内容详尽
- 3分: 1500-3000字，内容基本完整
- 2分: 800-1500字，内容偏少
- 1分: <800字，内容严重不足""",
            "analyze": lambda o: Evaluator._analyze_text_depth(o)
        }
    }
    
    FORMAT_RULES = {
        "markdown": """
【Markdown格式标准】(5分制):
- 5分: 完全符合Markdown规范，包括:
  * 正确使用标题层级(# ## ###)
  * 列表(有序/无序)格式正确
  * 代码块使用```包裹并标注语言
  * 链接和图片语法正确
  * 表格使用标准语法
  * 强调(*或_)正确使用
- 4分: 格式基本正确，有1-2处不规范
- 3分: 格式部分正确，但有多处不规范
- 2分: 格式混乱，难以阅读
- 1分: 完全未使用Markdown格式
""",
        "json": """
【JSON格式标准】(5分制):
- 5分: 完全符合JSON规范:
  * 有效的JSON字符串
  * 键名使用双引号
  * 无尾部逗号
  * 数据类型正确
  * 结构清晰完整
- 4分: 基本有效JSON，有1处小问题
- 3分: JSON基本有效，但结构不完整
- 2分: JSON格式错误但可修复
- 1分: 完全无效的JSON
""",
        "yaml": """
【YAML格式标准】(5分制):
- 5分: 完全符合YAML规范:
  * 缩进正确(空格，不能用tab)
  * 键值对格式正确
  * 列表格式正确
  * 无语法错误
- 4分: 基本正确，有1-2处不规范
- 3分: 格式部分正确
- 2分: 格式混乱
- 1分: 完全无效的YAML
""",
        "xml": """
【XML格式标准】(5分制):
- 5分: 完全符合XML规范:
  * 标签配对正确
  * 属性使用引号
  * 有正确的声明
  * 结构清晰
- 4分: 基本正确，有1-2处不规范
- 3分: 格式部分正确
- 2分: 格式混乱
- 1分: 完全无效的XML
""",
        "csv": """
【CSV格式标准】(5分制):
- 5分: 完全符合CSV规范:
  * 表头清晰
  * 列数一致
  * 无多余空白
  * 逗号分隔正确
  * 无合并单元格
- 4分: 基本正确，有1-2处不规范
- 3分: 格式部分正确
- 2分: 格式混乱
- 1分: 完全无效的CSV
""",
        "ppt": """
【PPT格式标准】(5分制):
- 5分: 结构清晰，包含封面、目录、正文、结尾
- 4分: 结构较完整
- 3分: 有基本结构
- 2分: 结构混乱
- 1分: 无结构
""",
        "word": """
【Word格式标准】(5分制):
- 5分: 结构完整，有标题、目录、正文、页眉页脚
- 4分: 结构较完整
- 3分: 有基本格式
- 2分: 格式混乱
- 1分: 无格式
""",
        "excel": """
【Excel格式标准】(5分制):
- 5分: 有表头、公式、函数、多sheet
- 4分: 有表头和基本公式
- 3分: 有基本数据
- 2分: 数据混乱
- 1分: 无有效数据
""",
        "纯文本": """
【纯文本格式标准】(5分制):
- 5分: 文本清晰易读:
  * 段落分明
  * 无乱码
  * 适当换行
  * 层次清晰
- 4分: 基本清晰，有小问题
- 3分: 可读性一般
- 2分: 难以阅读
- 1分: 混乱无结构
"""
    }
    
    SCORING_RULES = """
========================================
评分规则说明
========================================

本评估系统从5个维度对Agent输出进行评分：

【内容质量】(权重30%)
- 5分: 输出内容完全符合任务要求，信息准确完整，逻辑清晰
- 4分: 内容基本符合要求，有少量遗漏或不够详细
- 3分: 内容基本满足任务要求，但有部分不准确或不完整
- 2分: 内容与任务要求有较大偏差
- 1分: 内容完全偏离任务要求

【内容深度与完整性】(权重25%)
根据不同文档类型有不同的评估标准

【格式符合度】(权重20%)
- 5分: 完全按照约定的输出格式要求，无任何偏差
- 4分: 格式基本正确，有轻微不符合
- 3分: 格式部分符合要求
- 2分: 格式与要求有较大偏差
- 1分: 完全不符合约定格式

【工具使用】(权重15%)
- 5分: 正确选择并使用了所有必要的工具
- 4分: 工具选择基本正确，使用效果良好
- 3分: 工具选择和使用基本合理
- 2分: 工具选择不当或使用效果不佳
- 1分: 未正确使用工具

【创意性】(权重10%)
- 5分: 输出非常有创意，超出预期
- 4分: 输出有一定创意
- 3分: 输出中规中矩
- 2分: 创意较少
- 1分: 缺乏创意

最终得分 = 内容质量×0.3 + 深度完整性×0.25 + 格式符合度×0.2 + 工具使用×0.15 + 创意性×0.10
========================================
"""
    
    @staticmethod
    def _analyze_markdown_depth(output: str) -> tuple:
        char_count = len(output)
        lines = output.strip().split('\n')
        line_count = len(lines)
        heading_count = len(re.findall(r'^#{1,6}\s+', output, re.MULTILINE))
        paragraph_count = len([p for p in output.split('\n\n') if p.strip()])
        
        depth_info = f"字数: {char_count}, 行数: {line_count}, 标题数: {heading_count}, 段落数: {paragraph_count}"
        
        if char_count > 5000:
            return 5, f"内容极其详尽({depth_info})"
        elif char_count > 3000:
            return 4, f"内容详尽({depth_info})"
        elif char_count > 1500:
            return 3, f"内容基本完整({depth_info})"
        elif char_count > 800:
            return 2, f"内容偏少({depth_info})"
        else:
            return 1, f"内容严重不足({depth_info})"
    
    @staticmethod
    def _analyze_ppt_depth(output: str) -> tuple:
        slide_markers = re.findall(r'Slide \d+|第\d+页|第\d+张|===|---', output, re.IGNORECASE)
        slide_count = len(slide_markers)
        if slide_count == 0:
            slide_count = len(re.findall(r'^#{1,6}\s+', output, re.MULTILINE))
        
        bullet_points = len(re.findall(r'^[\s]*[-*\d]+\.', output, re.MULTILINE))
        
        depth_info = f"页数: {slide_count}, 要点数: {bullet_points}"
        
        if slide_count > 15:
            return 5, f"PPT内容极其充实({depth_info})"
        elif slide_count > 10:
            return 4, f"PPT内容较充实({depth_info})"
        elif slide_count > 6:
            return 3, f"PPT内容基本完整({depth_info})"
        elif slide_count > 3:
            return 2, f"PPT内容偏少({depth_info})"
        else:
            return 1, f"PPT内容严重不足({depth_info})"
    
    @staticmethod
    def _analyze_word_depth(output: str) -> tuple:
        char_count = len(output)
        has_toc = bool(re.search(r'目录|Table of Contents', output))
        has_conclusion = bool(re.search(r'结论|总结|总结语', output))
        
        depth_info = f"字数: {char_count}, 有目录: {'是' if has_toc else '否'}, 有结论: {'是' if has_conclusion else '否'}"
        
        if char_count > 8000:
            return 5, f"Word文档极其详尽({depth_info})"
        elif char_count > 5000:
            return 4, f"Word文档详尽({depth_info})"
        elif char_count > 3000:
            return 3, f"Word文档基本完整({depth_info})"
        elif char_count > 1500:
            return 2, f"Word文档内容偏少({depth_info})"
        else:
            return 1, f"Word文档内容严重不足({depth_info})"
    
    @staticmethod
    def _analyze_excel_depth(output: str) -> tuple:
        sheet_markers = re.findall(r'Sheet\d+|Sheet \d+|\[[\w\s]+\]', output)
        sheet_count = max(len(sheet_markers), 1)
        
        table_pattern = re.findall(r'\|\s*\w+', output)
        row_count = len(table_pattern)
        
        depth_info = f"Sheet数: {sheet_count}, 表格行数: {row_count}"
        
        if sheet_count > 10:
            return 5, f"Excel内容极其丰富({depth_info})"
        elif sheet_count > 5:
            return 4, f"Excel内容较丰富({depth_info})"
        elif sheet_count > 2:
            return 3, f"Excel内容基本完整({depth_info})"
        elif sheet_count > 1:
            return 2, f"Excel内容偏少({depth_info})"
        else:
            return 1, f"Excel内容严重不足({depth_info})"
    
    @staticmethod
    def _analyze_json_depth(output: str) -> tuple:
        try:
            data = json.loads(output)
            key_count = Evaluator._count_json_keys(data)
            
            depth_info = f"键值对数: {key_count}"
            
            if key_count > 50:
                return 5, f"JSON数据极其丰富({depth_info})"
            elif key_count > 20:
                return 4, f"JSON数据较丰富({depth_info})"
            elif key_count > 10:
                return 3, f"JSON数据基本完整({depth_info})"
            elif key_count > 3:
                return 2, f"JSON数据偏少({depth_info})"
            else:
                return 1, f"JSON数据严重不足({depth_info})"
        except:
            return 2, "JSON格式无效，无法评估"
    
    @staticmethod
    def _count_json_keys(obj, count=0):
        if isinstance(obj, dict):
            count += len(obj)
            for v in obj.values():
                count = Evaluator._count_json_keys(v, count)
        elif isinstance(obj, list):
            for item in obj:
                count = Evaluator._count_json_keys(item, count)
        return count
    
    @staticmethod
    def _analyze_yaml_depth(output: str) -> tuple:
        try:
            data = yaml.safe_load(output)
            key_count = Evaluator._count_json_keys(data)
            
            depth_info = f"配置项数: {key_count}"
            
            if key_count > 30:
                return 5, f"YAML配置极其完整({depth_info})"
            elif key_count > 15:
                return 4, f"YAML配置较完整({depth_info})"
            elif key_count > 8:
                return 3, f"YAML配置基本完整({depth_info})"
            elif key_count > 3:
                return 2, f"YAML配置偏少({depth_info})"
            else:
                return 1, f"YAML配置严重不足({depth_info})"
        except:
            return 2, "YAML格式无效，无法评估"
    
    @staticmethod
    def _analyze_csv_depth(output: str) -> tuple:
        lines = [l for l in output.strip().split('\n') if l.strip()]
        row_count = len(lines)
        
        depth_info = f"数据行数: {row_count}"
        
        if row_count > 1000:
            return 5, f"CSV数据极其丰富({depth_info})"
        elif row_count > 500:
            return 4, f"CSV数据较丰富({depth_info})"
        elif row_count > 100:
            return 3, f"CSV数据基本完整({depth_info})"
        elif row_count > 20:
            return 2, f"CSV数据偏少({depth_info})"
        else:
            return 1, f"CSV数据严重不足({depth_info})"
    
    @staticmethod
    def _analyze_text_depth(output: str) -> tuple:
        char_count = len(output)
        line_count = len(output.strip().split('\n'))
        
        depth_info = f"字数: {char_count}, 行数: {line_count}"
        
        if char_count > 5000:
            return 5, f"内容极其详尽({depth_info})"
        elif char_count > 3000:
            return 4, f"内容详尽({depth_info})"
        elif char_count > 1500:
            return 3, f"内容基本完整({depth_info})"
        elif char_count > 800:
            return 2, f"内容偏少({depth_info})"
        else:
            return 1, f"内容严重不足({depth_info})"
    
    def __init__(self, eval_client: ModelClient, output_format: str):
        self.eval_client = eval_client
        self.output_format = output_format
        self.doc_type = self._detect_doc_type(output_format)
        self.format_type = self._detect_format_type(output_format)
    
    def _detect_doc_type(self, output_format: str) -> str:
        fmt_lower = output_format.lower()
        if "ppt" in fmt_lower or "powerpoint" in fmt_lower or "幻灯片" in fmt_lower or "演示" in fmt_lower:
            return "ppt"
        elif "word" in fmt_lower or "文档" in fmt_lower or "docx" in fmt_lower:
            return "word"
        elif "excel" in fmt_lower or "表格" in fmt_lower or "xlsx" in fmt_lower or "spreadsheet" in fmt_lower:
            return "excel"
        elif "markdown" in fmt_lower or "md" in fmt_lower:
            return "markdown"
        elif "json" in fmt_lower:
            return "json"
        elif "yaml" in fmt_lower or "yml" in fmt_lower:
            return "yaml"
        elif "csv" in fmt_lower:
            return "csv"
        elif "xml" in fmt_lower:
            return "xml"
        else:
            return "纯文本"
    
    def _detect_format_type(self, output_format: str) -> str:
        fmt_lower = output_format.lower()
        if "markdown" in fmt_lower or "md" in fmt_lower:
            return "markdown"
        elif "json" in fmt_lower:
            return "json"
        elif "yaml" in fmt_lower or "yml" in fmt_lower:
            return "yaml"
        elif "xml" in fmt_lower:
            return "xml"
        elif "csv" in fmt_lower:
            return "csv"
        else:
            return "纯文本"
    
    def _is_error_output(self, output: str) -> tuple:
        for pattern in self.ERROR_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return True, f"检测到执行错误: {pattern}"
        return False, ""
    
    def get_scoring_rules(self) -> str:
        rules = self.SCORING_RULES
        format_rule = self.FORMAT_RULES.get(self.format_type, "")
        if format_rule:
            rules += "\n" + format_rule
        
        doc_type_rule = self.DOC_TYPE_RULES.get(self.doc_type, {})
        if doc_type_rule.get("depth_criteria"):
            rules += "\n" + doc_type_rule["depth_criteria"]
        
        return rules
    
    def _validate_format(self, output: str) -> tuple:
        if self.format_type == "json":
            try:
                json.loads(output)
                return True, "JSON格式有效"
            except json.JSONDecodeError as e:
                return False, f"JSON格式错误: {str(e)}"
        
        elif self.format_type == "yaml":
            try:
                yaml.safe_load(output)
                return True, "YAML格式有效"
            except yaml.YAMLError as e:
                return False, f"YAML格式错误: {str(e)}"
        
        elif self.format_type == "xml":
            if re.search(r'<\?xml', output) and re.search(r'</\w+>', output):
                return True, "XML格式基本有效"
            return False, "XML格式可能无效"
        
        elif self.format_type == "markdown":
            has_headers = bool(re.search(r'^#{1,6}\s+', output, re.MULTILINE))
            has_lists = bool(re.search(r'^[\s]*[-*\d]+\.', output, re.MULTILINE))
            has_code = bool(re.search(r'```', output))
            score = sum([has_headers, has_lists, has_code])
            if score >= 2:
                return True, f"Markdown格式良好({score}/3元素)"
            return False, f"Markdown元素较少({score}/3)"
        
        elif self.format_type == "csv":
            lines = output.strip().split('\n')
            if len(lines) < 2:
                return False, "CSV数据不足"
            col_counts = [len(line.split(',')) for line in lines]
            if len(set(col_counts)) == 1:
                return True, "CSV格式正确"
            return False, f"列数不一致: {col_counts}"
        
        return True, "格式检查完成"
    
    def evaluate(self, task: str, output: str) -> EvaluationResult:
        is_error, error_msg = self._is_error_output(output)
        
        if is_error:
            return EvaluationResult(
                content_quality=1,
                format_compliance=1,
                tool_usage=1,
                creativity=1,
                depth_completeness=1,
                final_score=1.0,
                feedback=f"Agent执行失败，{error_msg}。输出内容: {output[:200]}",
                content_quality_reason="执行错误，无法评估",
                format_compliance_reason="执行错误，无法评估",
                tool_usage_reason="执行错误，无法评估",
                creativity_reason="执行错误，无法评估",
                depth_completeness_reason="执行错误，无法评估"
            )
        
        format_valid, format_msg = self._validate_format(output)
        
        format_compliance_score = 5 if format_valid else 2
        if not format_valid:
            format_msg = f"格式检查: {format_msg}"
        else:
            format_msg = f"格式检查通过: {format_msg}"
        
        doc_type_info = self.DOC_TYPE_RULES.get(self.doc_type, {})
        analyze_func = doc_type_info.get("analyze", lambda o: Evaluator._analyze_text_depth(o))
        depth_score, depth_msg = analyze_func(output)
        
        system = f"""你是一个严格的评委。根据任务和输出进行多维度评分。
输出必须是JSON格式：
{{
  "content_quality": 1-5分,
  "content_quality_reason": "详细说明为什么给这个分数",
  "depth_compliance": 1-5分,
  "depth_completeness_reason": "详细说明为什么给这个分数，结合字数/页数等指标",
  "format_compliance": 1-5分,
  "format_compliance_reason": "详细说明为什么给这个分数",
  "tool_usage": 1-5分,
  "tool_usage_reason": "详细说明为什么给这个分数",
  "creativity": 1-5分,
  "creativity_reason": "详细说明为什么给这个分数",
  "feedback": "总体改进建议"
}}

文档类型: {doc_type_info.get('name', '纯文本')}
任务：{task}
格式检查结果: {format_msg}
内容深度分析: {depth_msg}

评分标准：
- 内容质量(30%): 评估内容与任务的相关性、准确性
- 内容深度与完整性(25%): {doc_type_info.get('depth_criteria', '评估文档篇幅是否充足')}
- 格式符合度(20%): 评估是否严格遵循约定的输出格式
- 工具使用(15%): 评估工具选择和调用效果
- 创意性(10%): 评估输出的创新性

每个维度的reason必须具体说明：
1. 哪些地方做得好
2. 哪些地方需要改进
3. 给出具体的分数依据"""
        
        user = f"""输出内容：
{output}

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
                depth_completeness = clamp(data.get('depth_completeness', depth_score))
                format_compliance = clamp(data.get('format_compliance', format_compliance_score))
                tool_usage = clamp(data.get('tool_usage', 3))
                creativity = clamp(data.get('creativity', 3))
                
                final_score = (
                    content_quality * 0.3 +
                    depth_completeness * 0.25 +
                    format_compliance * 0.2 +
                    tool_usage * 0.15 +
                    creativity * 0.1
                )
                
                return EvaluationResult(
                    content_quality=content_quality,
                    format_compliance=format_compliance,
                    tool_usage=tool_usage,
                    creativity=creativity,
                    depth_completeness=depth_completeness,
                    final_score=round(final_score, 2),
                    feedback=data.get('feedback', '评估完成'),
                    content_quality_reason=data.get('content_quality_reason', ''),
                    format_compliance_reason=data.get('format_compliance_reason', format_msg),
                    tool_usage_reason=data.get('tool_usage_reason', ''),
                    creativity_reason=data.get('creativity_reason', ''),
                    depth_completeness_reason=data.get('depth_completeness_reason', depth_msg)
                )
        except Exception as e:
            pass
        
        return EvaluationResult(
            content_quality=3,
            format_compliance=format_compliance_score,
            tool_usage=3,
            creativity=3,
            depth_completeness=depth_score,
            final_score=(
                3 * 0.3 +
                depth_score * 0.25 +
                format_compliance_score * 0.2 +
                3 * 0.15 +
                3 * 0.1
            ),
            feedback="评分解析失败，使用默认评分",
            content_quality_reason="解析失败",
            format_compliance_reason=format_msg,
            tool_usage_reason="解析失败",
            creativity_reason="解析失败",
            depth_completeness_reason=depth_msg
        )
