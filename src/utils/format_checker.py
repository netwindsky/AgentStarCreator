import json
from typing import Optional


class FormatChecker:
    def check(self, output: str, format_type: str) -> str:
        format_type = format_type.lower()
        
        if 'json' in format_type:
            return self._check_json(output)
        elif 'yaml' in format_type:
            return self._check_yaml(output)
        elif 'xml' in format_type:
            return self._check_xml(output)
        elif 'markdown' in format_type:
            return self._check_markdown(output)
        elif 'csv' in format_type:
            return self._check_csv(output)
        else:
            return "跳过检查"
    
    def _check_json(self, output: str) -> str:
        try:
            json.loads(output)
            return "通过"
        except json.JSONDecodeError as e:
            return f"失败：无效的JSON格式 - {str(e)}"
        except Exception as e:
            return f"失败：{str(e)}"
    
    def _check_yaml(self, output: str) -> str:
        try:
            import yaml
            yaml.safe_load(output)
            return "通过"
        except ImportError:
            return "跳过：未安装PyYAML"
        except yaml.YAMLError as e:
            return f"失败：无效的YAML格式 - {str(e)}"
        except Exception as e:
            return f"失败：{str(e)}"
    
    def _check_xml(self, output: str) -> str:
        import xml.etree.ElementTree as ET
        try:
            ET.fromstring(output)
            return "通过"
        except ET.ParseError as e:
            return f"失败：无效的XML格式 - {str(e)}"
        except Exception as e:
            return f"失败：{str(e)}"
    
    def _check_markdown(self, output: str) -> str:
        markdown_indicators = ['#', '*', '-', '_', '`', '[', ']', '(', ')']
        has_markdown = any(indicator in output for indicator in markdown_indicators)
        
        if has_markdown:
            return "通过"
        return "警告：未检测到Markdown特征"
    
    def _check_csv(self, output: str) -> str:
        lines = output.strip().split('\n')
        if len(lines) < 1:
            return "失败：CSV内容为空"
        
        first_line_commas = lines[0].count(',')
        if first_line_commas == 0:
            return "警告：未检测到逗号分隔符"
        
        for i, line in enumerate(lines[1:], 2):
            if line.count(',') != first_line_commas:
                return f"失败：第{i}行列数不一致"
        
        return "通过"
