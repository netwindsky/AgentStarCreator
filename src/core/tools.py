from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
import math
import subprocess
import tempfile
import os
from typing import Optional


class CalculatorInput(BaseModel):
    expression: str = Field(description="数学表达式，如 '2 + 3 * 4'")


@tool
def calculator(expression: str) -> str:
    """
    用于数学计算。输入一个数学表达式，返回计算结果。
    支持基本运算和数学函数（sin, cos, sqrt等）。
    """
    try:
        allowed_names = {
            k: v for k, v in math.__dict__.items()
            if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


class PythonREPLInput(BaseModel):
    code: str = Field(description="要执行的Python代码")


class SafePythonREPLTool(BaseTool):
    name: str = "python_repl"
    description: str = """
    安全执行Python代码。输入Python代码字符串，返回执行结果。
    注意：代码在受限环境中运行，有以下限制：
    - 不允许文件操作（使用file_read/file_write工具）
    - 不允许网络请求
    - 不允许导入大部分模块
    - 执行超时限制为30秒
    """
    args_schema: type[BaseModel] = PythonREPLInput
    
    def _run(self, code: str) -> str:
        return self._execute_safely(code)
    
    async def _arun(self, code: str) -> str:
        return self._execute_safely(code)
    
    def _execute_safely(self, code: str) -> str:
        safe_builtins = {
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'bool': bool,
            'isinstance': isinstance,
            'type': type,
        }
        
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False
            ) as f:
                f.write(code)
                temp_path = f.name
            
            result = subprocess.run(
                ['python', '-c', f'''
import sys
safe_builtins = {safe_builtins}
__builtins__ = safe_builtins
exec(open("{temp_path}").read())
'''],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tempfile.gettempdir()
            )
            
            os.unlink(temp_path)
            
            if result.returncode == 0:
                return result.stdout or "执行成功（无输出）"
            else:
                return f"执行错误：{result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "执行超时（30秒限制）"
        except Exception as e:
            return f"执行错误：{str(e)}"


class FileReadInput(BaseModel):
    filepath: str = Field(description="要读取的文件路径")


@tool
def file_read(filepath: str) -> str:
    """
    读取文件内容。输入文件路径，返回文件内容。
    注意：只能读取允许目录下的文件。
    """
    allowed_dir = os.environ.get('AGENT_FILE_DIR', tempfile.gettempdir())
    
    abs_path = os.path.abspath(filepath)
    if not abs_path.startswith(os.path.abspath(allowed_dir)):
        return f"错误：只能读取 {allowed_dir} 目录下的文件"
    
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取失败：{e}"


class FileWriteInput(BaseModel):
    filepath: str = Field(description="要写入的文件路径")
    content: str = Field(description="要写入的内容")


@tool  
def file_write(filepath: str, content: str) -> str:
    """
    写入文件。输入文件路径和内容，返回操作结果。
    注意：只能写入允许目录下的文件，会覆盖已有文件。
    """
    allowed_dir = os.environ.get('AGENT_FILE_DIR', tempfile.gettempdir())
    
    abs_path = os.path.abspath(filepath)
    if not abs_path.startswith(os.path.abspath(allowed_dir)):
        return f"错误：只能写入 {allowed_dir} 目录"
    
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"已写入 {filepath}"
    except Exception as e:
        return f"写入失败：{e}"


def get_tools(search_api_key: Optional[str] = None, search_cse_id: Optional[str] = None):
    tools = [
        calculator,
        SafePythonREPLTool(),
        file_read,
        file_write,
    ]
    
    if search_api_key and search_cse_id:
        from langchain_community.utilities import GoogleSearchAPIWrapper
        from langchain_core.tools import Tool
        
        search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=search_api_key,
            google_cse_id=search_cse_id
        )
        search_tool = Tool(
            name="google_search",
            description="搜索网络信息。输入搜索关键词，返回相关结果。",
            func=search_wrapper.run
        )
        tools.append(search_tool)
    
    return tools
