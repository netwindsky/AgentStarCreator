import pytest
from src.core.tools import calculator, SafePythonREPLTool, file_read, file_write
from src.utils.format_checker import FormatChecker


class TestCalculator:
    def test_basic_calculation(self):
        result = calculator.invoke("2 + 3")
        assert result == "5"
    
    def test_multiplication(self):
        result = calculator.invoke("2 * 3 * 4")
        assert result == "24"
    
    def test_math_function(self):
        result = calculator.invoke("sqrt(16)")
        assert "4" in result
    
    def test_invalid_expression(self):
        result = calculator.invoke("invalid expression")
        assert "错误" in result


class TestSafePythonREPL:
    def test_simple_code(self):
        tool = SafePythonREPLTool()
        result = tool._run("print('hello')")
        assert "hello" in result
    
    def test_list_operations(self):
        tool = SafePythonREPLTool()
        result = tool._run("x = [1, 2, 3]; print(sum(x))")
        assert "6" in result
    
    def test_timeout(self):
        tool = SafePythonREPLTool()
        result = tool._run("while True: pass")
        assert "超时" in result


class TestFormatChecker:
    def test_json_valid(self):
        checker = FormatChecker()
        result = checker.check('{"key": "value"}', "json")
        assert result == "通过"
    
    def test_json_invalid(self):
        checker = FormatChecker()
        result = checker.check('{key: value}', "json")
        assert "失败" in result
    
    def test_markdown_valid(self):
        checker = FormatChecker()
        result = checker.check('# Title\n\nContent', "markdown")
        assert result == "通过"
    
    def test_xml_valid(self):
        checker = FormatChecker()
        result = checker.check('<root><item>test</item></root>', "xml")
        assert result == "通过"
    
    def test_unknown_format(self):
        checker = FormatChecker()
        result = checker.check('any content', "unknown")
        assert result == "跳过检查"
