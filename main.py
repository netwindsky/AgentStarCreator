"""
Agent Prompt优化系统 - 主程序入口
启动Streamlit Web界面
"""

import subprocess
import sys


def main():
    print("🤖 Agent Prompt优化系统")
    print("=" * 50)
    print("正在启动Streamlit服务...")
    print("请在浏览器中访问: http://localhost:8501")
    print("=" * 50)
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/app/main.py",
        "--server.port=8501",
        "--server.address=localhost"
    ])


if __name__ == '__main__':
    main()
