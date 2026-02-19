from langchain_openai import ChatOpenAI
from typing import List, Optional, Any, Dict
from pydantic import Field

from src.core.model_client import ModelClient


def create_agent(
    system_prompt: str,
    tools: list,
    model_client: ModelClient,
    temperature: float = 0.7,
    max_iterations: int = 10,
    timeout: int = 60
):
    base_url = str(model_client.client.base_url)
    
    llm = ChatOpenAI(
        model=model_client.model_name,
        base_url=base_url,
        api_key='ollama',
        temperature=temperature
    )
    
    return SimpleAgentRunner(llm, system_prompt, tools, max_iterations, timeout)


class SimpleAgentRunner:
    def __init__(self, llm, system_prompt: str, tools: list, max_iterations: int = 10, timeout: int = 60):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_iterations = max_iterations
        self.timeout = timeout
    
    def invoke(self, inputs: dict) -> dict:
        try:
            user_content = ""
            if "input" in inputs:
                user_content = inputs["input"]
            elif "messages" in inputs:
                messages = inputs["messages"]
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, dict):
                        user_content = last_msg.get("content", "")
                    elif hasattr(last_msg, 'content'):
                        user_content = last_msg.content
            
            full_prompt = f"{self.system_prompt}\n\n用户请求：{user_content}"
            
            if self.tools:
                tools_desc = self._format_tools_description()
                full_prompt = f"{full_prompt}\n\n可用工具：\n{tools_desc}\n\n如果需要使用工具，请在回复中说明。"
            
            response = self.llm.invoke(full_prompt)
            
            if hasattr(response, 'content'):
                return {"output": response.content}
            else:
                return {"output": str(response)}
                
        except Exception as e:
            return {"output": f"Agent执行错误: {str(e)}"}
    
    def _format_tools_description(self) -> str:
        descriptions = []
        for tool in self.tools:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                descriptions.append(f"- {tool.name}: {tool.description}")
            elif isinstance(tool, dict):
                name = tool.get('name', '未知工具')
                desc = tool.get('description', '无描述')
                descriptions.append(f"- {name}: {desc}")
        return "\n".join(descriptions)


def create_agent_with_openai(
    system_prompt: str,
    tools: list,
    api_key: str,
    model_name: str = "gpt-4",
    api_base: Optional[str] = None,
    temperature: float = 0.7,
    max_iterations: int = 10,
    timeout: int = 60
):
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=api_base,
        temperature=temperature
    )
    
    return SimpleAgentRunner(llm, system_prompt, tools, max_iterations, timeout)
