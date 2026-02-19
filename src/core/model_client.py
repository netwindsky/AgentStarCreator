from openai import OpenAI
from typing import Optional, Generator, Callable, List
import requests


def get_ollama_models(ollama_url: str = "http://localhost:11434") -> List[str]:
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return sorted(models)
        return []
    except Exception:
        return []


class ModelClient:
    def __init__(
        self,
        source: str = "ollama",
        model_name: str = "deepseek-r1:7b",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120
    ):
        self.source = source
        self.model_name = model_name
        self.timeout = timeout
        
        if source == "ollama":
            self.client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama',
                timeout=timeout
            )
        elif source == "api":
            self.client = OpenAI(
                base_url=api_base,
                api_key=api_key,
                timeout=timeout
            )
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def chat_stream(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        callback: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, None]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                if callback:
                    callback(content)
                yield content
    
    def chat_stream_collect(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        collected = []
        for content in self.chat_stream(messages, temperature, max_tokens, callback):
            collected.append(content)
        return ''.join(collected)
