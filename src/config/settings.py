import os
from dotenv import load_dotenv
from typing import Optional


load_dotenv()


class Settings:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "deepseek-r1:7b")
    
    DB_PATH: str = os.getenv("DB_PATH", "data/agents.db")
    
    AGENT_FILE_DIR: str = os.getenv("AGENT_FILE_DIR", "data/agent_files")
    
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID: Optional[str] = os.getenv("GOOGLE_CSE_ID")
    
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "5"))
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "4.5"))
    EARLY_STOP_PATIENCE: int = int(os.getenv("EARLY_STOP_PATIENCE", "3"))
    EARLY_STOP_THRESHOLD: float = float(os.getenv("EARLY_STOP_THRESHOLD", "0.1"))
    
    TASK_TIMEOUT: int = int(os.getenv("TASK_TIMEOUT", "60"))
    MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
    
    @classmethod
    def setup_file_dir(cls):
        os.makedirs(cls.AGENT_FILE_DIR, exist_ok=True)
        os.environ['AGENT_FILE_DIR'] = cls.AGENT_FILE_DIR


settings = Settings()
