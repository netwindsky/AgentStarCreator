from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class AgentStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Agent:
    id: Optional[int] = None
    role: str = ""
    user_requirement: str = ""
    output_format: str = ""
    created_at: Optional[datetime] = None
    final_prompt: Optional[str] = None
    status: str = AgentStatus.CREATED.value
    early_stop_patience: int = 3
    early_stop_threshold: float = 0.1


@dataclass
class Iteration:
    id: Optional[int] = None
    agent_id: int = 0
    iteration_number: int = 0
    prompt: str = ""
    avg_score: Optional[float] = None
    scores_detail: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class TaskResult:
    id: Optional[int] = None
    iteration_id: int = 0
    task_description: str = ""
    output: Optional[str] = None
    scores: Optional[str] = None
    final_score: Optional[float] = None
    feedback: Optional[str] = None
    format_check: Optional[str] = None
    error_log: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class Task:
    id: Optional[int] = None
    agent_id: int = 0
    task_description: str = ""
    difficulty: str = "medium"
    is_active: bool = True
    created_at: Optional[datetime] = None


@dataclass
class ModelConfig:
    id: Optional[int] = None
    agent_id: int = 0
    iteration_id: Optional[int] = None
    model_type: str = ""
    model_source: str = "ollama"
    model_name: str = ""
    api_endpoint: Optional[str] = None
    api_key_encrypted: Optional[str] = None
