import sqlite3
import os
from typing import Optional, List, Dict, Any


DB_PATH = "data/agents.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    user_requirement TEXT NOT NULL,
    output_format TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    final_prompt TEXT,
    status TEXT DEFAULT 'created',
    early_stop_patience INTEGER DEFAULT 3,
    early_stop_threshold REAL DEFAULT 0.1
);

CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL,
    iteration_number INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    avg_score REAL,
    scores_detail TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS task_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER NOT NULL,
    task_description TEXT NOT NULL,
    output TEXT,
    scores TEXT,
    final_score INTEGER,
    feedback TEXT,
    format_check TEXT,
    error_log TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (iteration_id) REFERENCES iterations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL,
    task_description TEXT NOT NULL,
    difficulty TEXT DEFAULT 'medium',
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL,
    iteration_id INTEGER,
    model_type TEXT NOT NULL,
    model_source TEXT NOT NULL,
    model_name TEXT NOT NULL,
    api_endpoint TEXT,
    api_key_encrypted TEXT,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE,
    FOREIGN KEY (iteration_id) REFERENCES iterations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_iterations_agent ON iterations(agent_id);
CREATE INDEX IF NOT EXISTS idx_iterations_number ON iterations(agent_id, iteration_number);
CREATE INDEX IF NOT EXISTS idx_task_results_iteration ON task_results(iteration_id);
CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks(agent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_active ON tasks(agent_id, is_active);
CREATE INDEX IF NOT EXISTS idx_model_configs_agent ON model_configs(agent_id);
"""


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_db()
    
    def _ensure_db(self):
        init_db(self.db_path)
    
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def create_agent(
        self,
        role: str,
        user_requirement: str,
        output_format: str,
        early_stop_patience: int = 3,
        early_stop_threshold: float = 0.1
    ) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO agents 
               (role, user_requirement, output_format, early_stop_patience, early_stop_threshold)
               VALUES (?, ?, ?, ?, ?)""",
            (role, user_requirement, output_format, early_stop_patience, early_stop_threshold)
        )
        agent_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return agent_id
    
    def get_agent(self, agent_id: int) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def update_agent_status(self, agent_id: int, status: str, final_prompt: Optional[str] = None):
        conn = self._get_conn()
        cursor = conn.cursor()
        if final_prompt:
            cursor.execute(
                "UPDATE agents SET status = ?, final_prompt = ? WHERE id = ?",
                (status, final_prompt, agent_id)
            )
        else:
            cursor.execute(
                "UPDATE agents SET status = ? WHERE id = ?",
                (status, agent_id)
            )
        conn.commit()
        conn.close()
    
    def list_agents(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM agents ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def add_model_config(
        self,
        agent_id: int,
        model_type: str,
        model_source: str,
        model_name: str,
        api_endpoint: Optional[str] = None,
        api_key_encrypted: Optional[str] = None
    ):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO model_configs
               (agent_id, model_type, model_source, model_name, api_endpoint, api_key_encrypted)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (agent_id, model_type, model_source, model_name, api_endpoint, api_key_encrypted)
        )
        conn.commit()
        conn.close()
    
    def get_model_configs(self, agent_id: int) -> Dict[str, Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM model_configs WHERE agent_id = ? AND iteration_id IS NULL",
            (agent_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return {row['model_type']: dict(row) for row in rows}
    
    def add_task(self, agent_id: int, task_description: str, difficulty: str = 'medium') -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tasks (agent_id, task_description, difficulty) VALUES (?, ?, ?)",
            (agent_id, task_description, difficulty)
        )
        task_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return task_id
    
    def get_active_tasks(self, agent_id: int) -> List[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT task_description FROM tasks WHERE agent_id = ? AND is_active = 1",
            (agent_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [row['task_description'] for row in rows]
    
    def deactivate_tasks(self, agent_id: int):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tasks SET is_active = 0 WHERE agent_id = ?",
            (agent_id,)
        )
        conn.commit()
        conn.close()
    
    def get_iterations(self, agent_id: int) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM iterations WHERE agent_id = ? ORDER BY iteration_number",
            (agent_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_iteration_count(self, agent_id: int) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM iterations WHERE agent_id = ?",
            (agent_id,)
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_latest_prompt(self, agent_id: int) -> Optional[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT prompt FROM iterations WHERE agent_id = ? ORDER BY iteration_number DESC LIMIT 1",
            (agent_id,)
        )
        row = cursor.fetchone()
        conn.close()
        return row['prompt'] if row else None
