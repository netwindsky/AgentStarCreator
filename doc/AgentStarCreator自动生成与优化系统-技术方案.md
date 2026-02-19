好的，基于我们确认的需求文档，我将为您设计一个详细的技术方案。本方案将采用**LangChain**作为核心框架，**Streamlit**作为Web界面，**SQLite**作为数据库，并集成Ollama本地模型及外部API。以下是完整的设计。

---

# Agent自动生成与优化系统 - 技术方案

## 1. 系统架构概览

系统采用模块化设计，主要包含以下组件：

- **Web界面层**：Streamlit应用，负责用户交互、参数配置、过程可视化、人工介入。
- **核心控制层**：Python模块，协调整个优化流程（迭代循环、状态管理）。
- **模型接入层**：封装Ollama本地模型和外部API（如OpenAI），提供统一的调用接口。
- **工具层**：基于LangChain的Tool集合，包括搜索引擎、计算器、Python REPL、文件读写等，可动态扩展。
- **数据持久层**：SQLite数据库，存储所有Agent、迭代、任务结果等数据。
- **评估与优化模块**：独立的模型调用逻辑，用于评分和prompt改进。

系统流程：

1. 用户在Web界面输入角色、需求、输出格式、参数等，点击“开始”。
2. 系统生成初始prompt，并自动生成测试任务列表。
3. 进入迭代循环：
   a. 使用当前prompt实例化Agent，依次执行测试任务（调用工具）。
   b. 对每个任务的输出，调用评估模型评分并收集反馈。
   c. 计算平均分，保存本次迭代数据到数据库。
   d. 若平均分达到阈值或达到最大迭代次数，暂停并通知用户；否则，调用优化模型生成新prompt，继续下一轮。
4. 暂停后，用户可查看结果、修改prompt或任务，然后点击“继续”重新进入迭代循环。
5. 整个过程中，界面实时显示进度、输出和评分变化。

---

## 2. 数据库设计（SQLite）

### 表结构

```sql
-- 存储创建的Agent
CREATE TABLE agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,                -- 角色名称
    user_requirement TEXT NOT NULL,     -- 用户需求描述
    output_format TEXT NOT NULL,        -- 输出格式约定（如JSON, Markdown等）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    final_prompt TEXT,                  -- 最终采用的prompt（可选）
    status TEXT DEFAULT 'created'       -- created, running, paused, completed
);

-- 存储每次迭代
CREATE TABLE iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL,
    iteration_number INTEGER NOT NULL,
    prompt TEXT NOT NULL,                -- 本次迭代使用的prompt
    avg_score REAL,                      -- 平均分
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

-- 存储每次迭代中的任务结果
CREATE TABLE task_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER NOT NULL,
    task_description TEXT NOT NULL,
    output TEXT,                          -- Agent的输出
    score INTEGER,                        -- 评分（1-5）
    feedback TEXT,                        -- 评估反馈
    format_check TEXT,                    -- 格式检查结果（可选）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (iteration_id) REFERENCES iterations(id) ON DELETE CASCADE
);

-- 存储每次迭代的测试任务列表（可复用或记录）
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL,
    task_description TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1,           -- 是否在当前测试集中
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

-- 存储模型配置（可选，用于记录每次使用的模型）
CREATE TABLE model_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL,
    iteration_id INTEGER,                   -- 可为空，表示全局配置
    model_type TEXT NOT NULL,                -- 'base', 'evaluator', 'optimizer', 'task_generator'
    model_source TEXT NOT NULL,               -- 'ollama' 或 'api'
    model_name TEXT NOT NULL,                 -- 模型名称（如 deepseek-r1:7b）
    api_endpoint TEXT,                        -- 如果使用外部API
    api_key TEXT,                             -- 加密存储？简单起见可明文或由用户输入
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE,
    FOREIGN KEY (iteration_id) REFERENCES iterations(id) ON DELETE CASCADE
);
```

### 说明
- `agents`表记录每个创建的Agent，包括最终确定的prompt。
- `iterations`表记录每次迭代的prompt和平均分。
- `task_results`表详细记录每个任务的执行结果，便于回溯。
- `tasks`表存储该Agent关联的测试任务，可动态增删。
- `model_configs`用于记录每次使用的模型，便于复现和分析。

---

## 3. 核心模块设计

### 3.1 模型调用封装

创建一个统一的模型调用类 `ModelClient`，支持Ollama和外部API（OpenAI兼容格式）。

```python
# model_client.py
import requests
from openai import OpenAI
from typing import Optional, Dict, Any

class ModelClient:
    def __init__(self, source: str = "ollama", model_name: str = "deepseek-r1:7b",
                 api_base: Optional[str] = None, api_key: Optional[str] = None):
        self.source = source
        self.model_name = model_name
        if source == "ollama":
            self.client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        elif source == "api":
            # 假设是OpenAI兼容格式
            self.client = OpenAI(base_url=api_base, api_key=api_key)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def chat(self, messages: list, temperature: float = 0.7, stream: bool = False) -> str:
        """发送聊天请求，返回内容字符串。若stream=True，则返回完整内容（非流式）。"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            stream=stream
        )
        if stream:
            # 处理流式输出（用于实时显示）
            collected = []
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected.append(content)
                    # 这里可以通过回调或yield实现实时输出
            return ''.join(collected)
        else:
            return response.choices[0].message.content
```

**说明**：为支持实时输出，可以在调用时传入一个回调函数来处理每个chunk。

### 3.2 工具集成（LangChain Tools）

定义一组工具，使用LangChain的`Tool`类封装。每个工具需实现`_run`方法。

```python
# tools.py
from langchain.tools import BaseTool, StructuredTool
from langchain.utilities import GoogleSearchAPIWrapper  # 示例
import math
import subprocess  # 注意安全
import os

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "用于数学计算，输入一个数学表达式，返回计算结果。"

    def _run(self, query: str) -> str:
        try:
            # 安全评估，仅允许基本运算
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            allowed_names.update({"abs": abs, "round": round})
            result = eval(query, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"计算错误：{e}"

    async def _arun(self, query: str):
        raise NotImplementedError

class PythonREPLTool(BaseTool):
    name = "python_repl"
    description = "执行Python代码，输入代码字符串，返回执行结果。注意：代码会在沙箱中运行，但仍有风险。"

    def _run(self, code: str) -> str:
        # 此处应使用安全沙箱，如PyPy沙箱、或限制模块
        # 简单示例：使用exec捕获输出，但极不安全，实际需用RestrictedPython或容器
        try:
            local_vars = {}
            exec(code, {"__builtins__": {}}, local_vars)
            return str(local_vars.get('result', '执行完成（无返回值）'))
        except Exception as e:
            return f"执行错误：{e}"

# 搜索引擎工具（需用户提供API密钥）
class GoogleSearchTool(BaseTool):
    name = "google_search"
    description = "搜索网络信息，输入搜索关键词，返回前几条结果。"

    def __init__(self, api_key: str, cse_id: str):
        super().__init__()
        self.search = GoogleSearchAPIWrapper(google_api_key=api_key, google_cse_id=cse_id)

    def _run(self, query: str) -> str:
        try:
            results = self.search.results(query, num_results=3)
            return "\n".join([f"{r['title']}: {r['snippet']}" for r in results])
        except Exception as e:
            return f"搜索失败：{e}"

# 文件读写工具
class FileReadTool(BaseTool):
    name = "file_read"
    description = "读取文件内容，输入文件路径，返回文件内容。"

    def _run(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"读取失败：{e}"

class FileWriteTool(BaseTool):
    name = "file_write"
    description = "写入文件，输入格式：文件路径|内容。注意：会覆盖已有文件。"

    def _run(self, input_str: str) -> str:
        try:
            path, content = input_str.split('|', 1)
            with open(path.strip(), 'w', encoding='utf-8') as f:
                f.write(content)
            return f"已写入 {path}"
        except Exception as e:
            return f"写入失败：{e}"
```

**安全提示**：Python REPL和文件读写存在安全风险，后续需强化沙箱机制（如使用`exec`受限环境、Docker容器等）。此处仅作概念设计。

### 3.3 Agent工厂

基于LangChain的`initialize_agent`或`AgentExecutor`创建Agent。我们使用`ZeroShotAgent`，并配置工具。

```python
# agent_factory.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.llms import OllamaLLM  # 但我们使用自定义ModelClient适配LangChain的LLM

# 需要将ModelClient包装为LangChain的LLM类
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

class CustomLLM(LLM):
    client: ModelClient
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.client.chat(messages, temperature=self.temperature)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.client.model_name}

def create_agent(prompt: str, tools: list, model_client: ModelClient, temperature=0.7):
    """根据prompt和工具创建AgentExecutor"""
    llm = CustomLLM(client=model_client, temperature=temperature)
    # LangChain的Agent需要提示模板，我们将整个prompt作为系统消息的一部分
    # 可以使用ChatPromptTemplate，但为了简单，我们使用一个包含系统消息的PromptTemplate
    # 实际上，我们可以使用MessagesPlaceholder，但简化起见，我们用字符串模板
    template = """{system_message}

Tools: {tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""
    prompt_template = PromptTemplate.from_template(template)
    # 创建agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor
```

**注意**：上述方式需要将系统消息嵌入模板。更优雅的方式是使用`ChatPromptTemplate`和`MessagesPlaceholder`，但需要额外处理。为简化，我们采用字符串替换。

### 3.4 核心优化循环

定义`OptimizationLoop`类，管理一次创建任务的全部流程。

```python
# core.py
import sqlite3
from datetime import datetime
from typing import List, Dict, Any
import json

class OptimizationLoop:
    def __init__(self, agent_id: int, db_path: str = "agents.db"):
        self.agent_id = agent_id
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.load_agent_info()
        # 初始化模型客户端（从数据库配置加载，或使用默认）
        self.base_client = ModelClient(source="ollama", model_name=self.base_model)  # 需从配置获取
        self.eval_client = ModelClient(source="ollama", model_name=self.eval_model)
        self.optimizer_client = ModelClient(source="ollama", model_name=self.optimizer_model)
        self.task_gen_client = ModelClient(source="ollama", model_name=self.task_gen_model)
        # 工具列表（动态构建）
        self.tools = self._build_tools()

    def load_agent_info(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM agents WHERE id = ?", (self.agent_id,))
        self.agent = dict(cursor.fetchone())
        # 加载配置
        cursor.execute("SELECT * FROM model_configs WHERE agent_id = ? AND iteration_id IS NULL", (self.agent_id,))
        configs = cursor.fetchall()
        self.base_model = next((c['model_name'] for c in configs if c['model_type']=='base'), 'deepseek-r1:7b')
        self.eval_model = next((c['model_name'] for c in configs if c['model_type']=='evaluator'), 'deepseek-r1:7b')
        self.optimizer_model = next((c['model_name'] for c in configs if c['model_type']=='optimizer'), 'deepseek-r1:7b')
        self.task_gen_model = next((c['model_name'] for c in configs if c['model_type']=='task_generator'), 'deepseek-r1:7b')
        # 加载测试任务
        cursor.execute("SELECT task_description FROM tasks WHERE agent_id = ? AND is_active = 1", (self.agent_id,))
        self.tasks = [row['task_description'] for row in cursor.fetchall()]

    def _build_tools(self):
        tools = []
        # 根据用户配置的工具启用（这里简化，假设所有工具都启用）
        tools.append(CalculatorTool())
        # 如果需要搜索引擎，需从配置中读取API密钥
        # 此处应从model_configs或agent表中读取密钥，示例略
        # tools.append(GoogleSearchTool(api_key, cse_id))
        tools.append(PythonREPLTool())  # 注意安全
        tools.append(FileReadTool())
        tools.append(FileWriteTool())
        return tools

    def generate_initial_prompt(self):
        """生成初始prompt"""
        system = "你是一个专业的Prompt工程师。根据用户需求创建一个详细的Agent系统提示。"
        user = f"为{self.agent['role']}角色创建系统提示。需求：{self.agent['user_requirement']}。输出格式必须遵循：{self.agent['output_format']}。请包含角色描述、核心能力、工具使用说明、工作流程、输出格式要求。"
        prompt = self.base_client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
        return prompt

    def generate_tasks(self, num_tasks=3):
        """生成测试任务（如果任务表为空）"""
        if self.tasks:
            return
        system = "你是一个任务生成器。根据角色生成典型的测试任务。"
        user = f"角色：{self.agent['role']}。请生成{num_tasks}个测试任务，每个任务需明确描述，适合作为LLM的输入，并且能检验角色能力和输出格式（{self.agent['output_format']}）。"
        response = self.task_gen_client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
        # 假设模型返回每行一个任务
        tasks = [line.strip() for line in response.split('\n') if line.strip()]
        # 保存到数据库
        cursor = self.conn.cursor()
        for t in tasks[:num_tasks]:
            cursor.execute("INSERT INTO tasks (agent_id, task_description) VALUES (?, ?)", (self.agent_id, t))
        self.conn.commit()
        self.tasks = tasks[:num_tasks]

    def run_iteration(self, prompt: str, iteration_number: int) -> Dict[str, Any]:
        """执行一轮迭代：运行所有任务，评估，返回结果和平均分"""
        agent = create_agent(prompt, self.tools, self.base_client)
        results = []
        total_score = 0
        for task in self.tasks:
            # 执行任务
            output = agent.run(task)  # 同步执行
            # 评估
            score, feedback = self.evaluate(task, output)
            total_score += score
            results.append({
                "task": task,
                "output": output,
                "score": score,
                "feedback": feedback
            })
        avg_score = total_score / len(self.tasks)
        # 保存迭代和结果到数据库
        self.save_iteration(iteration_number, prompt, results, avg_score)
        return {"avg_score": avg_score, "results": results}

    def evaluate(self, task: str, output: str) -> tuple:
        """调用评估模型评分"""
        system = "你是一个严格的评委。根据任务和输出，按照标准评分（1-5分），并提供改进意见。输出JSON格式：{\"score\": 整数, \"feedback\": \"意见\"}"
        user = f"任务：{task}\n输出：{output}\n输出格式约定：{self.agent['output_format']}\n请给出分数和反馈。"
        response = self.eval_client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
        try:
            # 提取JSON
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get('score', 3), data.get('feedback', '')
        except:
            pass
        return 3, "评分解析失败"

    def improve_prompt(self, old_prompt: str, results: List[Dict], avg_score: float) -> str:
        """调用优化模型生成新prompt"""
        system = "你是一个Prompt优化专家。根据测试反馈修改prompt，只输出新prompt。"
        # 汇总反馈
        feedback_summary = "\n".join([f"任务：{r['task']}\n反馈：{r['feedback']}" for r in results])
        user = f"当前prompt：\n{old_prompt}\n\n平均分：{avg_score}\n反馈详情：\n{feedback_summary}\n\n请生成改进后的新prompt。"
        new_prompt = self.optimizer_client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
        return new_prompt

    def save_iteration(self, iteration_number: int, prompt: str, results: List[Dict], avg_score: float):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO iterations (agent_id, iteration_number, prompt, avg_score) VALUES (?, ?, ?, ?)",
            (self.agent_id, iteration_number, prompt, avg_score)
        )
        iter_id = cursor.lastrowid
        for r in results:
            cursor.execute(
                "INSERT INTO task_results (iteration_id, task_description, output, score, feedback) VALUES (?, ?, ?, ?, ?)",
                (iter_id, r['task'], r['output'], r['score'], r['feedback'])
            )
        self.conn.commit()

    def run_loop(self, max_iterations: int = 5, score_threshold: float = 4.5, 
                 on_iteration_complete=None, on_pause=None):
        """主循环，on_iteration_complete可用于实时更新界面"""
        # 获取当前迭代次数（从数据库）
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM iterations WHERE agent_id = ?", (self.agent_id,))
        start_iter = cursor.fetchone()['cnt']
        
        if start_iter == 0:
            # 首次运行，生成初始prompt和任务
            self.generate_tasks()
            current_prompt = self.generate_initial_prompt()
        else:
            # 获取最后一次迭代的prompt
            cursor.execute("SELECT prompt FROM iterations WHERE agent_id = ? ORDER BY iteration_number DESC LIMIT 1", (self.agent_id,))
            current_prompt = cursor.fetchone()['prompt']
        
        for i in range(start_iter, max_iterations):
            iter_num = i + 1
            # 运行迭代
            result = self.run_iteration(current_prompt, iter_num)
            avg_score = result['avg_score']
            # 回调（用于界面更新）
            if on_iteration_complete:
                on_iteration_complete(iter_num, result)
            
            if avg_score >= score_threshold:
                # 达到阈值，暂停
                if on_pause:
                    on_pause(iter_num, avg_score, result)
                break
            
            # 优化prompt
            current_prompt = self.improve_prompt(current_prompt, result['results'], avg_score)
        
        # 循环结束，更新agent状态
        cursor.execute("UPDATE agents SET status = ? WHERE id = ?", ('paused' if avg_score >= score_threshold else 'completed', self.agent_id))
        self.conn.commit()
```

**说明**：`on_iteration_complete`和`on_pause`是回调函数，用于在Streamlit中更新界面。

---

## 4. Web界面设计（Streamlit）

我们将构建一个多页面Streamlit应用，包含：
- 首页：创建新Agent
- 运行页面：显示优化过程
- 历史页面：查看已创建的Agent和迭代历史

### 4.1 文件结构

```
agent_builder/
├── app.py                 # 主入口
├── pages/
│   ├── 1_Create_Agent.py  # 创建新Agent
│   └── 2_History.py       # 历史记录
├── core.py                # 优化循环类
├── model_client.py        # 模型封装
├── tools.py               # 工具定义
├── agent_factory.py       # Agent创建
├── database.py            # 数据库初始化及操作
└── config.py              # 配置常量
```

### 4.2 主要界面设计

#### 4.2.1 创建新Agent页面（`1_Create_Agent.py`）

```python
import streamlit as st
import sqlite3
from core import OptimizationLoop
from database import init_db, create_agent

st.title("新建Agent优化任务")

with st.form("agent_form"):
    role = st.text_input("Agent角色", value="文案策划")
    requirement = st.text_area("需求描述", value="能够撰写吸引人的广告文案、产品描述等")
    output_format = st.selectbox("输出格式", ["Markdown", "JSON", "YAML", "XML", "纯文本", "自定义"])
    if output_format == "自定义":
        output_format = st.text_input("请输入自定义格式描述")
    
    col1, col2 = st.columns(2)
    with col1:
        max_iters = st.number_input("最大迭代次数", min_value=1, max_value=20, value=5)
    with col2:
        threshold = st.slider("评分阈值（达到后暂停）", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
    
    st.subheader("模型配置")
    # 获取Ollama模型列表（需调用ollama list命令）
    import subprocess
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')[1:]  # 跳过标题
    models = [line.split()[0] for line in lines if line]
    
    base_model = st.selectbox("基础模型（生成prompt、执行任务）", models, index=0)
    eval_model = st.selectbox("评估模型", models, index=0)
    optimizer_model = st.selectbox("优化模型", models, index=0)
    task_gen_model = st.selectbox("任务生成模型", models, index=0)
    
    # 外部API配置（可选）
    use_external = st.checkbox("使用外部API（如OpenAI）")
    if use_external:
        api_base = st.text_input("API Base URL", value="https://api.openai.com/v1")
        api_key = st.text_input("API Key", type="password")
        # 可选择对应的模型名称
        ext_model = st.text_input("外部模型名称", value="gpt-4")
    
    st.subheader("工具配置")
    enable_search = st.checkbox("启用搜索引擎", value=False)
    if enable_search:
        google_api_key = st.text_input("Google API Key", type="password")
        google_cse_id = st.text_input("Google CSE ID")
    enable_python = st.checkbox("启用Python REPL（注意安全）", value=True)
    enable_file = st.checkbox("启用文件读写", value=True)
    
    submitted = st.form_submit_button("开始优化")
    
    if submitted:
        # 保存到数据库
        conn = sqlite3.connect('agents.db')
        agent_id = create_agent(conn, role, requirement, output_format)
        # 保存模型配置
        cursor = conn.cursor()
        cursor.execute("INSERT INTO model_configs (agent_id, model_type, model_source, model_name) VALUES (?,?,?,?)",
                       (agent_id, 'base', 'ollama', base_model))
        cursor.execute("INSERT INTO model_configs (agent_id, model_type, model_source, model_name) VALUES (?,?,?,?)",
                       (agent_id, 'evaluator', 'ollama', eval_model))
        cursor.execute("INSERT INTO model_configs (agent_id, model_type, model_source, model_name) VALUES (?,?,?,?)",
                       (agent_id, 'optimizer', 'ollama', optimizer_model))
        cursor.execute("INSERT INTO model_configs (agent_id, model_type, model_source, model_name) VALUES (?,?,?,?)",
                       (agent_id, 'task_generator', 'ollama', task_gen_model))
        conn.commit()
        conn.close()
        
        # 将agent_id存入session_state，跳转到运行页面
        st.session_state['current_agent_id'] = agent_id
        st.session_state['max_iters'] = max_iters
        st.session_state['threshold'] = threshold
        st.switch_page("pages/2_Run_Optimization.py")  # 假设运行页面为2_Run_Optimization.py
```

#### 4.2.2 运行优化页面（`2_Run_Optimization.py`）

```python
import streamlit as st
import time
from core import OptimizationLoop
import pandas as pd
import matplotlib.pyplot as plt

st.title("优化进行中")

if 'current_agent_id' not in st.session_state:
    st.warning("请先在创建页面新建Agent")
    st.stop()

agent_id = st.session_state['current_agent_id']
max_iters = st.session_state.get('max_iters', 5)
threshold = st.session_state.get('threshold', 4.5)

# 初始化循环对象
loop = OptimizationLoop(agent_id)

# 创建占位符
status = st.empty()
progress_bar = st.progress(0)
iter_display = st.container()

# 定义回调函数
def on_iteration_complete(iter_num, result):
    status.write(f"迭代 {iter_num} 完成，平均分：{result['avg_score']:.2f}")
    progress_bar.progress(iter_num / max_iters)
    with iter_display:
        st.subheader(f"迭代 {iter_num}")
        st.write(f"平均分：{result['avg_score']:.2f}")
        for r in result['results']:
            st.markdown(f"**任务：** {r['task']}")
            st.text(f"输出：{r['output'][:200]}...")
            st.write(f"评分：{r['score']}，反馈：{r['feedback']}")
            st.divider()

def on_pause(iter_num, avg_score, result):
    status.write(f"达到阈值（{avg_score:.2f}），暂停。")
    st.session_state['paused'] = True
    st.session_state['last_result'] = result
    st.session_state['last_prompt'] = loop.get_current_prompt()  # 需要实现此方法

# 启动循环（需在后台线程或直接运行）
# 注意：Streamlit是同步的，长时间运行会阻塞界面。可以用线程或st.experimental_fragment
# 简单起见，我们直接运行，但界面会卡住直到完成。更好的做法是使用st.session_state和异步。
# 这里演示直接运行：

if st.button("开始/继续优化"):
    loop.run_loop(max_iterations=max_iters, score_threshold=threshold,
                  on_iteration_complete=on_iteration_complete,
                  on_pause=on_pause)
    st.success("优化完成！")

if st.session_state.get('paused'):
    st.info("人工介入模式")
    # 显示当前prompt，允许编辑
    new_prompt = st.text_area("编辑Prompt", value=st.session_state['last_prompt'], height=300)
    # 显示任务列表，允许增删
    tasks = loop.tasks
    st.write("当前测试任务：")
    for i, t in enumerate(tasks):
        col1, col2 = st.columns([5,1])
        col1.write(t)
        if col2.button("删除", key=f"del_{i}"):
            # 删除任务逻辑
            pass
    new_task = st.text_input("添加新任务")
    if st.button("添加任务"):
        # 添加任务逻辑
        pass
    
    if st.button("保存修改并继续"):
        # 更新prompt和任务到数据库
        loop.update_prompt_and_tasks(new_prompt, tasks)  # 需实现
        st.session_state['paused'] = False
        st.rerun()
```

### 4.3 实时输出实现

为在Streamlit中实现实时输出，可以将任务执行过程改为流式，并使用`st.write_stream`或自定义占位符不断追加内容。例如，在`run_iteration`中，可以收集输出并逐步更新。

由于篇幅，这里不展开所有细节，但原理是在执行Agent时，设置`verbose=True`并将输出重定向到Streamlit的容器。

---

## 5. 安全性考虑

- **Python REPL**：必须限制执行环境，例如使用`RestrictedPython`或运行在Docker容器中。
- **文件读写**：限制可访问的目录，避免系统文件被篡改。
- **API密钥**：建议加密存储或由用户每次输入，不保存在数据库明文。
- **模型调用**：对外部API调用需设置超时，避免无限等待。

---

## 6. 开发步骤建议

1. **环境搭建**：安装Python依赖（langchain, streamlit, openai, requests, matplotlib, pandas, sqlite3等）。
2. **数据库初始化**：编写`database.py`，创建表。
3. **模型客户端**：实现`ModelClient`，测试与Ollama的通信。
4. **工具实现**：逐个实现工具，测试其功能。
5. **Agent工厂**：实现`create_agent`，测试单个任务执行。
6. **核心循环**：实现`OptimizationLoop`，完成迭代、评估、优化逻辑，可先用简单prompt测试。
7. **Streamlit界面**：逐步构建页面，集成回调。
8. **测试与调试**：使用小型模型（如`tinyllama`）快速验证流程。
9. **优化与扩展**：添加更多工具、支持外部API、完善错误处理。

---

## 7. 扩展性说明

- **模型接入**：`ModelClient`设计为可插拔，未来可添加更多源（如Anthropic、Cohere）。
- **工具注册**：工具列表可从配置文件加载，动态导入。
- **评估标准**：可在评估提示词中配置不同的评分维度。
- **输出格式**：可增加格式验证器（如JSON schema校验）作为辅助。

---

收到您的反馈，我将进一步明确各个代码模块的**角色与职责**，确保设计清晰、职责单一，便于后续开发和维护。以下是补充的“组件角色与职责”部分，您可将其整合到技术方案中。

---

## 组件角色与职责（明确分工）

### 1. `ModelClient`（模型客户端）
- **角色**：所有与LLM交互的统一接口。
- **职责**：
  - 封装对Ollama本地模型和外部API（如OpenAI）的调用。
  - 提供统一的`chat`方法，支持同步和流式响应。
  - 管理模型名称、API密钥、基础URL等配置。
- **为什么独立**：将模型调用与核心业务逻辑解耦，方便后续增加新模型源或更换模型。

### 2. `Tool`系列（工具类）
- **角色**：Agent可调用的具体能力。
- **职责**：
  - 每个工具（如`CalculatorTool`、`GoogleSearchTool`）继承自LangChain的`BaseTool`，实现`_run`方法执行具体操作。
  - 工具内部处理错误、安全限制（如计算器仅允许数学函数）。
- **为什么独立**：符合单一职责原则，每种工具独立开发、测试，并可动态启用/禁用。

### 3. `CustomLLM`（LangChain适配器）
- **角色**：将我们的`ModelClient`适配为LangChain可用的LLM类。
- **职责**：
  - 继承LangChain的`LLM`基类，实现`_call`方法。
  - 将LangChain的提示词转换为`ModelClient`所需的`messages`格式。
- **为什么独立**：使LangChain的Agent能够使用我们自定义的模型客户端，保持框架兼容性。

### 4. `AgentFactory`（Agent工厂）
- **角色**：根据prompt、工具列表和模型客户端创建可执行的Agent。
- **职责**：
  - 接收系统提示（prompt）、工具列表、`ModelClient`实例。
  - 构建LangChain的`AgentExecutor`，配置提示模板和解析器。
  - 返回一个可直接运行任务的`AgentExecutor`对象。
- **为什么独立**：将Agent的创建过程封装起来，简化主循环中的调用。

### 5. `OptimizationLoop`（优化循环控制器）
- **角色**：整个自动优化流程的指挥中心。
- **职责**：
  - 加载Agent信息（角色、需求、输出格式、测试任务）和模型配置。
  - 管理迭代流程：生成初始prompt、运行任务、评估、优化prompt。
  - 与数据库交互，保存每次迭代的结果。
  - 提供回调接口，供UI层实时更新。
- **为什么独立**：集中控制优化逻辑，便于维护和测试。

### 6. `Database`模块（数据访问层）
- **角色**：所有数据库操作的封装。
- **职责**：
  - 初始化数据库表结构。
  - 提供插入、查询Agent、迭代、任务结果的方法。
  - 避免SQL语句散落在各处，统一管理。
- **为什么独立**：将数据持久化与业务逻辑分离，方便未来更换数据库。

### 7. `Streamlit Pages`（UI页面）
- **角色**：用户交互界面。
- **职责**：
  - `Create_Agent`页：收集用户输入（角色、需求、格式、模型、工具配置），创建Agent记录。
  - `Run_Optimization`页：展示优化过程，实时显示迭代结果，提供人工介入编辑功能。
  - `History`页：查看所有已创建的Agent及其历史迭代。
- **为什么独立**：遵循MVC思想，UI层只负责展示和收集输入，不包含核心逻辑。

### 8. `config.py`（配置管理）
- **角色**：全局配置常量。
- **职责**：
  - 定义默认参数（如最大迭代次数、评分阈值）。
  - 存放数据库路径、Ollama默认地址等。
  - 可能包含工具的安全限制配置。
- **为什么独立**：集中管理配置，便于修改。

---

## 组件交互图（角色协作关系）

```
[用户] ↔ [Streamlit Pages] ↔ [Database]
                             ↕
[OptimizationLoop] ↔ [ModelClient] ↔ [Ollama/外部API]
                  ↕
            [AgentFactory] ↔ [CustomLLM] ↔ [ModelClient]
                  ↕
            [Tool 集合] (Calculator, Search, etc.)
```

**说明**：
- `OptimizationLoop`是核心协调者，它通过`ModelClient`获取模型响应，通过`AgentFactory`创建执行任务的Agent，并通过`Database`读写数据。
- `AgentFactory`内部使用`CustomLLM`适配`ModelClient`，使LangChain Agent能够使用我们的模型。
- `Tool`集合由`AgentFactory`注入到Agent中，供任务执行时调用。
- `Streamlit Pages`通过调用`OptimizationLoop`的方法（如`run_loop`）并传入回调，实现实时更新。

---

## 为什么这种设计符合“角色明确”？

1. **单一职责**：每个类/模块只负责一块独立的功能，易于理解和测试。
2. **接口清晰**：类之间的依赖通过构造函数或方法参数传递，无隐藏耦合。
3. **可扩展性**：增加新工具只需新建Tool子类；增加新模型源只需修改`ModelClient`。
4. **可维护性**：修改数据库逻辑不会影响优化流程；修改UI布局不会影响核心算法。

---
