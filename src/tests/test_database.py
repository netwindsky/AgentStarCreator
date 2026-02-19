import pytest
import os
import tempfile
from src.db.database import Database, init_db


class TestDatabase:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_db(self.db_path)
        self.db = Database(self.db_path)
        yield
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_create_agent(self):
        agent_id = self.db.create_agent(
            role="测试角色",
            user_requirement="测试需求",
            output_format="JSON"
        )
        assert agent_id > 0
        
        agent = self.db.get_agent(agent_id)
        assert agent is not None
        assert agent['role'] == "测试角色"
        assert agent['status'] == 'created'
    
    def test_add_model_config(self):
        agent_id = self.db.create_agent(
            role="测试角色",
            user_requirement="测试需求",
            output_format="JSON"
        )
        
        self.db.add_model_config(
            agent_id=agent_id,
            model_type='base',
            model_source='ollama',
            model_name='test-model'
        )
        
        configs = self.db.get_model_configs(agent_id)
        assert 'base' in configs
        assert configs['base']['model_name'] == 'test-model'
    
    def test_add_task(self):
        agent_id = self.db.create_agent(
            role="测试角色",
            user_requirement="测试需求",
            output_format="JSON"
        )
        
        task_id = self.db.add_task(
            agent_id=agent_id,
            task_description="测试任务"
        )
        assert task_id > 0
        
        tasks = self.db.get_active_tasks(agent_id)
        assert "测试任务" in tasks
    
    def test_list_agents(self):
        self.db.create_agent(
            role="角色1",
            user_requirement="需求1",
            output_format="JSON"
        )
        self.db.create_agent(
            role="角色2",
            user_requirement="需求2",
            output_format="Markdown"
        )
        
        agents = self.db.list_agents()
        assert len(agents) >= 2
    
    def test_update_agent_status(self):
        agent_id = self.db.create_agent(
            role="测试角色",
            user_requirement="测试需求",
            output_format="JSON"
        )
        
        self.db.update_agent_status(agent_id, 'running')
        agent = self.db.get_agent(agent_id)
        assert agent['status'] == 'running'
