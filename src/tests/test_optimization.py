from src.core.model_client import ModelClient
from src.core.optimization import OptimizationLoop, StopReason
from src.core.evaluator import Evaluator, EvaluationResult


class TestModelClient:
    def test_init_ollama(self):
        client = ModelClient(source="ollama", model_name="test-model")
        assert client.source == "ollama"
        assert client.model_name == "test-model"
    
    def test_init_api(self):
        client = ModelClient(
            source="api",
            model_name="gpt-4",
            api_base="https://api.example.com/v1",
            api_key="test-key"
        )
        assert client.source == "api"
        assert client.model_name == "gpt-4"


class TestEvaluator:
    def test_evaluation_result_calculation(self):
        result = EvaluationResult(
            content_quality=5,
            format_compliance=4,
            tool_usage=3,
            creativity=4,
            final_score=5 * 0.4 + 4 * 0.25 + 3 * 0.2 + 4 * 0.15,
            feedback="测试反馈"
        )
        assert result.final_score == 4.1
        assert result.feedback == "测试反馈"


class TestStopReason:
    def test_stop_reasons_exist(self):
        assert StopReason.SCORE_THRESHOLD.value == "score_threshold"
        assert StopReason.MAX_ITERATIONS.value == "max_iterations"
        assert StopReason.EARLY_STOP.value == "early_stop"
        assert StopReason.PERFECT_SCORE.value == "perfect_score"
        assert StopReason.USER_STOP.value == "user_stop"
