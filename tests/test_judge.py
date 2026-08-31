"""
Tests for Tri-Axial LLM-as-Judge Evaluator.
"""
from src.core.state_store import StateStore
from src.models.lifecycle import ModelLifecycleManager
from src.models.model_client import ModelClient
from src.validation.judge import LLMJudge


def test_llm_judge_faithfulness_and_relevance():
    """Verifies that the LLM Judge validates grounded outputs and rejects unverified numbers."""
    store = StateStore("sqlite:///:memory:")
    lifecycle = ModelLifecycleManager(store)
    client = ModelClient(store, lifecycle)
    judge = LLMJudge(store, client)

    query = "What is the mandatory retirement thickness under SOP-17?"
    valid_answer = "Under SOP-17 Section 4.2, the mandatory retirement thickness is 4.80 mm."
    context = ["SOP-17 Section 4.2 specifies mandatory retirement thickness of 4.80 mm."]

    # 1. Grounded answer should pass
    score = judge.evaluate_output(query, valid_answer, context, project_id="P_TEST")
    assert score.is_passing is True
    assert score.faithfulness == 5
    assert score.relevance >= 4
    assert score.overall_score >= 4.0

    # 2. Hallucinated number (e.g. 19.75 mm not in context) should fail faithfulness
    hallucinated_answer = "The retirement limit is 19.75 mm according to custom calculations."
    fail_score = judge.evaluate_output(query, hallucinated_answer, context, project_id="P_TEST")
    assert fail_score.faithfulness <= 2
