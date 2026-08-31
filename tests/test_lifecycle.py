"""
Tests for Model Lifecycle Manager and Single-GPU Sequential Residency.
"""
from src.core.state_store import StateStore
from src.models.lifecycle import ModelLifecycleManager


def test_model_lifecycle_sequential_swaps():
    store = StateStore("sqlite:///:memory:")
    lifecycle = ModelLifecycleManager(store)

    # 1. Load General Reasoning Model
    res1 = lifecycle.ensure_model_loaded("qwen3.5:9b", project_id="P1", task_id="T1")
    assert res1["status"] == "loaded"
    assert lifecycle.currently_loaded_model == "qwen3.5:9b"

    # 2. Loading same model should reuse resident instance without reload penalty
    res1_cached = lifecycle.ensure_model_loaded("qwen3.5:9b", project_id="P1", task_id="T1")
    assert res1_cached["status"] == "already_loaded"
    assert res1_cached["duration_ms"] == 0.0

    # 3. Requesting Multimodal model must UNLOAD General and LOAD Multimodal (single-GPU constraint)
    res2 = lifecycle.ensure_model_loaded("frob/unlimited-ocr:3b", project_id="P1", task_id="T2")
    assert res2["status"] == "loaded"
    assert lifecycle.currently_loaded_model == "frob/unlimited-ocr:3b"

    # 4. Verify telemetry log in state store
    logs = store.get_recent_model_activity(limit=10)
    actions = [log.action for log in logs]
    assert "LOAD" in actions
    assert "UNLOAD" in actions
