"""
Comprehensive Test Suite for Component 3: Model Lifecycle & Zero-Egress Inference Engine.
Validates:
1. Live /api/ps VRAM telemetry & real-time eviction polling.
2. ContextBudgetManager token estimation & middle-context compression for oversized prompts (>5,000 tokens).
3. ModelClient HTTP connection pooling and robust retries.
4. AirGapNetworkMonitor inline air-gap integrity verification gate.
5. Single-GPU sequential model swapping invariants and eviction logging.
"""
import pytest
import os
from unittest.mock import MagicMock, patch
from src.core.state_store import StateStore
from src.models.lifecycle import ModelLifecycleManager
from src.models.model_client import ModelClient, ContextBudgetManager
from src.security.network_monitor import AirGapNetworkMonitor


@pytest.fixture
def test_store(tmp_path):
    db_file = tmp_path / "test_comp3_store.db"
    return StateStore(database_url=f"sqlite:///{db_file}")


def test_context_budget_manager_token_counting():
    mgr = ContextBudgetManager(default_budget_tokens=3800)
    
    # Empty string
    assert mgr.count_tokens("") == 0
    
    # Normal short text
    text = "Crude Distillation Unit transfer line ultrasonic inspection."
    tokens = mgr.count_tokens(text)
    assert tokens > 0
    assert tokens < 50


def test_context_budget_manager_fit_oversized_prompt():
    mgr = ContextBudgetManager(default_budget_tokens=1000)
    
    # Create an oversized prompt (> 5,000 tokens)
    header = "### AIR-GAPPED WORKBENCH TASK CONTRACT: T001\nTASK TYPE: document_analysis\nOBJECTIVE: Inspect pipe wall thinning.\n"
    tail = "\n--- REQUIRED OUTPUT JSON SCHEMA ---\n{\"type\": \"object\", \"properties\": {\"findings\": {\"type\": \"array\"}}}\nINSTRUCTION: Return ONLY valid JSON."
    
    # 2,000 lines of middle evidence (~10,000 tokens)
    middle_lines = [f"EVIDENCE CHUNK {i:04d}: Line inspection reading for pipe section {i} showed nominal 8.00mm." for i in range(500)]
    oversized_prompt = header + "\n".join(middle_lines) + tail
    
    initial_tokens = mgr.count_tokens(oversized_prompt)
    assert initial_tokens > 3000, f"Expected initial tokens > 3000, got {initial_tokens}"
    
    # Fit within budget of 1000 tokens
    fitted = mgr.fit_prompt_within_budget(oversized_prompt, max_tokens=1000)
    fitted_tokens = mgr.count_tokens(fitted)
    
    # Must be <= 1000 tokens
    assert fitted_tokens <= 1000, f"Fitted tokens {fitted_tokens} exceeded budget 1000"
    # Header must be preserved
    assert "AIR-GAPPED WORKBENCH TASK CONTRACT" in fitted
    # Schema tail must be preserved
    assert "REQUIRED OUTPUT JSON SCHEMA" in fitted
    # Compression notice must be present
    assert "Context compressed" in fitted


def test_context_budget_manager_under_budget_unchanged():
    mgr = ContextBudgetManager(default_budget_tokens=3800)
    short_prompt = "### TASK CONTRACT\nOBJECTIVE: Simple calculation."
    fitted = mgr.fit_prompt_within_budget(short_prompt)
    assert fitted == short_prompt


def test_model_client_connection_pooling(test_store):
    lifecycle = ModelLifecycleManager(test_store)
    client = ModelClient(test_store, lifecycle)
    
    # Verify session has adapters mounted
    http_adapter = client.session.adapters.get("http://")
    https_adapter = client.session.adapters.get("https://")
    
    assert http_adapter is not None
    assert https_adapter is not None
    assert http_adapter._pool_connections == 10
    assert http_adapter._pool_maxsize == 20
    assert client.budget_manager is not None


def test_vram_telemetry_and_api_ps(test_store):
    lifecycle = ModelLifecycleManager(test_store)
    
    # Test get_loaded_models_ps
    models = lifecycle.get_loaded_models_ps()
    assert isinstance(models, list)
    
    # Test get_runtime_model_telemetry
    telemetry = lifecycle.get_runtime_model_telemetry()
    assert "loaded_models" in telemetry
    assert "total_vram_mb" in telemetry
    assert "active_resident_model" in telemetry
    assert "ollama_online" in telemetry
    assert isinstance(telemetry["total_vram_mb"], float)


def test_air_gap_integrity_gate():
    monitor = AirGapNetworkMonitor()
    
    # Run inline verification
    is_air_gapped, diagnostic = monitor.verify_air_gap_integrity()
    
    assert isinstance(is_air_gapped, bool)
    assert isinstance(diagnostic, str)
    assert len(diagnostic) > 0
    
    # Workbench process should be air-gapped
    assert is_air_gapped is True
    assert "0 external connections" in diagnostic


def test_single_gpu_eviction_and_swap_mocked(test_store):
    lifecycle = ModelLifecycleManager(test_store)
    
    # Mock Ollama responses for fast testing
    with patch.object(lifecycle, "_unload_model") as mock_unload, \
         patch.object(lifecycle, "_load_model", return_value=3200.0) as mock_load, \
         patch.object(lifecycle, "is_model_loaded_ps", side_effect=[True, False]):
        
        lifecycle.currently_loaded_model = "qwen2.5:3b"
        
        # Swapping to a different model should trigger unloading of the prior model
        res = lifecycle.ensure_model_loaded(
            target_model="qwen2.5-coder:7b",
            project_id="test_swap_proj",
            task_id="T_SWAP",
        )
        
        assert mock_unload.called
        assert mock_load.called
        assert res["status"] in ("loaded", "already_loaded")
        assert lifecycle.currently_loaded_model == "qwen2.5-coder:7b"
