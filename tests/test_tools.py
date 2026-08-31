"""
Tests for Deterministic Calculator & Hardened Code Sandbox.
"""
from src.tools.calculator import DeterministicCalculator
from src.tools.sandbox import CodeSandbox


def test_deterministic_calculator_wall_thinning():
    # 3.42 mm measured vs 8.00 mm nominal vs 4.80 mm retirement limit
    res = DeterministicCalculator.calculate_wall_thinning_deviation(
        measured_thickness_mm=3.42,
        nominal_thickness_mm=8.00,
        retirement_thickness_mm=4.80,
    )
    assert res["total_loss_mm"] == 4.58
    assert res["loss_percentage_nominal"] == 57.25
    assert res["is_threshold_breached"] is True
    assert res["breach_margin_mm"] == 1.38
    assert res["deviation_percentage_below_retirement"] == 28.75
    assert res["severity_level"] == "Critical"
    assert len(res["audit_trail"]) >= 5


def test_deterministic_calculator_safe_expression():
    res = DeterministicCalculator.compute_expression("(8.0 - 3.42) / 8.0 * 100")
    assert round(res["result"], 2) == 57.25


def test_sandbox_watchdog_kills_runaway_loop():
    sandbox = CodeSandbox(timeout_seconds=2)
    
    # Runaway infinite loop script
    runaway_code = """
import time
while True:
    time.sleep(0.1)
"""
    res = sandbox.execute_python_code(runaway_code, timeout_seconds=2)
    assert res.is_timed_out is True
    assert res.exit_code == -1
    assert "timed out" in res.stderr.lower() or "terminated" in res.stderr.lower()


def test_sandbox_executes_valid_code():
    sandbox = CodeSandbox(timeout_seconds=5)
    valid_code = "print('SOVEREIGN_SANDBOX_SUCCESS')"
    res = sandbox.execute_python_code(valid_code)
    assert res.is_timed_out is False
    assert res.exit_code == 0
    assert "SOVEREIGN_SANDBOX_SUCCESS" in res.stdout
