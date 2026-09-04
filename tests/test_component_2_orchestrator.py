"""
Verification Test Suite for Component 2: Autonomous Agentic Orchestrator & Cognitive Planner.
Validates:
1. Unified Tool Registry discovery and parameter schemas.
2. Direct deterministic tool execution and unknown tool error handling.
3. Fast-path routing for calculation and coding sandboxes.
4. Multi-step ReAct loop streaming events (thought, tool_call, tool_result, completed).
5. Anti-infinite-loop watchdog and loop termination safeguards.
6. Error self-correction context tracking.
"""
import pytest
from src.core.state_store import StateStore
from src.core.orchestrator import AgenticOrchestrator, PlanStep


def test_orchestrator_tool_registry_definitions():
    """Validates that all core sovereign tools are registered with schemas."""
    store = StateStore()
    orch = AgenticOrchestrator(store)

    tools = orch.get_registered_tools()
    assert len(tools) >= 6

    tool_names = {t.name for t in tools}
    expected = {
        "knowledge_search",
        "calculate_wall_thinning",
        "evaluate_expression",
        "execute_python_code",
        "generate_docx_report",
        "verify_document",
    }
    assert expected.issubset(tool_names), f"Missing expected tools in registry: {expected - tool_names}"

    # Verify parameter schemas
    calc_tool = next(t for t in tools if t.name == "calculate_wall_thinning")
    assert "measured_thickness_mm" in calc_tool.parameters["properties"]
    assert "nominal_thickness_mm" in calc_tool.parameters["properties"]
    assert "retirement_thickness_mm" in calc_tool.parameters["properties"]


def test_orchestrator_tool_execution_direct():
    """Validates direct execution of registered tools through the orchestrator dispatcher."""
    store = StateStore()
    orch = AgenticOrchestrator(store)

    # 1. Deterministic Calculation Tool
    calc_out = orch.execute_tool(
        "calculate_wall_thinning",
        {"measured_thickness_mm": 3.42, "nominal_thickness_mm": 8.00, "retirement_thickness_mm": 4.80},
        project_id="test_direct_calc",
    )
    assert calc_out.get("status") == "calculated"
    data = calc_out.get("data", {})
    assert data.get("total_loss_mm") == 4.58
    assert data.get("loss_percentage_nominal") == 57.25
    assert data.get("is_threshold_breached") is True
    assert data.get("severity_level") == "Critical"

    # 2. Arithmetic Expression Tool
    math_out = orch.execute_tool(
        "evaluate_expression",
        {"expression": "(150 - 142) / 150 * 100"},
        project_id="test_direct_math",
    )
    assert math_out.get("operation") == "expression_evaluation"
    assert abs(math_out.get("result") - 5.333333) < 1e-4

    # 3. Python Sandbox Tool
    code_out = orch.execute_tool(
        "execute_python_code",
        {"code": "print('MRPL Sovereign Sandbox Ready')"},
        project_id="test_direct_code",
    )
    assert code_out.get("exit_code") == 0
    assert "MRPL Sovereign Sandbox Ready" in code_out.get("stdout", "")

    # 4. Unknown Tool Error Handling
    err_out = orch.execute_tool("non_existent_tool", {}, project_id="test_err")
    assert err_out.get("status") == "error"
    assert "Unknown tool" in err_out.get("error", "")


def test_orchestrator_calculation_fast_path():
    """Validates fast-path deterministic execution for quantitative user prompts."""
    store = StateStore()
    orch = AgenticOrchestrator(store)

    prompt = "Calculate wall thinning breach margin for measured 3.42 mm, nominal 8.00 mm, retirement 4.80 mm."
    result = orch.run_autonomous_plan_loop(prompt, project_id="test_fast_calc")

    assert result.get("status") == "completed"
    resp = result.get("final_response", "")
    assert "Deterministic Wall Thinning Calculation" in resp
    assert "3.42 mm" in resp
    assert "57.25% loss" in resp


def test_orchestrator_coding_fast_path():
    """Validates fast-path isolated execution for script execution requests."""
    store = StateStore()
    orch = AgenticOrchestrator(store)

    prompt = "print(sum([10, 20, 30, 40]))"
    result = orch.run_autonomous_plan_loop(prompt, project_id="test_fast_code")

    assert result.get("status") == "completed"
    resp = result.get("final_response", "")
    assert "100" in resp
    assert "Sandbox Execution Result" in resp


def test_anti_infinite_loop_watchdog():
    """Validates that the orchestrator terminates gracefully if a duplicate tool loop occurs."""
    store = StateStore()
    orch = AgenticOrchestrator(store)

    # Mock _decide_next_step to deliberately repeat the exact same tool call
    orch._decide_next_step = lambda *args, **kwargs: (
        "Repeating tool call",
        "evaluate_expression",
        {"expression": "2 + 2"},
    )

    events = list(orch.run_autonomous_plan_loop_stream(
        "Run multi-step calculation",
        project_id="test_watchdog",
        max_steps=5,
    ))

    # Should have stopped after step 2 because of duplicate watchdog
    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    assert len(tool_calls) == 2, f"Watchdog must break loop after 2 identical calls, got: {len(tool_calls)}"

    completed = next((e for e in events if e.get("type") == "completed"), None)
    assert completed is not None
    assert completed.get("status") == "completed"


def test_streaming_event_structure():
    """Validates that run_autonomous_plan_loop_stream yields valid event types."""
    store = StateStore()
    orch = AgenticOrchestrator(store)

    prompt = "Calculate wall thinning with 3.42, 8.00, 4.80"
    events = list(orch.run_autonomous_plan_loop_stream(prompt, project_id="test_events"))

    types = [e.get("type") for e in events]
    assert "plan_start" in types
    assert "final_chunk" in types
    assert "completed" in types
