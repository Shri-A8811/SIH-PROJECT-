"""
Full System End-to-End Integration Test Suite.
Verifies all 5 elevated components working seamlessly together:
1. Component 1: Document Intelligence & Structure-Aware RAG (Outlines, Tables, BGE, Hybrid RRF, Breadcrumbs)
2. Component 2: Autonomous Agentic Orchestrator & Cognitive Planner (ReAct loop, 8-tool registry, Watchdog)
3. Component 3: Model Lifecycle & Zero-Egress Engine (/api/ps VRAM, ContextBudgetManager, Connection Pool, Air-Gap Gate)
4. Component 4: Deterministic Tools & Sandbox Hardening (ASME B31.3, API 570 RUL, AST Security, Verifier)
5. Component 5: Sovereign Web Interface & State Synchronization (Sessions, Chunk Inspector, Telemetry Payloads)
"""
import pytest
from pathlib import Path

from src.core.state_store import StateStore
from src.core.orchestrator import AgenticOrchestrator
from src.models.lifecycle import ModelLifecycleManager
from src.models.model_client import ContextBudgetManager
from src.security.network_monitor import AirGapNetworkMonitor
from src.tools.calculator import DeterministicCalculator
from src.tools.sandbox import CodeSandbox, SecurityASTValidator
from src.generation.verifier import ArtifactVerifier
from config.settings import settings, SAMPLE_INPUTS_DIR, KNOWLEDGE_BASE_DIR


@pytest.fixture
def integrated_env(tmp_path):
    """Sets up a complete isolated integration test environment."""
    db_path = tmp_path / "integration_workbench.db"
    store = StateStore(f"sqlite:///{db_path}")
    monitor = AirGapNetworkMonitor()
    orchestrator = AgenticOrchestrator(store)
    calculator = DeterministicCalculator()
    sandbox = CodeSandbox()
    budget_mgr = ContextBudgetManager(default_budget_tokens=3800)
    verifier = ArtifactVerifier(store)

    return {
        "store": store,
        "monitor": monitor,
        "orchestrator": orchestrator,
        "calculator": calculator,
        "sandbox": sandbox,
        "budget_mgr": budget_mgr,
        "verifier": verifier,
        "tmp_path": tmp_path,
    }


def test_e2e_rag_ingestion_and_scoped_retrieval(integrated_env):
    """
    Component 1 + 5 Integration:
    Ingests structured document with tables & outlines, verifies chunk storage,
    breadcrumb indexing, category scoping, and UI chunk inspector data contract.
    """
    store = integrated_env["store"]
    orchestrator = integrated_env["orchestrator"]

    # 1. Ingest sample SOP markdown
    sop_content = (
        "# MRPL SOP-17: Crude Distillation Unit Piping\n\n"
        "## Section 4.0 Design Limits\n"
        "All ASTM A106 Grade B piping operates under maximum allowable stress S = 137.9 MPa.\n\n"
        "| Line No | Nominal OD (mm) | Design Press (bar) | Corrosion Allow (mm) |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| 01-CDU-P101 | 219.1 | 35.0 | 3.0 |\n"
        "| 01-CDU-P102 | 323.8 | 28.0 | 3.0 |\n"
    )
    doc_path = integrated_env["tmp_path"] / "MRPL_SOP_17_CDU.md"
    doc_path.write_text(sop_content, encoding="utf-8")

    orchestrator.retriever.ingest_file(doc_path, category="Standards & SOPs")

    # 2. Verify state store has chunks with breadcrumb and category
    chunks = store.get_document_chunks_by_filename("MRPL_SOP_17_CDU.md")
    assert len(chunks) > 0
    assert any("Design Limits" in c.section_title for c in chunks)
    assert any("Line No" in c.content for c in chunks)

    # 3. Hybrid Retrieval with Category Scoping
    search_res = orchestrator.retriever.search(
        query="What is the allowable stress and design pressure for 01-CDU-P101?",
        project_id="TEST_RAG_E2E",
        category="Standards & SOPs",
        top_k=3,
    )
    results = search_res.get("results", [])
    assert len(results) > 0
    top_doc = results[0]["document_name"]
    assert "MRPL_SOP_17_CDU.md" in top_doc
    assert any("137.9" in r["content"] or "219.1" in r["content"] for r in results)


def test_e2e_autonomous_react_loop_with_all_tools(integrated_env):
    """
    Component 1 + 2 + 4 Integration:
    Runs the autonomous ReAct planner loop to execute a multi-step inspection task:
    1. Search knowledge base
    2. Run ASME B31.3 calculation
    3. Run API 570 RUL calculation
    """
    orchestrator = integrated_env["orchestrator"]
    store = integrated_env["store"]
    project_id = "E2E_REACT_PROJ"
    store.create_project(project_id=project_id, name="ReAct Test", objective="End-to-End Autonomous Loop Test")

    # Multi-step ReAct sequence
    steps = [
        (
            "I need to calculate the ASME B31.3 min wall thickness for 01-CDU-P101.",
            "calculate_asme_b31_3",
            {
                "design_pressure_bar": 35.0,
                "outside_diameter_mm": 219.1,
                "allowable_stress_mpa": 115.0,
                "measured_thickness_mm": 5.50,
            },
        ),
        (
            "I will now assess corrosion rate and remaining life under API 570.",
            "calculate_corrosion_rate_and_rul",
            {
                "previous_thickness_mm": 8.18,
                "current_thickness_mm": 6.20,
                "time_interval_years": 5.0,
                "required_thickness_mm": 4.794,
            },
        ),
        (
            "All deterministic calculations verified. Ready to synthesize answer.",
            "final_answer",
            {},
        ),
    ]
    step_iter = iter(steps)
    orchestrator._decide_next_step = lambda *args, **kwargs: next(step_iter)

    events = list(
        orchestrator.run_autonomous_plan_loop_stream(
            user_prompt="Assess pipe 01-CDU-P101 under 35 bar design pressure against ASME B31.3 and API 570.",
            project_id=project_id,
            max_steps=5,
        )
    )

    event_types = [e["type"] for e in events]
    assert "plan_start" in event_types
    assert "thought" in event_types
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "completed" in event_types

    # Verify tool results produced valid outputs
    tool_results = [e.get("output", {}) for e in events if e.get("type") == "tool_result"]
    assert any("min_required_thickness_mm" in str(res) for res in tool_results)
    assert any("remaining_useful_life_years" in str(res) for res in tool_results)


def test_e2e_sandbox_security_and_ast_shield(integrated_env):
    """
    Component 4 Integration:
    Ensures that AST security validator halts malicious exploits before process
    spawning, while cleanly allowing deterministic engineering computations.
    """
    sb = integrated_env["sandbox"]

    # 1. Block forbidden socket import
    exploit_code = "import socket\ns = socket.socket()\nprint('Hacked')"
    is_safe, violations = sb.validate_code_security(exploit_code)
    assert is_safe is False
    assert any("Forbidden module import: 'socket'" in v for v in violations)

    res = sb.execute_python_code(exploit_code)
    assert res.exit_code == -2
    assert "Forbidden module import: 'socket'" in res.stderr

    # 2. Block forbidden subprocess execution
    proc_code = "import subprocess\nsubprocess.run(['dir'])"
    is_safe2, violations2 = sb.validate_code_security(proc_code)
    assert is_safe2 is False
    assert any("Forbidden module import: 'subprocess'" in v for v in violations2)

    res2 = sb.execute_python_code(proc_code)
    assert res2.exit_code == -2
    assert "Forbidden module import: 'subprocess'" in res2.stderr

    # 3. Allow valid engineering calculation
    safe_code = (
        "t_act = 6.20\n"
        "t_min = 5.76\n"
        "cr = (8.18 - t_act) / 5.0\n"
        "rul = (t_act - t_min) / cr\n"
        "print(f'RUL={rul:.2f}')\n"
    )
    is_safe3, violations3 = sb.validate_code_security(safe_code)
    assert is_safe3 is True
    assert len(violations3) == 0

    res3 = sb.execute_python_code(safe_code)
    assert res3.exit_code == 0
    assert "RUL=1.11" in res3.stdout


def test_e2e_model_lifecycle_vram_and_airgap(integrated_env):
    """
    Component 3 Integration:
    Tests ContextBudgetManager token preservation, /api/ps telemetry polling,
    and strict air-gap socket assertion.
    """
    budget_mgr = integrated_env["budget_mgr"]
    monitor = integrated_env["monitor"]
    lifecycle = ModelLifecycleManager(integrated_env["store"])

    # 1. Context Budget Compression preserving schema and prompt headers
    long_middle = "Turnaround log inspection details for tray 1... " * 300
    prompt = (
        "### SYSTEM CONTRACT: REFINERY INSPECTION AGENT\n"
        "You must respond in strict JSON format conforming to schema:\n"
        f"DATA:\n{long_middle}\n"
        "### SCHEMA REQUIREMENTS:\n"
        '{"status": "PASS", "t_min_mm": float, "rul_years": float}'
    )
    compressed = budget_mgr.fit_prompt_within_budget(prompt, max_tokens=150)
    assert "### SYSTEM CONTRACT" in compressed
    assert "### SCHEMA REQUIREMENTS" in compressed
    assert "Context compressed" in compressed

    # 2. Air-Gap Zero-Egress Assertion
    is_airgap, detail = monitor.verify_air_gap_integrity()
    assert is_airgap is True
    assert "Air-gap integrity verified" in detail

    # 3. Runtime Model Telemetry
    telemetry = lifecycle.get_runtime_model_telemetry()
    assert "active_models" in telemetry
    assert "total_vram_mb" in telemetry
    assert isinstance(telemetry["active_models"], list)


def test_e2e_hero_pipeline_full_workflow(integrated_env):
    """
    Full End-to-End Turnaround Hero Inspection Pipeline:
    Ingests turnaround report, executes SOP retrieval, performs calculations,
    generates approval note (.docx), and verifies deliverable tables & tolerances.
    """
    store = integrated_env["store"]
    orchestrator = integrated_env["orchestrator"]

    # Ingest internal SOPs
    orchestrator.retriever.ingest_directory(str(KNOWLEDGE_BASE_DIR))

    sample_doc = SAMPLE_INPUTS_DIR / "MRPL_Turnaround_Inspection_Report_2026.md"
    project_id = "FULL_E2E_HERO_PROJ"

    output = orchestrator.run_hero_inspection_workflow(
        project_id=project_id,
        document_path=str(sample_doc),
        user_prompt="Analyze this turnaround inspection report, retrieve internal SOPs, verify calculations, and generate approval note.",
    )

    # 1. Workflow completed
    assert output["status"] == "workflow_completed"
    assert output["verification_status"] is True
    assert Path(output["generated_deliverable"]).exists()

    # 2. Verify all tasks registered in SQLite StateStore
    tasks = store.list_tasks_for_project(project_id)
    task_types = {t.task_type for t in tasks}
    assert {"multimodal_extraction", "retrieval", "calculation", "synthesis", "document_generation", "verification"}.issubset(task_types)

    # 3. Verify grounded evidence entries
    evidence = store.get_all_evidence_for_project(project_id)
    assert len(evidence) >= 2


def test_e2e_ui_state_and_inspector_synchronicity(integrated_env):
    """
    Component 5 + StateStore Integration:
    Verifies that the interactive UI Chunk & Hierarchy Inspector,
    Engineering Tool payloads, and Chat Sessions seamlessly synchronize.
    """
    store = integrated_env["store"]
    calculator = integrated_env["calculator"]

    # 1. Multi-session lifecycle
    session = store.create_chat_session(title="Unit 01 CDU Turnaround Review", knowledge_scope="SOPs")
    store.save_chat_message(session.id, "user", "Check pipe thinning on 01-CDU-P101.")
    store.save_chat_message(session.id, "assistant", "Evaluated ASME B31.3 min wall thickness is 5.76 mm.")

    messages = store.get_chat_messages(session.id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"

    # 2. Interactive Calculator UI payload execution
    asme_result = calculator.calculate_asme_b31_3_min_thickness(
        design_pressure_bar=35.0,
        outside_diameter_mm=219.1,
        allowable_stress_mpa=115.0,
        weld_efficiency_e=1.0,
        temp_factor_y=0.4,
        corrosion_allowance_mm=1.5,
        measured_thickness_mm=5.50,
    )
    assert asme_result["standard"] == "ASME B31.3"
    assert asme_result["is_compliant"] is True
    assert abs(asme_result["min_required_thickness_mm"] - 4.794) < 0.01

    api_result = calculator.calculate_corrosion_rate_and_rul(
        previous_thickness_mm=8.18,
        current_thickness_mm=6.20,
        time_interval_years=5.0,
        required_thickness_mm=4.794,
    )
    assert abs(api_result["corrosion_rate_mm_per_year"] - 0.396) < 0.01
    assert abs(api_result["remaining_useful_life_years"] - 3.55) < 0.05
    assert abs(api_result["api_570_next_inspection_interval_years"] - 1.775) < 0.05
