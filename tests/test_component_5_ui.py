"""
Verification Test Suite for Component 5: Sovereign Web Interface & User Experience.
Validates:
1. UI Workbench component initialization (StateStore, AirGapNetworkMonitor, AgenticOrchestrator).
2. StateStore chunk retrieval by filename for the interactive Chunk & Hierarchy Inspector.
3. ASME B31.3 on-demand calculation payload & formatting in the interactive drawer.
4. API 570 Corrosion Rate & RUL calculation payload & formatting in the interactive drawer.
5. Real-time hardware VRAM telemetry and resident model payload formatting for UI gauges.
6. Programmatic air-gap integrity verification and diagnostic message formatting.
"""
import pytest
from src.core.state_store import StateStore, KnowledgeChunkRecord, DocumentRecord
from src.core.orchestrator import AgenticOrchestrator
from src.security.network_monitor import AirGapNetworkMonitor


@pytest.fixture
def test_env(tmp_path):
    db_file = tmp_path / "test_comp5_store.db"
    store = StateStore(database_url=f"sqlite:///{db_file}")
    net_monitor = AirGapNetworkMonitor()
    orchestrator = AgenticOrchestrator(store)
    return store, net_monitor, orchestrator


def test_ui_workbench_components_initialization(test_env):
    store, net_monitor, orchestrator = test_env
    assert store is not None
    assert net_monitor is not None
    assert orchestrator is not None
    assert hasattr(orchestrator, "run_autonomous_plan_loop_stream")
    assert hasattr(orchestrator, "calculator")


def test_state_store_chunk_retrieval_for_ui_inspector(test_env):
    store, _, _ = test_env
    
    # 1. Register a document in inventory
    store.upsert_document(
        filename="MRPL_Turnaround_2026.md",
        category="Turnaround",
        file_path="/data/knowledge_base/Turnaround/MRPL_Turnaround_2026.md",
        file_size_bytes=10240,
        chunk_count=2,
    )
    
    # 2. Add knowledge chunk records
    store.upsert_knowledge_chunk(
        chunk_id="CHK_TEST_001",
        document_name="MRPL_Turnaround_2026.md",
        category="Turnaround",
        section_title="CDU-1 Ultrasonic Gauging",
        page_number=1,
        content="Ultrasonic wall thickness reading for line P-104B is 3.42 mm.",
    )
    store.upsert_knowledge_chunk(
        chunk_id="CHK_TEST_002",
        document_name="MRPL_Turnaround_2026.md",
        category="Turnaround",
        section_title="Retirement Limit Table",
        page_number=2,
        content="| Component | Measured | Retirement Limit |\n| P-104B | 3.42 mm | 4.80 mm |",
    )

    # 3. Retrieve chunks by filename for the UI Chunk Inspector
    chunks = store.get_document_chunks_by_filename("MRPL_Turnaround_2026.md")
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "CHK_TEST_001"
    assert chunks[0].section_title == "CDU-1 Ultrasonic Gauging"
    assert chunks[1].chunk_id == "CHK_TEST_002"
    assert "| P-104B |" in chunks[1].content


def test_ui_interactive_asme_b31_3_calculation_payload(test_env):
    _, _, orchestrator = test_env
    
    # Test calculation invocation as performed by the UI calculate button
    asme_res = orchestrator.calculator.calculate_asme_b31_3_min_thickness(
        design_pressure_bar=35.0,
        outside_diameter_mm=219.1,
        allowable_stress_mpa=115.0,
        weld_efficiency_e=1.0,
        corrosion_allowance_mm=1.5,
        measured_thickness_mm=3.42,
    )
    
    assert asme_res["operation"] == "asme_b31_3_min_thickness"
    assert "min_required_thickness_mm" in asme_res
    assert "pressure_design_thickness_mm" in asme_res
    assert asme_res["is_compliant"] is False
    assert asme_res["status"] == "NON_COMPLIANT_RETIREMENT_REQUIRED"
    assert len(asme_res["audit_trail"]) >= 6


def test_ui_interactive_api_570_calculation_payload(test_env):
    _, _, orchestrator = test_env
    
    # Test calculation invocation as performed by the UI calculate button
    api_res = orchestrator.calculator.calculate_corrosion_rate_and_rul(
        previous_thickness_mm=8.0,
        current_thickness_mm=6.2,
        time_interval_years=5.0,
        required_thickness_mm=4.8,
    )
    
    assert api_res["operation"] == "corrosion_rate_and_rul"
    assert abs(api_res["corrosion_rate_mm_per_year"] - 0.36) < 0.001
    assert abs(api_res["remaining_useful_life_years"] - 3.89) < 0.02
    assert abs(api_res["api_570_next_inspection_interval_years"] - 1.95) < 0.02
    assert "audit_trail" in api_res


def test_ui_hardware_telemetry_payload_format(test_env):
    _, _, orchestrator = test_env
    
    telemetry = orchestrator.lifecycle_manager.get_runtime_model_telemetry()
    assert "total_vram_mb" in telemetry
    assert "active_models" in telemetry
    assert "is_live_daemon" in telemetry
    assert isinstance(telemetry["total_vram_mb"], (int, float))


def test_ui_airgap_integrity_gate_payload(test_env):
    _, net_monitor, _ = test_env
    
    is_airgap_ok, airgap_diag = net_monitor.verify_air_gap_integrity()
    assert isinstance(is_airgap_ok, bool)
    assert is_airgap_ok is True
    assert isinstance(airgap_diag, str)
    assert "0 external connections" in airgap_diag
