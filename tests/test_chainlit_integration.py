"""
Integration test for Chainlit UI pipeline methods and signatures.
Ensures all method calls in chainlit_app.py (multimodal extractor, RAG, calculator,
docgen, verifier, network monitor) execute with valid arguments and return types.
"""
from pathlib import Path
from config.settings import SAMPLE_INPUTS_DIR
from src.core.state_store import StateStore
from src.core.orchestrator import AgenticOrchestrator
from src.security.network_monitor import AirGapNetworkMonitor


def test_chainlit_pipeline_method_signatures():
    """Verifies that all methods invoked by chainlit_app.py exist with matching signatures."""
    store = StateStore("sqlite:///:memory:")
    orchestrator = AgenticOrchestrator(store)
    network_monitor = AirGapNetworkMonitor()
    project_id = "TEST_CL_001"

    # 1. Project creation
    project = store.create_project(project_id, "Test Project", "Test Objective")
    assert project.id == project_id

    # 2. Network monitor
    snap = network_monitor.inspect_current_egress()
    assert snap.external_connections == 0
    assert hasattr(snap, "loopback_connections")

    # 3. Multimodal Document Extractor
    sample_doc = SAMPLE_INPUTS_DIR / "MRPL_Turnaround_Inspection_Report_2026.md"
    extraction_res = orchestrator.multimodal_extractor.extract_inspection_report(
        document_path=str(sample_doc),
        project_id=project_id,
        task_id="T001",
    )
    assert extraction_res["status"] == "error"
    assert extraction_res["findings"] == []

    # 4. Hybrid Retriever
    rag_res = orchestrator.retriever.search(
        query="CDU crude transfer piping minimum allowable retirement thickness SOP 17",
        project_id=project_id,
        top_k=3,
    )
    assert "results" in rag_res
    results = rag_res["results"]
    assert len(results) > 0

    # 5. Deterministic Calculator
    calc_res = orchestrator.calculator.calculate_wall_thinning_deviation(
        measured_thickness_mm=3.42,
        nominal_thickness_mm=8.00,
        retirement_thickness_mm=4.80,
    )
    assert calc_res["deviation_percentage_below_retirement"] == 28.75
    assert calc_res["is_threshold_breached"] is True

    # 6. Docx Generator
    docgen_res = orchestrator.docx_generator.generate_approval_note(
        project_id=project_id,
        title="CDU-1 & VGO Turnaround Inspection Approval Note",
        executive_summary="Testing executive summary.",
        findings=[],
        calculation_data=calc_res,
        sop_citations=results,
    )
    assert isinstance(docgen_res, dict)
    assert "file_path" in docgen_res
    assert "artifact_id" in docgen_res
    docx_path = docgen_res["file_path"]
    artifact_id = docgen_res["artifact_id"]
    assert Path(docx_path).exists()

    # 7. Artifact Verifier
    v_res = orchestrator.verifier.verify_docx_deliverable(
        artifact_id=artifact_id,
        file_path=docx_path,
        expected_numeric_values=["3.42", "4.80", "28.75"],
    )
    assert hasattr(v_res, "is_passed")
    assert v_res.is_passed is True
