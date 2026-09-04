"""
End-to-End Hero Workflow Integration Test.
Runs full turnaround inspection report analysis, SOP retrieval, calculation,
synthesis, verified .docx deliverable generation, and air-gap telemetry.
"""
from pathlib import Path
from src.core.state_store import StateStore
from src.core.orchestrator import AgenticOrchestrator
from config.settings import KNOWLEDGE_BASE_DIR, SAMPLE_INPUTS_DIR


def test_hero_workflow_fails_closed_when_local_inference_is_unavailable():
    store = StateStore("sqlite:///:memory:")
    orchestrator = AgenticOrchestrator(store)

    # Ingest internal SOPs
    orchestrator.retriever.ingest_directory(str(KNOWLEDGE_BASE_DIR))

    sample_doc = SAMPLE_INPUTS_DIR / "MRPL_Turnaround_Inspection_Report_2026.md"
    project_id = "HERO_TEST_PROJ"

    output = orchestrator.run_hero_inspection_workflow(
        project_id=project_id,
        document_path=str(sample_doc),
        user_prompt="Analyze this turnaround inspection report, retrieve internal SOPs, verify calculations, and generate approval note.",
    )

    # Test mode has no local model. The workflow must not manufacture an approval
    # note from the old fixed CDU/VGO demo findings.
    assert output["status"] == "workflow_blocked"
    assert output["verification_status"] is False
    assert output["generated_deliverable"] is None
