"""
End-to-End Hero Workflow Integration Test.
Runs full turnaround inspection report analysis, SOP retrieval, calculation,
synthesis, verified .docx deliverable generation, and air-gap telemetry.
"""
from pathlib import Path
from src.core.state_store import StateStore
from src.core.orchestrator import AgenticOrchestrator
from config.settings import KNOWLEDGE_BASE_DIR, SAMPLE_INPUTS_DIR


def test_full_hero_inspection_workflow():
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

    # 1. Check Overall Workflow Status
    assert output["status"] == "workflow_completed"
    assert output["verification_status"] is True
    assert Path(output["generated_deliverable"]).exists()

    # 2. Check Tasks Executed in State Store
    tasks = store.list_tasks_for_project(project_id)
    task_types = [t.task_type for t in tasks]
    assert "multimodal_extraction" in task_types
    assert "retrieval" in task_types
    assert "calculation" in task_types
    assert "synthesis" in task_types
    assert "document_generation" in task_types
    assert "verification" in task_types

    # 3. Check Evidence Grounding in State Store
    evidence = store.get_all_evidence_for_project(project_id)
    assert len(evidence) >= 2
    e_ids = [e.evidence_id for e in evidence]
    assert "E001" in e_ids
    assert "E002" in e_ids

    # 4. Check Model Swaps Recorded in Telemetry
    logs = store.get_recent_model_activity(limit=20)
    assert len(logs) > 0
