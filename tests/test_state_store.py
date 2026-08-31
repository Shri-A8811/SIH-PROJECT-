"""
Tests for Persistent State Store & Restart Safety.
"""
import pytest
from src.core.state_store import StateStore, TaskStatus


def test_state_store_project_and_task_crud():
    store = StateStore("sqlite:///:memory:")
    
    # 1. Create Project
    project = store.create_project("PROJ_001", "MRPL Turnaround", "Inspect CDU-1")
    assert project.id == "PROJ_001"
    
    # 2. Add Task
    task = store.add_task(
        task_id="T001",
        project_id="PROJ_001",
        task_type="document_analysis",
        objective="Extract UTG wall thinning data",
        assigned_model="frob/unlimited-ocr:3b",
        inputs={"doc_id": "DOC01"},
    )
    assert task.status == TaskStatus.PENDING
    assert task.retry_count == 0

    # 3. Mark Running (Strict Status Discipline)
    store.mark_task_running("T001")
    t_running = store.get_task("T001")
    assert t_running.status == TaskStatus.RUNNING
    assert t_running.started_at is not None

    # 4. Complete Task Atomically with Result
    store.complete_task("T001", {"status": "ok", "findings": []})
    t_done = store.get_task("T001")
    assert t_done.status == TaskStatus.COMPLETED
    assert t_done.completed_at is not None


def test_restart_safety_task_recovery():
    store = StateStore("sqlite:///:memory:")
    store.create_project("PROJ_RECOVER", "Recovery Test", "Test crash recovery")

    # Add tasks in different states
    store.add_task("T_DONE", "PROJ_RECOVER", "retrieval", "Done task", "qwen3.5:9b")
    store.complete_task("T_DONE", {"result": "saved"})

    store.add_task("T_CRASHED", "PROJ_RECOVER", "calc", "Interrupted task", "deterministic_calc")
    store.mark_task_running("T_CRASHED")

    # Simulate crash & restart
    recovered = store.recover_pending_and_running_tasks("PROJ_RECOVER")
    assert len(recovered) == 1
    assert recovered[0].task_id == "T_CRASHED"
    
    # Task should be safely reset to PENDING for clean idempotent retry
    t_after = store.get_task("T_CRASHED")
    assert t_after.status == TaskStatus.PENDING


def test_evidence_and_artifact_records():
    store = StateStore("sqlite:///:memory:")
    store.create_project("PROJ_EV", "Evidence Test", "Testing evidence registration")

    # Add Evidence
    ev = store.add_evidence(
        evidence_id="E001",
        project_id="PROJ_EV",
        source_type="multimodal_ocr",
        source_document="Turnaround_Report_2026.pdf",
        page_number=4,
        section="UTG Wall Thickness",
        extracted_text="Measured thickness: 3.42 mm",
        structured_data={"measured_mm": 3.42},
        confidence=0.98,
    )
    assert ev.evidence_id == "E001"
    
    fetched = store.get_evidence("E001")
    assert fetched is not None
    assert fetched.page_number == 4

    # Record Artifact
    art = store.record_artifact(
        artifact_id="ART_001",
        project_id="PROJ_EV",
        artifact_type="docx",
        file_path="/path/to/Approval_Note.docx",
        file_size_bytes=15420,
    )
    assert art.artifact_id == "ART_001"
