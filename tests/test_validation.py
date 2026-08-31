"""
Tests for Acceptance Gate, Output Schema Validation, and Evidence Grounding.
"""
import pytest
from src.core.state_store import StateStore
from src.core.validation import AcceptanceGate


def test_acceptance_gate_rejects_unregistered_evidence_id():
    store = StateStore("sqlite:///:memory:")
    store.create_project("PROJ_VAL", "Validation Test", "Testing grounding")
    gate = AcceptanceGate(store)

    contract = {
        "task_id": "T001",
        "task_type": "document_analysis",
        "output_schema": {"required": ["findings"]},
    }

    # Model fabricated a citation E999 which does NOT exist in the evidence table
    fabricated_output = {
        "findings": [
            {
                "equipment": "CDU-1 Column C-101",
                "issue": "Corrosion detected",
                "severity": "Critical",
                "evidence_id": "E999",  # Non-existent
            }
        ]
    }

    res = gate.validate_task_result(contract, fabricated_output, project_id="PROJ_VAL")
    assert not res.is_valid
    assert any("E999" in err for err in res.errors)
    assert any("Grounding Check Failed" in err for err in res.errors)


def test_acceptance_gate_accepts_grounded_evidence():
    store = StateStore("sqlite:///:memory:")
    store.create_project("PROJ_VAL2", "Validation Test 2", "Testing valid grounding")
    gate = AcceptanceGate(store)

    # Register real evidence
    store.add_evidence(
        evidence_id="E001",
        project_id="PROJ_VAL2",
        source_type="multimodal_ocr",
        extracted_text="UTG residual thickness: 3.42 mm",
    )

    contract = {
        "task_id": "T002",
        "task_type": "document_analysis",
        "output_schema": {"required": ["findings"]},
    }

    valid_output = {
        "findings": [
            {
                "equipment": "CDU-1 Column C-101",
                "issue": "Corrosion detected",
                "severity": "Critical",
                "evidence_id": "E001",  # Matches real evidence row
            }
        ]
    }

    res = gate.validate_task_result(contract, valid_output, project_id="PROJ_VAL2")
    assert res.is_valid
    assert len(res.errors) == 0


def test_acceptance_gate_semantic_sanity_empty_findings():
    store = StateStore("sqlite:///:memory:")
    store.create_project("PROJ_VAL3", "Semantic Sanity", "Testing empty findings")
    gate = AcceptanceGate(store)

    contract = {
        "task_id": "T003",
        "task_type": "document_analysis",
        "output_schema": {"required": ["findings"]},
    }

    empty_findings_output = {"findings": []}
    res = gate.validate_task_result(contract, empty_findings_output, project_id="PROJ_VAL3")
    assert not res.is_valid
    assert any("Semantic Sanity Failed" in err for err in res.errors)
