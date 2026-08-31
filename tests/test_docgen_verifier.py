"""
Tests for Document Generation & Inline Artifact Verification.
"""
from pathlib import Path
from src.core.state_store import StateStore
from src.generation.docx_generator import DocxApprovalNoteGenerator
from src.generation.verifier import ArtifactVerifier


def test_docx_generation_and_inline_verification():
    store = StateStore("sqlite:///:memory:")
    docgen = DocxApprovalNoteGenerator(store)
    verifier = ArtifactVerifier(store)

    project_id = "PROJ_TEST_DOC"
    findings = [
        {
            "evidence_id": "E001",
            "equipment": "CDU-1 Transfer Line P-104B",
            "issue": "Severe wall thinning (3.42 mm vs 8.00 mm nominal)",
            "severity": "Critical",
            "measured_value": "3.42 mm",
            "threshold_value": "4.80 mm",
            "status": "NON-COMPLIANT",
        }
    ]
    calc_data = {
        "audit_trail": ["Measured: 3.42 mm vs Limit: 4.80 mm (28.75% breach)"]
    }
    sop_data = [
        {"evidence_id": "E_RET_01", "document_name": "MRPL_SOP_17.md", "section_title": "Sec 4.2", "page_number": 4, "content": "Retirement threshold is 4.80 mm."}
    ]

    # 1. Generate Docx
    gen_res = docgen.generate_approval_note(
        project_id=project_id,
        title="CDU-1 Test Approval Note",
        executive_summary="Urgent spool replacement required.",
        findings=findings,
        calculation_data=calc_data,
        sop_citations=sop_data,
    )
    assert gen_res["human_review_disclaimer_included"] is True
    file_path = gen_res["file_path"]
    assert Path(file_path).exists()

    # 2. Run Verification
    verif_res = verifier.verify_docx_deliverable(
        artifact_id=gen_res["artifact_id"],
        file_path=file_path,
        expected_numeric_values=["3.42", "4.80"],
    )
    assert verif_res.is_passed is True
    assert len(verif_res.errors) == 0


def test_verifier_catches_corrupt_or_missing_file():
    store = StateStore("sqlite:///:memory:")
    verifier = ArtifactVerifier(store)

    verif_res = verifier.verify_docx_deliverable(
        artifact_id="ART_FAKE",
        file_path="/non/existent/file.docx",
    )
    assert verif_res.is_passed is False
    assert len(verif_res.errors) > 0
