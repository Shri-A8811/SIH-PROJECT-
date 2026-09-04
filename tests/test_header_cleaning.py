import pytest
from src.ui.app import clean_boilerplate_header
from src.core.orchestrator import clean_boilerplate_header as orch_clean_boilerplate_header


def test_clean_boilerplate_header_exact_user_screenshot():
    raw = """Sovereign On-Premise Industrial AI Assistant Report
Facility: Mangalore Refinery and Petrochemicals Limited (MRPL) Subject: Project Documentation Generation – Customer Review Analytics Dashboard Status: Verified & Ready for PDF Conversion Source Material: Shridhar_Cisco_SIP_Final_Report.pdf

## 1. Executive Summary
This is the verified engineering content."""
    
    cleaned = clean_boilerplate_header(raw)
    assert "Sovereign On-Premise Industrial AI Assistant Report" not in cleaned
    assert "Facility: Mangalore Refinery" not in cleaned
    assert "Subject:" not in cleaned
    assert "Status: Verified" not in cleaned
    assert "Source Material:" not in cleaned
    assert cleaned.startswith("## 1. Executive Summary")


def test_clean_boilerplate_header_variations():
    raw_markdown = """# Sovereign On-Premise Industrial AI Assistant Report
**Facility:** Mangalore Refinery and Petrochemicals Limited (MRPL)
**Subject:** Turnaround Evaluation
**Status:** Completed

---

## Findings Matrix
All findings verified."""

    cleaned = clean_boilerplate_header(raw_markdown)
    assert not cleaned.startswith("# Sovereign On-Premise")
    assert "Facility:" not in cleaned
    assert cleaned.startswith("## Findings Matrix")


def test_clean_boilerplate_header_preserves_regular_response():
    normal = """## Calculation Results
The wall thinning percentage is 28.75%."""
    assert clean_boilerplate_header(normal) == normal
