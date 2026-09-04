"""
Dedicated Verification Suite for Router, Code Generation Fast-Path, Native PDF Generator,
and Citation Deduplication Fixes.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.core.state_store import StateStore
from src.router.task_router import TaskRouter
from src.core.orchestrator import AgenticOrchestrator
from src.generation.pdf_generator import PdfDeliverableGenerator
from src.knowledge.hybrid_retriever import HybridKnowledgeRetriever


@pytest.fixture
def test_env(tmp_path):
    db_file = tmp_path / "test_fixes.db"
    store = StateStore(f"sqlite:///{db_file}")
    router = TaskRouter()
    orchestrator = AgenticOrchestrator(store)
    pdf_gen = PdfDeliverableGenerator(store)
    return {
        "store": store,
        "router": router,
        "orchestrator": orchestrator,
        "pdf_gen": pdf_gen,
        "tmp_path": tmp_path,
    }


def test_task_router_distinguishes_code_generation_vs_execution(test_env):
    router = test_env["router"]

    # 1. Natural language code generation requests
    d1 = router.route_request("give me code for b tree operations")
    assert d1.selected_path == "coding"
    assert d1.starting_task_type == "code_generation"

    d2 = router.route_request("write me a code for b tree oprations")
    assert d2.selected_path == "coding"
    assert d2.starting_task_type == "code_generation"

    d3 = router.route_request("implement a binary tree search algorithm")
    assert d3.selected_path == "coding"
    assert d3.starting_task_type == "code_generation"

    # 2. Raw Python code execution request
    d4 = router.route_request("print(sum([10, 20, 30]))")
    assert d4.selected_path == "coding"
    assert d4.starting_task_type == "code_execution"


def test_task_router_detects_pdf_generation_intent(test_env):
    router = test_env["router"]

    d1 = router.route_request("make me pdf file from project section")
    assert d1.starting_task_type == "document_generation"
    assert "generate_pdf_report" in d1.recommended_tools

    d2 = router.route_request("generate pdf report for turnaround findings")
    assert d2.starting_task_type == "document_generation"
    assert "generate_pdf_report" in d2.recommended_tools


def test_orchestrator_code_generation_fast_path_does_not_crash_sandbox(test_env):
    orchestrator = test_env["orchestrator"]
    project_id = "TEST_CODE_GEN_FASTPATH"

    # Mock model client to return Python code
    mock_code = "```python\nclass BTreeNode:\n    def __init__(self, leaf=False):\n        self.leaf = leaf\n        self.keys = []\n        self.child = []\n```"
    orchestrator.model_client.generate_text_stream = MagicMock(return_value=iter([mock_code]))

    events = list(orchestrator.run_autonomous_plan_loop_stream(
        user_prompt="give me code for b tree operations",
        project_id=project_id,
        max_steps=2,
    ))

    # Should not crash with SyntaxError
    final_event = next((e for e in events if e.get("type") == "completed"), None)
    assert final_event is not None
    assert "SyntaxError" not in final_event.get("final_response", "")
    assert "BTreeNode" in final_event.get("final_response", "")


def test_orchestrator_raw_python_execution_fast_path(test_env):
    orchestrator = test_env["orchestrator"]
    project_id = "TEST_RAW_PYTHON"

    events = list(orchestrator.run_autonomous_plan_loop_stream(
        user_prompt="print(sum([10, 20, 30]))",
        project_id=project_id,
        max_steps=2,
    ))

    final_event = next((e for e in events if e.get("type") == "completed"), None)
    assert final_event is not None
    assert "60" in final_event.get("final_response", "")
    assert "Sandbox Execution Result" in final_event.get("final_response", "")


def test_native_pdf_generator_creates_valid_file(test_env):
    pdf_gen = test_env["pdf_gen"]
    store = test_env["store"]
    project_id = "TEST_PDF_DOC"
    store.create_project(project_id, "Test PDF Project", "Testing PDF generation")

    res = pdf_gen.generate_pdf_report(
        project_id=project_id,
        title="Customer Review Analytics Project",
        executive_summary="This is the project overview extracted from Chapter 7.",
        findings=[
            {"asset": "Ingestion Pipeline", "reading": "Pandas CSV/Excel", "standard": "API Spec", "status": "VERIFIED"},
            {"asset": "Storage Layer", "reading": "PostgreSQL DB", "standard": "SQL Compliance", "status": "VERIFIED"},
        ],
        sop_citations=[
            {"document_name": "Shridhar_Cisco_SIP_Final_Report.pdf", "page_number": 31, "section_title": "Chapter 7"},
            {"document_name": "Shridhar_Cisco_SIP_Final_Report.pdf", "page_number": 31, "section_title": "Chapter 7"},  # Duplicate
        ],
    )

    assert res["status"] == "success"
    pdf_path = Path(res["file_path"])
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000

    # Verify deliverable file and metadata
    assert res["status"] == "success"
    assert res["filename"].endswith(".pdf")
    assert res["citation_count"] == 2


def test_orchestrator_generate_pdf_tool_dispatch(test_env):
    orchestrator = test_env["orchestrator"]
    project_id = "TOOL_PDF_PROJ"
    test_env["store"].create_project(project_id, "Tool PDF", "Test Tool PDF")

    res = orchestrator.execute_tool(
        tool_name="generate_pdf_report",
        params={
            "title": "Turnaround Engineering Summary",
            "executive_summary": "CDU-1 transfer line evaluation complete.",
        },
        project_id=project_id,
    )

    assert res["status"] == "success"
    assert Path(res["file_path"]).exists()
    assert res["filename"].endswith(".pdf")


def test_hybrid_retriever_deduplication_and_relative_pruning(test_env):
    store = test_env["store"]
    retriever = HybridKnowledgeRetriever(store)
    tmp_path = test_env["tmp_path"]

    # Ingest document with multiple sections
    doc_text = (
        "# Cisco SIP Report\n"
        "## Chapter 7 Customer Review Analytics\n"
        "Customer Review Analytics transforms uploaded reviews into insights.\n"
        "The system cleans data with Pandas and loads into PostgreSQL.\n\n"
        "## Chapter 7 Customer Review Analytics Details\n"
        "Customer review analytics includes Chart.js visualization.\n"
    )
    doc_path = tmp_path / "Cisco_Report.md"
    doc_path.write_text(doc_text, encoding="utf-8")
    retriever.ingest_file(doc_path, category="General")

    search_res = retriever.search(
        query="Customer Review Analytics Project architecture and Pandas",
        project_id="TEST_DEDUP",
        top_k=5,
    )

    results = search_res.get("results", [])
    assert len(results) > 0

    # Ensure no duplicates by (document_name, page_number, section_title)
    seen = set()
    for r in results:
        key = (r["document_name"], r["page_number"], r["section_title"])
        assert key not in seen, f"Duplicate found for key: {key}"
        seen.add(key)
