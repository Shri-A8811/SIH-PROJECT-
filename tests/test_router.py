"""
Tests for Heuristic Task Router.
"""
from src.router.task_router import TaskRouter


def test_task_router_multimodal_path():
    router = TaskRouter()
    
    # 1. Scanned file extension
    decision = router.route_request("Extract thickness readings", uploaded_file_path="report.pdf")
    assert decision.selected_path == "multimodal"
    assert decision.starting_task_type == "multimodal_extraction"

    # 2. Image keywords
    decision2 = router.route_request("Review this scanned inspection report image for P&ID tags")
    assert decision2.selected_path == "multimodal"


def test_task_router_coding_path():
    router = TaskRouter()
    decision = router.route_request("Write a python script to run in sandbox to calculate density")
    assert decision.selected_path == "coding"
    assert decision.starting_task_type == "code_execution"


def test_task_router_calculation_path():
    router = TaskRouter()
    decision = router.route_request("calculate 8.0 - 3.42")
    assert decision.selected_path == "calculation"
    assert decision.starting_task_type == "calculation"


def test_task_router_default_reasoning():
    router = TaskRouter()
    decision = router.route_request("Evaluate internal SOP compliance and synthesize turnaround executive summary")
    assert decision.selected_path == "reasoning"
    assert decision.starting_task_type == "synthesis"
