"""
Verification script for user test prompts:
1. "write me a code for b tree oprations" (Code generation vs raw sandbox execution)
2. "make me pdf file from project section" (Automated PDF synthesis and deliverable compilation)
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.core.orchestrator import AgenticOrchestrator as SovereignOrchestrator
from src.tools.sandbox import CodeSandbox
from src.router.task_router import TaskRouter as SovereignTaskRouter

def test_coding_guard_and_route():
    print("=" * 70)
    print("TEST 1: Verifying Sandbox Prompt Guard & Coding Route")
    print("=" * 70)
    
    sandbox = CodeSandbox()
    test_prompts = [
        "write me a code for b tree oprations",
        "write me a code for b tree",
        "give me code for b tree operations",
    ]
    
    for p in test_prompts:
        sb_res = sandbox.execute_python_code(p)
        print(f"\nPrompt: '{p}'")
        print(f"  • Exit Code: {sb_res.exit_code}")
        print(f"  • Backend: {sb_res.sandbox_backend}")
        print(f"  • Stderr Preview: {sb_res.stderr[:80]}...")
        assert sb_res.exit_code == -1, f"Expected prompt guard exit code -1, got {sb_res.exit_code}"
        assert "prompt_guard" in sb_res.sandbox_backend, f"Expected prompt_guard backend, got {sb_res.sandbox_backend}"
        print("  ✅ Safely blocked natural language from raw Python execution!")

    router = SovereignTaskRouter()
    route = router.route_request("write me a code for b tree oprations")
    print(f"\nRouter Decision for '{p}':")
    print(f"  • Selected Path: {route.selected_path}")
    print(f"  • Assigned Model: {route.assigned_model}")
    print(f"  • Starting Task Type: {route.starting_task_type}")
    assert route.selected_path == "coding"
    assert "qwen2.5-coder" in route.assigned_model
    print("  ✅ Cleanly routed to coding specialist (qwen2.5-coder)!")

def test_pdf_deliverable_generation():
    print("\n" + "=" * 70)
    print("TEST 2: Verifying 'make me pdf file from project section'")
    print("=" * 70)
    
    orchestrator = SovereignOrchestrator()
    user_prompt = "make me pdf file from project section"
    project_id = "VERIFY_PDF_001"
    
    print(f"Running autonomous plan loop for: '{user_prompt}'...")
    events = []
    generated_deliverable = None
    
    for event in orchestrator.run_autonomous_plan_loop_stream(
        user_prompt=user_prompt,
        project_id=project_id,
        max_steps=5,
    ):
        events.append(event)
        event_type = event.get("type")
        if event_type == "thought":
            print(f"  💭 Thought (Step {event.get('step')}): {event.get('content')}")
        elif event_type == "tool_call":
            print(f"  🔧 Tool Call: `{event.get('tool')}` -> {event.get('input').keys() if isinstance(event.get('input'), dict) else event.get('input')}")
        elif event_type == "tool_result":
            print(f"  📊 Tool Result: `{event.get('tool')}` finished successfully.")
        elif event_type == "completed":
            generated_deliverable = event.get("generated_deliverable")
            print(f"\n  ✅ Autonomous Loop Completed!")
            print(f"  • Generated Deliverable: {generated_deliverable}")
            print(f"  • Citations Count: {len(event.get('citations', []))}")
            print(f"  • Response Preview: {event.get('final_response', '')[:200]}...")

    assert generated_deliverable is not None, "Deliverable file path must not be None!"
    pdf_path = Path(generated_deliverable)
    assert pdf_path.exists(), f"PDF deliverable does not exist at {pdf_path}"
    assert pdf_path.suffix == ".pdf", f"Expected .pdf extension, got {pdf_path.suffix}"
    assert pdf_path.stat().st_size > 1000, f"PDF file size too small: {pdf_path.stat().st_size} bytes"
    print(f"\n  🎉 SUCCESS: Verified generated PDF at '{pdf_path.name}' ({pdf_path.stat().st_size} bytes)!")

if __name__ == "__main__":
    try:
        test_coding_guard_and_route()
        test_pdf_deliverable_generation()
        print("\n" + "=" * 70)
        print("ALL VERIFICATIONS PASSED SUCCESSFULLY!")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
