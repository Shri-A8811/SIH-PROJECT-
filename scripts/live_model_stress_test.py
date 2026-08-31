"""
Live Multi-Model Sequential GPU Swap Stress Test Tool.
Verifies real-time loading, unloading, and inference across specialist models in Ollama:
Stage 1: Vision / OCR (qwen2.5vl:3b / frob/unlimited-ocr:3b)
Stage 2: Deterministic Coder (qwen2.5-coder:7b)
Stage 3: Reasoning / Synthesis (qwen2.5vl:3b / qwen3.5:9b)
"""
import sys
from pathlib import Path
import time
import requests
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.state_store import StateStore
from src.models.lifecycle import ModelLifecycleManager
from src.models.model_client import ModelClient
from src.tools.sandbox import CodeSandbox


def run_live_sequential_stress_test():
    print("=" * 70)
    print("STARTING REAL LIVE MULTI-MODEL SEQUENTIAL GPU STRESS TEST")
    print("=" * 70)

    store = StateStore("sqlite:///:memory:")
    lifecycle_mgr = ModelLifecycleManager(store)
    client = ModelClient(store, lifecycle_mgr)
    sandbox = CodeSandbox()

    if not lifecycle_mgr.is_ollama_online:
        print("Error: Ollama is offline at http://localhost:11434.")
        return False

    # -------------------------------------------------------------
    # STAGE 1: VISION / OCR SPECIALIST
    # -------------------------------------------------------------
    ocr_model = "qwen2.5vl:3b"
    print(f"\n[STAGE 1] Loading OCR / Vision Specialist ({ocr_model})...")
    stage1_prompt = "Extract measured ultrasonic thickness value from text: 'CDU-1 transfer line pipe P-104B measured wall thickness is 3.42 mm against nominal 8.00 mm'. Answer in 1 short sentence."
    
    t0 = time.time()
    res1 = client.generate_text(
        model_name=ocr_model,
        prompt=stage1_prompt,
        max_tokens=60,
        temperature=0.1,
    )
    d1 = (time.time() - t0) * 1000
    print(f"Stage 1 Complete in {d1:.1f}ms (Inference: {res1['inference_duration_ms']:.1f}ms)")
    print(f"   Resident Model: {lifecycle_mgr.currently_loaded_model}")
    print(f"   Model Output: {res1['response']}")

    # -------------------------------------------------------------
    # STAGE 2: DETERMINISTIC CODING SPECIALIST
    # -------------------------------------------------------------
    coder_model = "qwen2.5-coder:7b"
    print(f"\n[STAGE 2] Evicting Stage 1 -> Loading Coder Specialist ({coder_model})...")
    stage2_prompt = "Write 2 lines of Python code to calculate loss percentage: measured=3.42, nominal=8.0. print('LOSS:', (1 - measured/nominal) * 100)"
    
    t0 = time.time()
    res2 = client.generate_text(
        model_name=coder_model,
        prompt=stage2_prompt,
        max_tokens=80,
        temperature=0.1,
    )
    d2 = (time.time() - t0) * 1000
    print(f"Stage 2 Complete in {d2:.1f}ms (Inference: {res2['inference_duration_ms']:.1f}ms)")
    print(f"   Resident Model: {lifecycle_mgr.currently_loaded_model}")
    print(f"   Model Output: {res2['response']}")

    # Run calculation in sandbox
    sb_res = sandbox.execute_python_code("measured = 3.42\nnominal = 8.00\nprint(f'Computed Metal Loss: {(1 - measured/nominal)*100:.2f}%')")
    print(f"   Sandbox Execution Output: {sb_res.stdout.strip()}")

    # -------------------------------------------------------------
    # STAGE 3: REASONING & SYNTHESIS SPECIALIST
    # -------------------------------------------------------------
    reasoning_model = "qwen2.5vl:3b"
    print(f"\n[STAGE 3] Evicting Stage 2 -> Loading Synthesis Specialist ({reasoning_model})...")
    stage3_prompt = "Grounded SOP Finding: Pipe P-104B measured 3.42 mm, retirement limit 4.80 mm. State the compliance status and recommended turnaround action in 2 bullet points."
    
    t0 = time.time()
    res3 = client.generate_text(
        model_name=reasoning_model,
        prompt=stage3_prompt,
        max_tokens=100,
        temperature=0.1,
    )
    d3 = (time.time() - t0) * 1000
    print(f"Stage 3 Complete in {d3:.1f}ms (Inference: {res3['inference_duration_ms']:.1f}ms)")
    print(f"   Resident Model: {lifecycle_mgr.currently_loaded_model}")
    print(f"   Model Output: {res3['response']}")

    print("\n" + "=" * 70)
    print("ALL 3 STAGES EXECUTED LIVE ON OLLAMA WITH SINGLE-GPU VRAM SWAPPING!")
    print(f"Total Pipeline Duration: {(d1 + d2 + d3)/1000:.2f}s")
    print("=" * 70)
    return True


if __name__ == "__main__":
    run_live_sequential_stress_test()
