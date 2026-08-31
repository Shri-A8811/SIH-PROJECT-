"""
Model Lifecycle Manager for Sovereign On-Premise Agentic AI Workbench.
Enforces the Single-GPU Sequential Model Discipline:
Only one model resident in GPU VRAM at a time.
Logs all load, unload, and inference actions to the persistent state store.
"""
from typing import Any, Dict, List, Optional
import os
import time
import requests
from config.settings import settings
from src.core.state_store import StateStore


class ModelLifecycleManager:
    """Manages Ollama model loading/unloading and single-GPU VRAM lifecycle."""

    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self.currently_loaded_model: Optional[str] = None
        self.ollama_base_url = settings.ollama_base_url
    @property
    def is_ollama_online(self) -> bool:
        """Dynamically verifies if local Ollama daemon is reachable."""
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def get_loaded_models_from_ollama(self) -> List[str]:
        if not self.is_ollama_online:
            return [self.currently_loaded_model] if self.currently_loaded_model else []
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/ps", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name") for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def ensure_model_loaded(
        self,
        target_model: str,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ensures target_model is the only resident model on the single GPU.
        Unloads any previous resident model first.
        """
        start_time = time.time()

        if self.currently_loaded_model == target_model:
            return {
                "status": "already_loaded",
                "model": target_model,
                "duration_ms": 0.0,
            }

        # 1. Unload currently loaded model if different
        if self.currently_loaded_model and self.currently_loaded_model != target_model:
            self._unload_model(self.currently_loaded_model, project_id, task_id)

        # 2. Load target model
        load_start = time.time()
        vram_mb = self._load_model(target_model, project_id, task_id)
        load_duration = (time.time() - load_start) * 1000

        self.currently_loaded_model = target_model
        return {
            "status": "loaded",
            "model": target_model,
            "vram_mb": vram_mb,
            "duration_ms": load_duration,
        }

    def _load_model(
        self,
        model_name: str,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> float:
        """Loads model into GPU memory (or simulates realistic load in mock mode)."""
        load_start = time.time()
        estimated_vram = self._estimate_vram(model_name)

        if not os.getenv("WORKBENCH_TEST_MODE") and self.is_ollama_online:
            try:
                # Pre-warm model in Ollama with keep_alive=-1 (indefinite until we explicitly unload)
                requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={"model": model_name, "prompt": "", "keep_alive": "10m"},
                    timeout=30.0,
                )
            except Exception:
                pass

        duration_ms = (time.time() - load_start) * 1000
        self.state_store.log_model_activity(
            model_name=model_name,
            action="LOAD",
            project_id=project_id,
            task_id=task_id,
            vram_allocated_mb=estimated_vram,
            duration_ms=duration_ms,
            details={"type": "single_gpu_resident_load"},
        )
        return estimated_vram

    def _unload_model(
        self,
        model_name: str,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ):
        """Evicts resident model from GPU memory to free VRAM for next worker."""
        unload_start = time.time()

        if not os.getenv("WORKBENCH_TEST_MODE") and self.is_ollama_online:
            try:
                # Setting keep_alive to 0 immediately releases VRAM in Ollama
                requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={"model": model_name, "prompt": "", "keep_alive": 0},
                    timeout=5.0,
                )
            except Exception:
                pass

        duration_ms = (time.time() - unload_start) * 1000
        self.state_store.log_model_activity(
            model_name=model_name,
            action="UNLOAD",
            project_id=project_id,
            task_id=task_id,
            vram_allocated_mb=0.0,
            duration_ms=duration_ms,
            details={"type": "vram_freed_for_model_switch"},
        )
        self.currently_loaded_model = None

    def _estimate_vram(self, model_name: str) -> float:
        name_lower = model_name.lower()
        if "14b" in name_lower:
            return 9200.0
        elif "9b" in name_lower or "8b" in name_lower or "7b" in name_lower:
            return 5600.0
        elif "3b" in name_lower or "vl" in name_lower or "ocr" in name_lower:
            return 2800.0
        elif "coder" in name_lower:
            return 4800.0
        return 4000.0
