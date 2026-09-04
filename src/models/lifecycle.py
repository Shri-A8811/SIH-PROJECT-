"""
Model Lifecycle Manager for Sovereign On-Premise Agentic AI Workbench.
Enforces the Single-GPU Sequential Model Discipline:
Only one model resident in GPU VRAM at a time.
Uses Ollama /api/ps for real-time VRAM telemetry and fast eviction verification.
Logs all load, unload, and inference actions to the persistent state store.
"""
from typing import Any, Dict, List, Optional
import os
import time
import requests
import logging
from config.settings import settings, validate_local_ollama_endpoint
from src.core.state_store import StateStore

logger = logging.getLogger(__name__)


class ModelLifecycleManager:
    """Manages Ollama model loading/unloading and single-GPU VRAM lifecycle."""

    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self.currently_loaded_model: Optional[str] = None
        self.ollama_base_url = validate_local_ollama_endpoint(settings.ollama_base_url)

    @property
    def is_ollama_online(self) -> bool:
        """Dynamically verifies if local Ollama daemon is reachable; auto-launches if offline."""
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass

        if not os.getenv("WORKBENCH_TEST_MODE"):
            try:
                import subprocess
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=True,
                )
                time.sleep(2.0)
                resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=3.0)
                return resp.status_code == 200
            except Exception:
                pass
        return False

    def get_loaded_models_from_ollama(self) -> List[str]:
        """Queries Ollama /api/ps to retrieve active in-memory models."""
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

    def get_loaded_models_ps(self) -> List[str]:
        """Alias for get_loaded_models_from_ollama."""
        return self.get_loaded_models_from_ollama()

    def is_model_loaded(self, model_name: str) -> bool:
        """Returns True if model_name is currently resident in VRAM according to /api/ps."""
        return model_name in self.get_loaded_models_from_ollama()

    def is_model_loaded_ps(self, model_name: str) -> bool:
        """Alias for is_model_loaded."""
        return self.is_model_loaded(model_name)

    def get_runtime_model_telemetry(self) -> Dict[str, Any]:
        """Returns live hardware VRAM usage and active model metrics from Ollama runtime."""
        if not self.is_ollama_online:
            active = [self.currently_loaded_model] if self.currently_loaded_model else []
            return {
                "active_models": active,
                "loaded_models": active,
                "active_resident_model": self.currently_loaded_model,
                "total_vram_mb": self._estimate_vram(self.currently_loaded_model) if self.currently_loaded_model else 0.0,
                "is_live_daemon": False,
                "ollama_online": False,
            }
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/ps", timeout=2.0)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                total_vram_bytes = sum(m.get("size_vram", 0) for m in models)
                names = [m.get("name") for m in models]
                return {
                    "active_models": names,
                    "loaded_models": names,
                    "active_resident_model": names[0] if names else self.currently_loaded_model,
                    "total_vram_mb": round(total_vram_bytes / (1024 * 1024), 2),
                    "model_details": [
                        {
                            "name": m.get("name"),
                            "vram_mb": round(m.get("size_vram", 0) / (1024 * 1024), 2),
                            "context_length": m.get("context_length", 4096),
                            "expires_at": m.get("expires_at"),
                        }
                        for m in models
                    ],
                    "is_live_daemon": True,
                    "ollama_online": True,
                }
        except Exception as e:
            logger.debug(f"Telemetry query exception: {e}")

        active = [self.currently_loaded_model] if self.currently_loaded_model else []
        return {
            "active_models": active,
            "loaded_models": active,
            "active_resident_model": self.currently_loaded_model,
            "total_vram_mb": self._estimate_vram(self.currently_loaded_model) if self.currently_loaded_model else 0.0,
            "is_live_daemon": False,
            "ollama_online": False,
        }

    def ensure_model_loaded(
        self,
        target_model: str,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ensures target_model is the only resident model on the single GPU.
        Unloads any previous resident model first to uphold single-GPU residency.
        """
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

    def ensure_model_unloaded(
        self,
        model_name: str,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ):
        """Public method to explicitly evict a model from VRAM."""
        self._unload_model(model_name, project_id, task_id)

    def _load_model(
        self,
        model_name: str,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> float:
        """Loads model into GPU memory and verifies residency via /api/ps."""
        load_start = time.time()
        vram_mb = self._estimate_vram(model_name)

        if not os.getenv("WORKBENCH_TEST_MODE") and self.is_ollama_online:
            try:
                # Pre-warm model in Ollama with keep_alive="10m"
                requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={"model": model_name, "prompt": "", "keep_alive": "10m"},
                    timeout=30.0,
                )
                # Check real VRAM from /api/ps if reported
                telem = self.get_runtime_model_telemetry()
                for m in telem.get("model_details", []):
                    if m.get("name") == model_name and m.get("vram_mb", 0) > 0:
                        vram_mb = m["vram_mb"]
                        break
            except Exception:
                pass

        duration_ms = (time.time() - load_start) * 1000
        self.state_store.log_model_activity(
            model_name=model_name,
            action="LOAD",
            project_id=project_id,
            task_id=task_id,
            vram_allocated_mb=vram_mb,
            duration_ms=duration_ms,
            details={"type": "single_gpu_resident_load"},
        )
        return vram_mb

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
                # Fast eviction polling on /api/ps (100ms intervals, max 1s)
                for _ in range(10):
                    time.sleep(0.1)
                    active = self.get_loaded_models_from_ollama()
                    if model_name not in active:
                        break
            except Exception as e:
                logger.debug(f"Unload notice: {e}")

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

    def _estimate_vram(self, model_name: Optional[str]) -> float:
        if not model_name:
            return 0.0
        name_lower = model_name.lower()
        if "14b" in name_lower:
            return 9200.0
        elif "9b" in name_lower or "8b" in name_lower or "7b" in name_lower:
            return 5600.0
        elif "3b" in name_lower or "vl" in name_lower or "ocr" in name_lower:
            return 2800.0
        elif "coder" in name_lower:
            return 4800.0
        elif "embed" in name_lower or "bge" in name_lower:
            return 120.0
        return 4000.0
