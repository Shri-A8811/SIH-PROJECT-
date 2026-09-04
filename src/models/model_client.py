"""
Model Client & Schema-Constrained Inference Engine.
Builds clean, minimal reconstructed context from the state store for each model call.
Supports Ollama JSON-constrained generation + deterministic fallback.
"""
from typing import Any, Dict, List, Optional, Generator
import os
import json
import time
import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config.settings import settings, validate_local_ollama_endpoint
from src.core.state_store import StateStore
from src.models.lifecycle import ModelLifecycleManager


class ContextBudgetManager:
    """
    Manages prompt token budgets for local LLMs with limited context windows (e.g. 4096 tokens).
    Ensures prompts leave adequate headroom for model generation without runtime overflow.
    """

    def __init__(self, default_budget_tokens: int = 3800):
        self.default_budget_tokens = default_budget_tokens
        self._encoder = None
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder:
            try:
                return len(self._encoder.encode(text, disallowed_special=()))
            except Exception:
                pass
        # Fallback heuristic: 1 token ~ 3.8 chars
        return max(1, int(len(text) / 3.8))

    def fit_prompt_within_budget(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Compresses/truncates prompt context to fit within the token budget.
        Preserves critical instruction header and schema/constraint tail while compressing
        middle evidence/context sections.
        """
        budget = max_tokens or self.default_budget_tokens
        current_tokens = self.count_tokens(prompt)
        if current_tokens <= budget:
            return prompt

        lines = prompt.split("\n")
        if len(lines) < 4:
            chars_allowed = int(budget * 3.5)
            head_chars = int(chars_allowed * 0.6)
            tail_chars = chars_allowed - head_chars
            return (
                prompt[:head_chars]
                + "\n\n[... Context compressed: middle content truncated to fit token budget ...]\n\n"
                + prompt[-tail_chars:]
            )

        target_tokens = budget - 50  # reserve tokens for truncation message
        head_lines = []
        tail_lines = []

        # Keep top instructions up to 55% of target budget
        head_tokens = 0
        head_idx = 0
        while head_idx < len(lines) and (head_tokens + self.count_tokens(lines[head_idx] + "\n")) < (target_tokens * 0.55):
            head_tokens += self.count_tokens(lines[head_idx] + "\n")
            head_lines.append(lines[head_idx])
            head_idx += 1

        # Keep bottom schema/instructions up to 35% of target budget
        tail_tokens = 0
        tail_idx = len(lines) - 1
        while tail_idx >= head_idx and (tail_tokens + self.count_tokens(lines[tail_idx] + "\n")) < (target_tokens * 0.35):
            tail_tokens += self.count_tokens(lines[tail_idx] + "\n")
            tail_lines.insert(0, lines[tail_idx])
            tail_idx -= 1

        compressed = (
            "\n".join(head_lines)
            + "\n\n[... Context compressed: middle evidence/state truncated to fit model token budget ...]\n\n"
            + "\n".join(tail_lines)
        )
        return compressed


class ModelClient:
    """Invokes local open-weight models with reconstructed context."""

    def __init__(
        self,
        state_store: StateStore,
        lifecycle_manager: ModelLifecycleManager,
        budget_manager: Optional[ContextBudgetManager] = None,
    ):
        self.state_store = state_store
        self.lifecycle_manager = lifecycle_manager
        self.ollama_base_url = validate_local_ollama_endpoint(settings.ollama_base_url)
        self.budget_manager = budget_manager or ContextBudgetManager()

        # Configure robust HTTP connection pooling for zero-overhead streaming & requests
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=Retry(total=2, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]),
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def build_reconstructed_context_prompt(self, task_contract: Dict[str, Any]) -> str:
        """
        Builds a clean, minimal prompt strictly from stored state.
        Never relies on chat history or model internal memory.
        """
        task_id = task_contract.get("task_id", "")
        objective = task_contract.get("objective", "")
        task_type = task_contract.get("task_type", "")
        inputs = task_contract.get("inputs", {})
        context = task_contract.get("context", {})
        allowed_tools = task_contract.get("allowed_tools", [])
        output_schema = task_contract.get("output_schema", {})
        last_error = task_contract.get("last_error")

        prompt_parts = [
            f"### AIR-GAPPED WORKBENCH TASK CONTRACT: {task_id}",
            f"TASK TYPE: {task_type}",
            f"OBJECTIVE: {objective}",
            "",
            "--- INPUTS ---",
            json.dumps(inputs, indent=2),
            "",
            "--- RECONSTRUCTED STATE & EVIDENCE ---",
            json.dumps(context, indent=2),
            "",
            f"ALLOWED TOOLS: {', '.join(allowed_tools) if allowed_tools else 'None'}",
            "",
            "--- REQUIRED OUTPUT JSON SCHEMA ---",
            json.dumps(output_schema, indent=2),
        ]

        if last_error:
            prompt_parts.extend([
                "",
                "--- PREVIOUS ATTEMPT VALIDATION ERROR (SELF-CORRECTION REQUIRED) ---",
                f"Your previous response was rejected by the Acceptance Gate for the following reason:",
                f"{last_error}",
                "Please correct your output strictly according to the schema and grounding rules above.",
            ])

        prompt_parts.extend([
            "",
            "INSTRUCTION: Return ONLY a valid JSON object matching the required schema. No conversational preamble.",
        ])

        return "\n".join(prompt_parts)

    def extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Robustly extracts valid JSON from model responses.
        Handles thinking models (<think>...</think>), markdown code fences (```json ... ```),
        and surrounding conversational text.
        """
        if not response_text or not response_text.strip():
            return None

        text = response_text.strip()
        import re

        # 1. Strip <think>...</think> reasoning tags (e.g. Qwen3.5, DeepSeek)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # 2. Try direct parsing
        try:
            return json.loads(text)
        except Exception:
            pass

        # 3. Extract from markdown code fences
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass

        # 4. Extract first outermost balanced { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass

        return None

    def generate_text_stream(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 1500,
        temperature: float = 0.2,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        timeout: float = 300.0,
        think: bool = False,
    ) -> Generator[str, None, None]:
        """
        Streams generated text tokens directly from Ollama as they are generated.
        This provides immediate feedback to the UI and completely prevents HTTP
        read timeouts during heavy inference.
        """
        self.lifecycle_manager.ensure_model_loaded(
            target_model=model_name,
            project_id=project_id,
            task_id=task_id,
        )

        # Fit prompt strictly within context budget
        fitted_prompt = self.budget_manager.fit_prompt_within_budget(prompt)

        if not os.getenv("WORKBENCH_TEST_MODE") and self.lifecycle_manager.is_ollama_online:
            try:
                payload = {
                    "model": model_name,
                    "prompt": fitted_prompt,
                    "stream": True,
                    "think": think,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                    "keep_alive": "10m",
                }
                resp = self.session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    for line in resp.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                chunk_json = json.loads(line)
                                chunk_str = chunk_json.get("response", "")
                                # Fallback: if model only yielded thinking tokens and response is empty
                                if not chunk_str and chunk_json.get("thinking"):
                                    chunk_str = chunk_json.get("thinking", "")
                                if chunk_str:
                                    yield chunk_str
                                if chunk_json.get("done", False):
                                    break
                            except Exception:
                                pass
                else:
                    yield f"Inference error on {model_name}: HTTP status {resp.status_code}"
            except Exception as e:
                yield f"Inference error on {model_name}: {str(e)}"
        else:
            yield f"Inference unavailable: local model '{model_name}' is not reachable. No draft was generated."

    def generate_text(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.2,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Executes raw text generation on a specific model, ensuring it is the
        only resident model on the single GPU, and logs telemetry.
        """
        load_res = self.lifecycle_manager.ensure_model_loaded(
            target_model=model_name,
            project_id=project_id,
            task_id=task_id,
        )

        start_time = time.time()
        response_text = ""

        if not os.getenv("WORKBENCH_TEST_MODE") and self.lifecycle_manager.is_ollama_online:
            try:
                chunks = []
                for chunk in self.generate_text_stream(
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    project_id=project_id,
                    task_id=task_id,
                    timeout=timeout,
                ):
                    chunks.append(chunk)
                raw_text = "".join(chunks).strip()
                import re
                response_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            except Exception as e:
                response_text = f"Inference error on {model_name}: {str(e)}"
        else:
            response_text = f"Inference unavailable: local model '{model_name}' is not reachable. No draft was generated."

        duration_ms = (time.time() - start_time) * 1000

        # 2. Log inference telemetry
        self.state_store.log_model_activity(
            model_name=model_name,
            action="INFERENCE",
            project_id=project_id,
            task_id=task_id,
            duration_ms=duration_ms,
            details={"prompt_preview": prompt[:80], "token_limit": max_tokens},
        )

        return {
            "model": model_name,
            "response": response_text,
            "load_status": load_res.get("status"),
            "load_duration_ms": load_res.get("duration_ms", 0.0),
            "inference_duration_ms": duration_ms,
            "vram_allocated_mb": load_res.get("vram_mb", 0.0),
            "status": "success",
        }

    def generate_with_images(
        self,
        model_name: str,
        prompt: str,
        image_paths: List[str],
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """Run a local vision model; image bytes are sent only to loopback Ollama."""
        if not image_paths:
            return {"status": "invalid_input", "error": "At least one image is required."}
        self.lifecycle_manager.ensure_model_loaded(model_name, project_id, task_id)
        if not self.lifecycle_manager.is_ollama_online:
            return {"status": "inference_unavailable", "error": "Local Ollama vision service is unavailable."}
        try:
            encoded_images = []
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    encoded_images.append(base64.b64encode(image_file.read()).decode("ascii"))
            response = self.session.post(
                f"{self.ollama_base_url}/api/generate",
                json={"model": model_name, "prompt": prompt, "images": encoded_images, "stream": False, "format": "json"},
                timeout=timeout,
            )
            if response.status_code != 200:
                return {"status": "inference_unavailable", "error": f"Local vision model returned HTTP {response.status_code}."}
            return {"status": "ok", "response": response.json().get("response", "")}
        except Exception as exc:
            return {"status": "inference_unavailable", "error": f"Local vision inference failed: {exc}"}

    def execute_task(
        self,
        task_contract: Dict[str, Any],
        project_id: str,
    ) -> Dict[str, Any]:
        """
        Loads the required model, executes inference with reconstructed context,
        and logs activity.
        """
        model_name = task_contract.get("assigned_model", settings.models.reasoning)
        task_id = task_contract.get("task_id", "T000")

        # 1. Ensure model is resident on single GPU
        self.lifecycle_manager.ensure_model_loaded(model_name, project_id=project_id, task_id=task_id)

        # 2. Build reconstructed prompt and fit within context budget
        prompt = self.build_reconstructed_context_prompt(task_contract)
        prompt = self.budget_manager.fit_prompt_within_budget(prompt)

        start_time = time.time()
        raw_result = None

        # 3. Try Ollama execution if not in fast test mode
        if not os.getenv("WORKBENCH_TEST_MODE") and self.lifecycle_manager.is_ollama_online:
            try:
                output_schema = task_contract.get("output_schema")
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                }
                
                # Pass JSON schema directly to Ollama for Grammar Constrained Decoding
                if output_schema:
                    payload["format"] = output_schema
                else:
                    payload["format"] = "json"

                resp = self.session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    timeout=300.0,
                )
                if resp.status_code == 200:
                    resp_json = resp.json()
                    response_text = resp_json.get("response", "")
                    raw_result = self.extract_json_from_response(response_text)
            except Exception:
                raw_result = None

        # Test-only deterministic fixture. Production must fail closed rather than
        # fabricate findings, citations, or approval recommendations.
        if raw_result is None and settings.enable_deterministic_mock_fallback:
            raw_result = self._generate_deterministic_response(task_contract, project_id)

        if raw_result is None:
            return {
                "status": "inference_unavailable",
                "error": f"Local model '{model_name}' did not return a valid schema-conformant result.",
            }

        duration_ms = (time.time() - start_time) * 1000

        # 5. Log inference telemetry
        self.state_store.log_model_activity(
            model_name=model_name,
            action="INFERENCE",
            project_id=project_id,
            task_id=task_id,
            duration_ms=duration_ms,
            details={"task_type": task_contract.get("task_type")},
        )

        return raw_result

    def _generate_deterministic_response(
        self,
        task_contract: Dict[str, Any],
        project_id: str,
    ) -> Dict[str, Any]:
        """High-fidelity deterministic response generator matching industrial PSU requirements."""
        task_type = task_contract.get("task_type", "")
        context = task_contract.get("context", {})
        evidence_ids = context.get("evidence_ids", [])

        # If evidence table has records, use valid evidence_ids
        stored_evidence = self.state_store.get_all_evidence_for_project(project_id)
        valid_e_id = stored_evidence[0].evidence_id if stored_evidence else "E001"
        valid_e_id_2 = stored_evidence[1].evidence_id if len(stored_evidence) > 1 else valid_e_id

        if task_type in ("document_analysis", "multimodal_extraction"):
            return {
                "findings": [
                    {
                        "equipment": "Crude Distillation Unit (CDU-1) Transfer Line Pipe Section P-104B",
                        "issue": "Severe ultrasonic wall thinning (residual thickness: 3.42 mm vs nominal 8.00 mm) and pitting corrosion",
                        "severity": "Critical",
                        "evidence_id": valid_e_id,
                        "measured_value": "3.42 mm",
                        "threshold_value": "4.80 mm",
                        "status": "NON-COMPLIANT",
                    },
                    {
                        "equipment": "Vacuum Gas Oil (VGO) Hydrocracker High-Pressure Flange FL-208",
                        "issue": "Micro-fissuring and trace gasket degradation observed during hydro-test at 142 bar",
                        "severity": "High",
                        "evidence_id": valid_e_id_2,
                        "measured_value": "142 bar (micro-fissures detected)",
                        "threshold_value": "Zero allowable surface fissuring",
                        "status": "REQUIRES_MAINTENANCE",
                    },
                ],
                "summary": "Visual and ultrasonic NDT inspection identified safety-critical thinning in CDU-1 line P-104B below retirement thickness and flange micro-fissuring.",
                "total_findings_count": 2,
            }

        elif task_type == "synthesis":
            return {
                "findings": [
                    {
                        "equipment": "Crude Distillation Unit (CDU-1) Transfer Line Pipe Section P-104B",
                        "issue": "Residual wall thickness of 3.42 mm is 28.75% below mandatory minimum retirement thickness (4.80 mm) under MRPL SOP-17 Sec 4.2",
                        "severity": "Critical",
                        "evidence_id": valid_e_id,
                        "measured_value": "3.42 mm",
                        "threshold_value": "4.80 mm",
                        "status": "NON-COMPLIANT",
                    },
                    {
                        "equipment": "Vacuum Gas Oil (VGO) Hydrocracker High-Pressure Flange FL-208",
                        "issue": "Gasket micro-fissuring violates MRPL SOP-04 Sec 3.1 zero-tolerance policy for Class 1500 hydrocracker service",
                        "severity": "High",
                        "evidence_id": valid_e_id_2,
                        "measured_value": "142 bar hydro-test",
                        "threshold_value": "0 fissures",
                        "status": "REQUIRES_MAINTENANCE",
                    },
                ],
                "executive_summary": "Comprehensive engineering evaluation of MRPL turnaround inspection report. CDU-1 line P-104B requires immediate emergency spool replacement prior to unit restart.",
                "recommended_actions": [
                    "Immediate de-inventorying and isolation of CDU-1 line P-104B for spool fabrication and weld inspection.",
                    "Torque verification and spiral-wound gasket replacement on VGO Flange FL-208.",
                    "NDT re-certification prior to hydrostatic recommissioning.",
                ],
                "sop_references": ["MRPL-SOP-17 (Pressure Vessel & Piping Integrity)", "MRPL-SOP-04 (High-Pressure Flanges)"],
            }

        elif task_type == "code_execution":
            return {
                "code": "def verify_thickness(measured, threshold):\n    deviation = ((threshold - measured) / threshold) * 100\n    return {'deviation_pct': round(deviation, 2), 'is_critical': measured < threshold}\n\nprint(verify_thickness(3.42, 4.80))",
                "language": "python",
                "explanation": "Generates verified deterministic calculation script for wall thinning percentage deviation.",
            }

        return {"status": "ok", "message": "Processed successfully"}
