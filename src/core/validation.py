"""
Validation Gate & Grounding Engine for Sovereign On-Premise Agentic AI Workbench.
Enforces:
1. Structural schema compliance.
2. Semantic sanity (non-empty meaningful fields, realistic ranges).
3. Evidence Grounding: Every factual finding MUST resolve to a real evidence_id in the state store.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.core.state_store import StateStore


class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None

    @property
    def formatted_error(self) -> str:
        if not self.errors:
            return "No errors."
        return "\n".join([f"- [FIELD/GROUNDING ERROR]: {e}" for e in self.errors])


class AcceptanceGate:
    """Rigorous acceptance gate gating entry into the persistent state store."""

    def __init__(self, state_store: StateStore):
        self.state_store = state_store

    def validate_task_result(
        self,
        task_contract: Dict[str, Any],
        raw_result: Any,
        project_id: str,
    ) -> ValidationResult:
        """
        Validates raw task output against its contract output schema,
        checks semantic sanity, and enforces evidence_id resolution.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if raw_result is None:
            return ValidationResult(is_valid=False, errors=["Task result cannot be None/empty."])

        # 1. Structural Check
        if not isinstance(raw_result, dict):
            return ValidationResult(
                is_valid=False,
                errors=[f"Expected JSON object / dict output, got type: {type(raw_result).__name__}"],
            )

        task_type = task_contract.get("task_type", "")
        expected_schema = task_contract.get("output_schema", {})

        # Check required top-level keys from schema if specified
        if "required" in expected_schema and isinstance(expected_schema["required"], list):
            for req_key in expected_schema["required"]:
                if req_key not in raw_result:
                    errors.append(f"Missing required top-level key '{req_key}' in model output.")

        # 2. Semantic Sanity & Grounding by Task Type
        if task_type in ("document_analysis", "multimodal_extraction", "synthesis"):
            findings = raw_result.get("findings", [])
            if not isinstance(findings, list):
                errors.append("Field 'findings' must be a list.")
            elif len(findings) == 0:
                # Semantic sanity: an industrial report parsing cannot return 0 findings silently
                errors.append(
                    "Semantic Sanity Failed: 'findings' list is empty. An inspection analysis must extract observed equipment findings or explicitly report status."
                )
            else:
                for idx, finding in enumerate(findings):
                    if not isinstance(finding, dict):
                        errors.append(f"Finding at index {idx} is not an object.")
                        continue

                    # Check essential fields
                    for field in ("equipment", "issue", "severity"):
                        val = finding.get(field)
                        if not val or not str(val).strip():
                            errors.append(f"Finding[{idx}] is missing required field '{field}'.")

                    # Check Evidence Grounding
                    evidence_id = finding.get("evidence_id")
                    if not evidence_id:
                        errors.append(
                            f"Finding[{idx}] ('{finding.get('equipment', 'unknown')}') has no 'evidence_id'. All factual claims must be grounded in verified evidence."
                        )
                    else:
                        evidence_record = self.state_store.get_evidence(evidence_id)
                        if not evidence_record:
                            errors.append(
                                f"Grounding Check Failed: finding[{idx}] cites non-existent evidence_id '{evidence_id}'. Fabricated citations are strictly rejected."
                            )
                        elif evidence_record.project_id != project_id:
                            errors.append(
                                f"Grounding Security Violation: evidence_id '{evidence_id}' belongs to project '{evidence_record.project_id}', not '{project_id}'."
                            )

        elif task_type == "calculation":
            if "calculated_results" not in raw_result and "result" not in raw_result:
                errors.append("Calculation result must contain 'calculated_results' or 'result' key.")
            if "audit_trail" not in raw_result:
                warnings.append("Calculation completed without an explicit audit_trail.")

        elif task_type == "retrieval":
            results = raw_result.get("results", [])
            grounding_status = raw_result.get("grounding_status", "matched")
            if not isinstance(results, list):
                errors.append("Retrieval 'results' must be a list.")
            if grounding_status == "unmatched":
                warnings.append(
                    "Grounding status is explicitly 'unmatched' due to low reranker score. Caveat flagged."
                )

        elif task_type == "document_generation":
            file_path = raw_result.get("file_path")
            if not file_path:
                errors.append("Document generation must return a 'file_path'.")
            if not raw_result.get("human_review_disclaimer_included"):
                errors.append("Mandatory requirement: 'human_review_disclaimer_included' must be True.")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            sanitized_data=raw_result if is_valid else None,
        )
