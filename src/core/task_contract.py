"""
Standard Task Contract definition for Sovereign On-Premise Agentic AI Workbench.
Every unit of work handed to a model or tool follows this exact schema.
Specialist models never need another model's internal reasoning or hidden state.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class FindingSchema(BaseModel):
    equipment: str = Field(description="Equipment or component identifier e.g. Crude Distillation Unit Column C-101")
    issue: str = Field(description="Observed anomaly, corrosion, leak, or wear")
    severity: str = Field(description="Safety severity: Critical, High, Medium, Low")
    evidence_id: str = Field(description="Required pointer to verified evidence record (e.g. E001)")
    measured_value: Optional[str] = Field(default=None, description="Quantitative measurement if available")
    threshold_value: Optional[str] = Field(default=None, description="Allowable safety threshold from SOP")
    status: Optional[str] = Field(default="NON-COMPLIANT", description="COMPLIANT, NON-COMPLIANT, or REQUIRES_MAINTENANCE")


class TaskContract(BaseModel):
    task_id: str = Field(description="Unique Task identifier, e.g. T001")
    project_id: str = Field(default="default_project", description="Associated Project ID")
    task_type: str = Field(
        description="Type of task: document_analysis, retrieval, calculation, code_execution, synthesis, document_generation, verification"
    )
    objective: str = Field(description="Unambiguous instruction of what must be accomplished")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters, document IDs, file paths")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reconstructed context: evidence_ids, prior_task_ids, and minimal prompt state",
    )
    assigned_model: str = Field(
        description="Model tag or tool name assigned to this task (e.g. qwen3.5:9b, frob/unlimited-ocr:3b, deterministic_calc)"
    )
    allowed_tools: List[str] = Field(
        default_factory=list,
        description="Restricted list of allowed tools (e.g. ['knowledge.search', 'calculator.compute', 'sandbox.run'])",
    )
    output_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema or specification that the returned result MUST conform to",
    )
    retry_count: int = Field(default=0, description="Number of validation retries executed")
    last_error: Optional[str] = Field(default=None, description="Detailed validation error message if retrying")
    status: str = Field(default="pending", description="pending, running, completed, failed, escalated")
    created_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
