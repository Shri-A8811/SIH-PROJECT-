"""
Task Router for Sovereign On-Premise Agentic AI Workbench.
A high-speed rule-based and intent-aware task classifier based on file types,
keywords, and structured task intent.
Identifies single-step versus multi-step workflows and selects the optimal specialist model upfront.
"""
from typing import Any, Dict, List, Optional
import os
import re
from pydantic import BaseModel, Field
from config.settings import settings


class RoutingDecision(BaseModel):
    selected_path: str  # multimodal, coding, reasoning, calculation
    assigned_model: str
    starting_task_type: str
    confidence: float
    rationale: str
    recommended_tools: List[str]
    plan_type: str = "single_step"  # single_step, multi_step_workflow
    detected_intents: List[str] = Field(default_factory=list)
    suggested_tools: List[str] = Field(default_factory=list)


class TaskRouter:
    """Heuristic, rule-based task classifier and workflow planner."""

    MULTIMODAL_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    CODE_KEYWORDS = {
        "code", "script", "python", "docker", "sandbox", "print", "import",
        "debug", "function", "sql", "query", "algorithm", "benchmark"
    }
    CALC_KEYWORDS = {
        "calculate", "computation", "deviation", "percentage", "arithmetic",
        "formula", "threshold comparison", "sum", "average", "pressure drop", "corrosion rate", "breach margin"
    }
    MULTIMODAL_KEYWORDS = {
        "scanned", "image", "diagram", "p&id", "drawing", "nameplate",
        "gauge", "ocr", "inspection report", "visual", "schematic"
    }

    DOCUMENT_KEYWORDS = {
        "pdf", "docx", "approval note", "generate report", "make pdf", "make me pdf",
        "create pdf", "export to pdf", "download pdf", "generate document"
    }

    def route_request(
        self,
        user_prompt: str,
        uploaded_file_path: Optional[str] = None,
        context_hint: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Determines the appropriate initial model and task path."""
        prompt_lower = user_prompt.lower()
        file_ext = ""
        if uploaded_file_path:
            file_ext = os.path.splitext(uploaded_file_path)[1].lower()

        def contains_keyword(text: str, keywords: set) -> bool:
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    return True
            return False

        detected_intents = []
        if file_ext in self.MULTIMODAL_EXTENSIONS or contains_keyword(prompt_lower, self.MULTIMODAL_KEYWORDS):
            detected_intents.append("multimodal")
        if contains_keyword(prompt_lower, self.CODE_KEYWORDS) or "run script" in prompt_lower or "run python" in prompt_lower:
            detected_intents.append("coding")
        if contains_keyword(prompt_lower, self.CALC_KEYWORDS):
            detected_intents.append("calculation")
        if contains_keyword(prompt_lower, self.DOCUMENT_KEYWORDS):
            detected_intents.append("document_generation")

        # Explicit multi-step indicator
        is_explicit_multi_step = (
            "multi-step" in prompt_lower
            or "multi step" in prompt_lower
            or "and calculate" in prompt_lower
            or "and generate" in prompt_lower
            or "then calculate" in prompt_lower
            or "then run" in prompt_lower
            or ("from" in prompt_lower and contains_keyword(prompt_lower, self.DOCUMENT_KEYWORDS))
        )
        plan_type = "multi_step_workflow" if is_explicit_multi_step else "single_step"

        # Explicit multi-step route overrides single-step fast-paths
        if is_explicit_multi_step:
            start_task = "document_generation" if "document_generation" in detected_intents else "synthesis"
            return RoutingDecision(
                selected_path="reasoning",
                assigned_model=settings.models.reasoning,
                starting_task_type=start_task,
                confidence=0.92,
                rationale="Detected compound multi-step workflow. Activating autonomous ReAct cognitive loop.",
                recommended_tools=["knowledge_search", "calculate_wall_thinning", "execute_python_code", "generate_pdf_report", "generate_docx_report"],
                plan_type="multi_step_workflow",
                detected_intents=detected_intents or ["reasoning"],
                suggested_tools=["knowledge_search", "calculate_wall_thinning", "generate_pdf_report"],
            )

        # 1. Check for multimodal scanned inputs / images
        if file_ext in self.MULTIMODAL_EXTENSIONS or contains_keyword(prompt_lower, self.MULTIMODAL_KEYWORDS):
            assigned_m = settings.models.ocr if "ocr" in prompt_lower else settings.models.vision
            return RoutingDecision(
                selected_path="multimodal",
                assigned_model=assigned_m,
                starting_task_type="multimodal_extraction",
                confidence=0.95,
                rationale=f"Detected multimodal document input (ext='{file_ext}' / visual keywords). Routing to OCR/Vision specialist.",
                recommended_tools=["file.read", "multimodal.extract"],
                plan_type=plan_type,
                detected_intents=detected_intents or ["multimodal"],
                suggested_tools=["knowledge_search", "calculate_wall_thinning"],
            )

        # 2. Check for PDF / Document generation requests
        if contains_keyword(prompt_lower, self.DOCUMENT_KEYWORDS):
            return RoutingDecision(
                selected_path="reasoning",
                assigned_model=settings.models.reasoning,
                starting_task_type="document_generation",
                confidence=0.92,
                rationale="Detected document / PDF report generation request. Routing to autonomous synthesizer with PDF generator.",
                recommended_tools=["generate_pdf_report", "generate_docx_report", "knowledge_search"],
                plan_type="multi_step_workflow" if "from" in prompt_lower or "section" in prompt_lower else "single_step",
                detected_intents=detected_intents or ["document_generation"],
                suggested_tools=["knowledge_search", "generate_pdf_report"],
            )

        # 3. Check for explicit coding / sandbox requests
        if (contains_keyword(prompt_lower, self.CODE_KEYWORDS) or "run in sandbox" in prompt_lower or "run sandbox" in prompt_lower) and "approval note" not in prompt_lower and "sop" not in prompt_lower:
            is_generation = any(kw in prompt_lower for kw in [
                "give me code", "write code", "write a code", "write me a code", "implement",
                "create code", "generate code", "show code", "code for", "function for",
                "algorithm for", "b tree", "btree", "binary tree", "how to code"
            ])
            return RoutingDecision(
                selected_path="coding",
                assigned_model=settings.models.coding,
                starting_task_type="code_generation" if is_generation else "code_execution",
                confidence=0.90,
                rationale="Detected coding request. Routing to Coding specialist model." if is_generation else "Detected script execution request. Routing to Coding specialist sandbox.",
                recommended_tools=["sandbox.run", "file.write"],
                plan_type="single_step",
                detected_intents=detected_intents or ["coding"],
                suggested_tools=["execute_python_code"],
            )

        # 3. Check for arithmetic / calculation requests
        if contains_keyword(prompt_lower, self.CALC_KEYWORDS) and len(prompt_lower.split()) < 40:
            return RoutingDecision(
                selected_path="calculation",
                assigned_model="deterministic_calculator",
                starting_task_type="calculation",
                confidence=0.92,
                rationale="Detected arithmetic calculation. Routing directly to Deterministic Calculator tool.",
                recommended_tools=["calculator.compute"],
                plan_type=plan_type,
                detected_intents=detected_intents or ["calculation"],
                suggested_tools=["calculate_wall_thinning", "evaluate_expression"],
            )

        # 4. Default: General reasoning, planning, and synthesis
        return RoutingDecision(
            selected_path="reasoning",
            assigned_model=settings.models.reasoning,
            starting_task_type="synthesis",
            confidence=0.88,
            rationale="Defaulting to General Reasoning model for structured workflow planning, retrieval orchestration, and synthesis.",
            recommended_tools=["knowledge.search", "calculator.compute", "docx.generate"],
            plan_type=plan_type,
            detected_intents=detected_intents or ["reasoning"],
            suggested_tools=["knowledge_search", "calculate_wall_thinning", "generate_docx_report"],
        )
