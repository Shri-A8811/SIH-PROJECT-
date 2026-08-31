"""
Task Router for Sovereign On-Premise Agentic AI Workbench.
A deliberately simple, high-speed rule-based classifier based on file types,
keywords, and structured task intent.
Picks the right starting model upfront to avoid wasteful multi-model load costs.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from config.settings import settings


class RoutingDecision(BaseModel):
    selected_path: str  # multimodal, coding, reasoning, calculation
    assigned_model: str
    starting_task_type: str
    confidence: float
    rationale: str
    recommended_tools: List[str]


class TaskRouter:
    """Heuristic, rule-based task classifier."""

    MULTIMODAL_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    CODE_KEYWORDS = {
        "code", "script", "python", "docker", "sandbox", "run", "test",
        "debug", "function", "sql", "query", "algorithm", "benchmark"
    }
    CALC_KEYWORDS = {
        "calculate", "computation", "deviation", "percentage", "arithmetic",
        "formula", "threshold comparison", "sum", "average", "pressure drop"
    }
    MULTIMODAL_KEYWORDS = {
        "scanned", "image", "diagram", "p&id", "drawing", "nameplate",
        "gauge", "ocr", "inspection report", "visual", "schematic"
    }

    def route_request(
        self,
        user_prompt: str,
        uploaded_file_path: Optional[str] = None,
        context_hint: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Determines the appropriate initial model and task path."""
        import re
        import os
        prompt_lower = user_prompt.lower()
        file_ext = ""
        if uploaded_file_path:
            file_ext = os.path.splitext(uploaded_file_path)[1].lower()

        def contains_keyword(text: str, keywords: set) -> bool:
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    return True
            return False

        # 1. Check for multimodal scanned inputs / images
        if file_ext in self.MULTIMODAL_EXTENSIONS or contains_keyword(prompt_lower, self.MULTIMODAL_KEYWORDS):
            return RoutingDecision(
                selected_path="multimodal",
                assigned_model=settings.models.ocr if "ocr" in prompt_lower else settings.models.vision,
                starting_task_type="multimodal_extraction",
                confidence=0.95,
                rationale=f"Detected multimodal document input (ext='{file_ext}' / visual keywords). Routing to OCR/Vision specialist.",
                recommended_tools=["file.read", "multimodal.extract"],
            )

        # 2. Check for explicit coding / sandbox requests
        if contains_keyword(prompt_lower, self.CODE_KEYWORDS) and "approval note" not in prompt_lower and "sop" not in prompt_lower:
            return RoutingDecision(
                selected_path="coding",
                assigned_model=settings.models.coding,
                starting_task_type="code_execution",
                confidence=0.90,
                rationale="Detected coding / script execution request. Routing to Coding specialist model with sandbox.",
                recommended_tools=["sandbox.run", "file.write"],
            )

        # 3. Check for pure arithmetic / calculation requests
        if contains_keyword(prompt_lower, self.CALC_KEYWORDS) and len(prompt_lower.split()) < 15:
            return RoutingDecision(
                selected_path="calculation",
                assigned_model="deterministic_calculator",
                starting_task_type="calculation",
                confidence=0.92,
                rationale="Detected isolated arithmetic calculation. Routing directly to Deterministic Calculator tool.",
                recommended_tools=["calculator.compute"],
            )

        # 4. Default: General reasoning, planning, and synthesis
        return RoutingDecision(
            selected_path="reasoning",
            assigned_model=settings.models.reasoning,
            starting_task_type="synthesis",
            confidence=0.88,
            rationale="Defaulting to General Reasoning model for structured workflow planning, retrieval orchestration, and synthesis.",
            recommended_tools=["knowledge.search", "calculator.compute", "docx.generate"],
        )
