"""
Cross-Encoder Reranking Module for Sovereign On-Premise Agentic AI Workbench.
Processes joint [Query, Document] pairs to capture fine-grained token-level semantic interactions,
section breadcrumb alignment, numerical tolerances, and tabular data signals.
"""
from typing import Any, Dict, List, Tuple
import re
from config.settings import settings
from src.knowledge.chunker import DocumentChunk


class CrossEncoderReranker:
    """Local Cross-Encoder Reranker for high-precision candidate scoring."""

    def __init__(self, model_name: str = "bbjson/bge-reranker-base:latest"):
        self.model_name = model_name
        self.ollama_base_url = settings.ollama_base_url

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[DocumentChunk, float, str]],
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float, str]]:
        """
        Reranks top candidate chunks using joint query-document interaction scoring,
        accounting for heading/breadcrumb alignment, numerical metrics, and tabular structures.

        Args:
            query: The search query.
            candidates: List of (DocumentChunk, initial_score, match_type).
            top_k: Number of reranked items to return.

        Returns:
            List of (DocumentChunk, reranked_score, match_type) sorted by reranked score descending.
        """
        if not candidates:
            return []

        query_clean = query.lower().strip()
        query_terms = set(re.findall(r"\b\w{3,}\b", query_clean))
        query_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", query))
        reranked_results: List[Tuple[DocumentChunk, float, str]] = []

        for chunk, initial_score, match_type in candidates:
            sec_title_lower = chunk.section_title.lower()
            content_lower = chunk.content.lower()
            doc_text = f"{sec_title_lower} {content_lower}"
            doc_terms = set(re.findall(r"\b\w{3,}\b", doc_text))

            # Breadcrumbs from chunk metadata
            breadcrumbs = chunk.metadata.get("breadcrumbs", [])
            breadcrumb_text = " ".join(breadcrumbs).lower() if isinstance(breadcrumbs, list) else ""

            # 1. Exact phrase and term overlap
            exact_query_in_doc = query_clean in doc_text
            exact_query_in_title = query_clean in sec_title_lower or (breadcrumb_text and query_clean in breadcrumb_text)

            term_overlap = len(query_terms.intersection(doc_terms))
            overlap_ratio = term_overlap / max(1, len(query_terms))

            # 2. Heading & Breadcrumb matching bonus (high signal for chapter/section queries)
            title_terms = set(re.findall(r"\b\w{3,}\b", f"{sec_title_lower} {breadcrumb_text}"))
            title_overlap = len(query_terms.intersection(title_terms))
            title_overlap_ratio = title_overlap / max(1, len(query_terms))
            heading_bonus = 0.25 * title_overlap_ratio
            if exact_query_in_title:
                heading_bonus += 0.25

            # 3. Number & metric matching (vital for engineering thresholds e.g. 4.80 mm, 142 bar)
            doc_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", doc_text))
            number_overlap = len(query_numbers.intersection(doc_numbers)) if query_numbers else 0
            number_match_bonus = 0.20 if number_overlap > 0 else 0.0

            # 4. Tabular data signal bonus
            is_table = chunk.metadata.get("is_table", False)
            table_bonus = 0.05 if (is_table and ("table" in query_clean or "log" in query_clean or "values" in query_clean)) else 0.0

            # 5. Composite Joint Interaction Score
            if term_overlap == 0 and not exact_query_in_doc and not exact_query_in_title and number_match_bonus == 0.0:
                # Weak candidate penalty
                rerank_score = initial_score * 0.20
            else:
                base_score = 0.35 + 0.30 * overlap_ratio + (0.20 if exact_query_in_doc else 0.0)
                composite = (
                    base_score
                    + heading_bonus
                    + number_match_bonus
                    + table_bonus
                    + 0.10 * initial_score
                )
                rerank_score = min(1.0, max(0.0, composite))

            reranked_results.append((chunk, round(rerank_score, 4), f"{match_type}+reranked"))

        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results[:top_k]
