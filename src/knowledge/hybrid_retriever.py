"""
Hybrid Knowledge Retrieval Engine with Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking.
Fuses BM25 sparse keyword search + Dense Vector search + Cross-Encoder reranking.
Enforces the strict requirement:
"Below a minimum reranker relevance score, a result carries an explicit grounding_status: unmatched,
rendered as a visible caveat rather than presenting a weak match with false confidence."
"""
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import re
import numpy as np
from config.settings import settings, KNOWLEDGE_BASE_DIR
from src.core.state_store import StateStore
from src.knowledge.chunker import DocumentChunk, DocumentChunker
from src.knowledge.bm25 import BM25SearchEngine
from src.knowledge.embeddings import LocalEmbeddingEngine
from src.knowledge.reranker import CrossEncoderReranker


def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Merges multiple ranked lists using standard Reciprocal Rank Fusion (RRF).
    
    RRF(d) = sum_{r in retrievers} 1 / (k + rank_r(d))
    
    Args:
        rankings: List of ranked lists containing chunk_ids (best-first).
        k: Smoothing constant (standard k=60).
        
    Returns:
        List of (chunk_id, fused_score) sorted by fused score descending.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class RetrievalResultItem:
    def __init__(
        self,
        chunk: DocumentChunk,
        score: float,
        evidence_id: str,
        match_type: str,
    ):
        self.chunk = chunk
        self.score = score
        self.evidence_id = evidence_id
        self.match_type = match_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "document_name": self.chunk.document_name,
            "section_title": self.chunk.section_title,
            "page_number": self.chunk.page_number,
            "content": self.chunk.content,
            "relevance_score": round(self.score, 4),
            "match_type": self.match_type,
        }


class HybridKnowledgeRetriever:
    """Ingests internal documents and performs grounded hybrid search with RRF & Reranking."""

    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self.chunker = DocumentChunker()
        self.bm25_engine = BM25SearchEngine()
        self.embedding_engine = LocalEmbeddingEngine()
        self.reranker = CrossEncoderReranker()
        self.chunks: List[DocumentChunk] = []
        self.chunk_embeddings: List[List[float]] = []
        self.min_relevance_threshold = settings.reranker_min_relevance_score

    def ingest_directory(self, dir_path: Optional[str] = None):
        """Ingests all Markdown, text, docx, and PDF documents from knowledge base directory."""
        target_dir = Path(dir_path) if dir_path else KNOWLEDGE_BASE_DIR
        all_chunks: List[DocumentChunk] = []
        
        for file_path in target_dir.glob("*.*"):
            ext = file_path.suffix.lower()
            if ext in (".md", ".txt"):
                content = file_path.read_text(encoding="utf-8", errors="replace")
                chunks = self.chunker.chunk_markdown_document(file_path.name, content)
                all_chunks.extend(chunks)
            elif ext == ".pdf":
                try:
                    chunks = self.chunker.chunk_pdf_document(file_path.name, str(file_path))
                    all_chunks.extend(chunks)
                except Exception:
                    pass
            elif ext == ".docx":
                try:
                    import docx
                    d = docx.Document(str(file_path))
                    full_text = "\n".join([p.text for p in d.paragraphs if p.text.strip()])
                    chunks = self.chunker.chunk_markdown_document(file_path.name, full_text)
                    all_chunks.extend(chunks)
                except Exception:
                    pass

        self.chunks = all_chunks
        self.bm25_engine.index_chunks(self.chunks)
        self.chunk_embeddings = [self.embedding_engine.get_embedding(c.content) for c in self.chunks]

    def search(
        self,
        query: str,
        project_id: str,
        top_k: int = settings.top_k_retrieval,
    ) -> Dict[str, Any]:
        """
        Executes:
        1. BM25 Sparse Search (top-50)
        2. Dense Vector Search (top-50)
        3. Reciprocal Rank Fusion (RRF k=60)
        4. Cross-Encoder Reranking (top-20 -> top-k)
        5. Strict Grounding Threshold Gate
        """
        if not self.chunks:
            return {
                "grounding_status": "unmatched",
                "caveat": "Knowledge base is empty. Ingestion required.",
                "results": [],
            }

        candidate_pool_size = max(20, top_k * 4)

        # 1. Sparse BM25 Search (top candidates)
        bm25_results = self.bm25_engine.search(query, top_k=candidate_pool_size)
        bm25_ranked_ids = [res[0].chunk_id for res in bm25_results]

        # 2. Dense Vector Search (top candidates)
        query_vec = self.embedding_engine.get_embedding(query)
        vec_scored_all = []
        for i, chunk in enumerate(self.chunks):
            sim = self.embedding_engine.cosine_similarity(query_vec, self.chunk_embeddings[i])
            vec_scored_all.append((chunk, sim))
        vec_scored_all.sort(key=lambda x: x[1], reverse=True)
        vec_ranked_ids = [res[0].chunk_id for res in vec_scored_all[:candidate_pool_size]]

        # 3. Reciprocal Rank Fusion (RRF)
        fused_rankings = reciprocal_rank_fusion([bm25_ranked_ids, vec_ranked_ids], k=60)
        
        # Build candidate tuples for reranker
        chunk_map = {c.chunk_id: c for c in self.chunks}
        fused_candidates: List[Tuple[DocumentChunk, float, str]] = []
        
        # Max theoretical RRF score for 2 retrievers at rank 1 is 2 * (1/61) ~ 0.0328
        max_rrf_score = 2.0 / 61.0

        for cid, rrf_score in fused_rankings[:candidate_pool_size]:
            if cid in chunk_map:
                norm_score = min(1.0, rrf_score / max_rrf_score)
                fused_candidates.append((chunk_map[cid], norm_score, "hybrid_rrf"))

        # 4. Cross-Encoder Joint Reranking
        reranked_candidates = self.reranker.rerank(query, fused_candidates, top_k=top_k)

        # 5. Strict Grounding Threshold Gate
        if not reranked_candidates or reranked_candidates[0][1] < self.min_relevance_threshold:
            return {
                "grounding_status": "unmatched",
                "top_score": round(reranked_candidates[0][1], 4) if reranked_candidates else 0.0,
                "caveat": f"No internal SOP or standard met the minimum confidence threshold ({self.min_relevance_threshold:.2f}). Claims must be flagged as unmatched.",
                "results": [],
            }

        # 6. Populate Evidence Table in Persistent State Store
        results_payload: List[Dict[str, Any]] = []
        for rank_idx, (chunk, final_score, match_type) in enumerate(reranked_candidates, start=1):
            evidence_id = f"E_RET_{chunk.chunk_id}"
            
            # Atomic state store registration
            self.state_store.add_evidence(
                evidence_id=evidence_id,
                project_id=project_id,
                source_type="hybrid_rag_reranked",
                source_document=chunk.document_name,
                page_number=chunk.page_number,
                section=chunk.section_title,
                extracted_text=chunk.content,
                structured_data={
                    "relevance_score": final_score,
                    "match_type": match_type,
                    "rank": rank_idx,
                },
                confidence=final_score,
            )

            results_payload.append(
                RetrievalResultItem(
                    chunk=chunk,
                    score=final_score,
                    evidence_id=evidence_id,
                    match_type=match_type,
                ).to_dict()
            )

        return {
            "grounding_status": "matched",
            "top_score": round(reranked_candidates[0][1], 4),
            "total_candidates": len(self.chunks),
            "results": results_payload,
        }
