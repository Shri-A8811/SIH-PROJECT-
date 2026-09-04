"""
Hybrid Knowledge Retrieval Engine with Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking.
Fuses BM25 sparse keyword search + Dense Vector search via PostgreSQL pgvector + Cross-Encoder reranking.
Enforces the strict requirement:
"Below a minimum reranker relevance score, a result carries an explicit grounding_status: unmatched,
rendered as a visible caveat rather than presenting a weak match with false confidence."
Includes real-time incremental document ingestion and category/folder-scoped retrieval.
"""
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import re
import logging
import numpy as np
from config.settings import settings, KNOWLEDGE_BASE_DIR
from src.core.state_store import StateStore, KnowledgeChunkRecord, DocumentRecord
from src.knowledge.chunker import DocumentChunk, DocumentChunker
from src.knowledge.bm25 import BM25SearchEngine
from src.knowledge.embeddings import LocalEmbeddingEngine
from src.knowledge.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


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
            "category": getattr(self.chunk, "category", "General"),
            "section_title": self.chunk.section_title,
            "page_number": self.chunk.page_number,
            "breadcrumbs": self.chunk.metadata.get("breadcrumbs", []),
            "is_table": self.chunk.metadata.get("is_table", False),
            "content": self.chunk.content,
            "relevance_score": round(self.score, 4),
            "match_type": self.match_type,
        }


class HybridKnowledgeRetriever:
    """Ingests internal documents and performs grounded hybrid search with pgvector RRF & Reranking."""

    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self.chunker = DocumentChunker()
        self.bm25_engine = BM25SearchEngine()
        self.embedding_engine = LocalEmbeddingEngine()
        self.reranker = CrossEncoderReranker()
        self.chunks: List[DocumentChunk] = []
        self.chunk_embeddings: List[List[float]] = []
        self.min_relevance_threshold = settings.reranker_min_relevance_score

    def ingest_file(self, file_path: Path, category: str = "General") -> int:
        """
        Ingests a single document into BM25, pgvector, and Document Inventory in real-time.
        Returns the number of chunks generated and indexed.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return 0

        doc_name = file_path.name
        ext = file_path.suffix.lower()
        new_chunks: List[DocumentChunk] = []

        if ext in (".md", ".txt"):
            content = file_path.read_text(encoding="utf-8", errors="replace")
            new_chunks = self.chunker.chunk_markdown_document(doc_name, content, category=category)
        elif ext == ".pdf":
            try:
                new_chunks = self.chunker.chunk_pdf_document(doc_name, str(file_path), category=category)
            except Exception as e:
                logger.warning(f"Failed to chunk PDF {doc_name}: {e}")
        elif ext == ".docx":
            try:
                import docx
                d = docx.Document(str(file_path))
                full_text = "\n".join([p.text for p in d.paragraphs if p.text.strip()])
                new_chunks = self.chunker.chunk_markdown_document(doc_name, full_text, category=category)
            except Exception as e:
                logger.warning(f"Failed to chunk DOCX {doc_name}: {e}")

        if not new_chunks:
            return 0

        # Remove previous chunks of this document from in-memory pool if re-indexing
        kept_chunks = []
        kept_embeddings = []
        for c, emb in zip(self.chunks, self.chunk_embeddings):
            if c.document_name != doc_name:
                kept_chunks.append(c)
                kept_embeddings.append(emb)

        # Compute embeddings for new chunks via fast batch endpoint
        new_embeddings = self.embedding_engine.get_embeddings_batch([c.content for c in new_chunks])

        # Upsert into state store (pgvector / SQLite)
        for chunk, vec in zip(new_chunks, new_embeddings):
            try:
                self.state_store.upsert_knowledge_chunk(
                    chunk_id=chunk.chunk_id,
                    document_name=chunk.document_name,
                    content=chunk.content,
                    section_title=chunk.section_title,
                    page_number=chunk.page_number,
                    embedding=vec,
                    category=category,
                )
            except Exception as e:
                logger.debug(f"Chunk upsert note: {e}")

        # Register in Document Inventory table
        file_size = file_path.stat().st_size if file_path.exists() else 0
        self.state_store.upsert_document(
            filename=doc_name,
            category=category,
            file_path=str(file_path),
            file_size_bytes=file_size,
            chunk_count=len(new_chunks),
        )

        # Update in-memory state and re-index BM25
        self.chunks = kept_chunks + new_chunks
        self.chunk_embeddings = kept_embeddings + new_embeddings
        self.bm25_engine.index_chunks(self.chunks)

        return len(new_chunks)

    def load_from_state_store(self) -> int:
        """Loads pre-indexed knowledge chunks from persistent state store into memory."""
        records = self.state_store.get_all_knowledge_chunks()
        if not records:
            return 0
        loaded_chunks = []
        loaded_embeddings = []
        for r in records:
            chunk = DocumentChunk(
                chunk_id=r.chunk_id,
                document_name=r.document_name,
                category=r.category,
                section_title=r.section_title or "Overview",
                page_number=r.page_number or 1,
                content=r.content,
                metadata={"breadcrumbs": [r.document_name, r.section_title or "Overview"]},
            )
            loaded_chunks.append(chunk)
            loaded_embeddings.append(r.embedding or self.embedding_engine.get_embedding(r.content))
        self.chunks = loaded_chunks
        self.chunk_embeddings = loaded_embeddings
        self.bm25_engine.index_chunks(self.chunks)
        return len(loaded_chunks)

    def ingest_directory(self, dir_path: Optional[str] = None):
        """
        Ingests all documents from knowledge base directory and subdirectories as categories.
        Loads existing indexed chunks from DB instantaneously and only indexes new files.
        """
        if not self.chunks:
            self.load_from_state_store()

        target_dir = Path(dir_path) if dir_path else KNOWLEDGE_BASE_DIR
        if not target_dir.exists():
            return

        indexed_docs = {c.document_name for c in self.chunks}

        # Walk all files in root and subdirectories
        for file_path in target_dir.rglob("*.*"):
            if file_path.is_file() and file_path.suffix.lower() in (".md", ".txt", ".pdf", ".docx"):
                if file_path.name in indexed_docs:
                    continue
                try:
                    rel_parent = file_path.parent.relative_to(target_dir)
                    category = rel_parent.parts[0] if rel_parent.parts else "General"
                except Exception:
                    category = "General"
                self.ingest_file(file_path, category=category)

    def search(
        self,
        query: str,
        project_id: str,
        top_k: int = settings.top_k_retrieval,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes:
        1. BM25 Sparse Search (top-50, optionally filtered by category)
        2. Dense Vector Search via pgvector / numpy (top-50, optionally filtered by category)
        3. Reciprocal Rank Fusion (RRF k=60)
        4. Cross-Encoder Reranking (top-20 -> top-k)
        5. Strict Grounding Threshold Gate
        """
        filter_cat = category if category and category not in ("All Documents", "All", "") else None

        # If a category is requested, verify if there are any chunks in that category
        if filter_cat:
            active_chunks = [c for c in self.chunks if getattr(c, "category", "General") == filter_cat]
            if not active_chunks:
                return {
                    "grounding_status": "unmatched",
                    "top_score": 0.0,
                    "caveat": f"No indexed knowledge chunks found for folder/category: '{filter_cat}'.",
                    "results": [],
                }
        else:
            active_chunks = self.chunks

        if not active_chunks:
            return {
                "grounding_status": "unmatched",
                "caveat": "Knowledge base is empty. Ingestion required.",
                "results": [],
            }

        candidate_pool_size = max(20, top_k * 4)

        # 1. Sparse BM25 Search (top candidates filtered by category)
        bm25_results = self.bm25_engine.search(query, top_k=candidate_pool_size * 2)
        bm25_ranked_ids = []
        for res in bm25_results:
            c = res[0]
            if not filter_cat or getattr(c, "category", "General") == filter_cat:
                bm25_ranked_ids.append(c.chunk_id)
                if len(bm25_ranked_ids) >= candidate_pool_size:
                    break

        # 2. Dense Vector Search via PostgreSQL pgvector / state store
        query_vec = self.embedding_engine.get_embedding(query)
        pgvector_results = self.state_store.search_vector_chunks(
            query_vec, top_k=candidate_pool_size, category=filter_cat
        )
        
        if pgvector_results:
            vec_ranked_ids = [rec.chunk_id for rec, _score in pgvector_results]
        else:
            # In-memory vector calculation fallback
            vec_scored_all = []
            for i, chunk in enumerate(self.chunks):
                if not filter_cat or getattr(chunk, "category", "General") == filter_cat:
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
        raw_reranked = self.reranker.rerank(query, fused_candidates, top_k=max(top_k * 3, 15))

        # Deduplicate candidates by (document_name, page_number, section_title)
        seen_keys = set()
        deduped_candidates = []
        for chunk, score, match_type in raw_reranked:
            key = (chunk.document_name, chunk.page_number, chunk.section_title)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_candidates.append((chunk, score, match_type))

        # Prune cross-domain candidates that score far below the top candidate
        if deduped_candidates:
            top_score = deduped_candidates[0][1]
            # Discard candidates that are below minimum threshold or significantly weaker than top hit
            relative_min = max(self.min_relevance_threshold, top_score * 0.60)
            reranked_candidates = [c for c in deduped_candidates if c[1] >= relative_min][:top_k]
        else:
            reranked_candidates = []

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
            
            # Atomic state store registration with embedding vector
            chunk_idx = next((i for i, c in enumerate(self.chunks) if c.chunk_id == chunk.chunk_id), -1)
            chunk_vec = self.chunk_embeddings[chunk_idx] if chunk_idx >= 0 and chunk_idx < len(self.chunk_embeddings) else None

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
                    "category": getattr(chunk, "category", "General"),
                    "breadcrumbs": chunk.metadata.get("breadcrumbs", []),
                    "is_table": chunk.metadata.get("is_table", False),
                },
                confidence=final_score,
                embedding=chunk_vec,
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
            "total_candidates": len(active_chunks),
            "results": results_payload,
            "category_scope": filter_cat or "All Documents",
        }

    def delete_document(self, filename: str) -> bool:
        """Deletes document from state store, removes from in-memory index, and unlinks file if local."""
        self.state_store.delete_document(filename)
        
        # Remove from memory
        kept_chunks = []
        kept_embeddings = []
        for c, emb in zip(self.chunks, self.chunk_embeddings):
            if c.document_name != filename:
                kept_chunks.append(c)
                kept_embeddings.append(emb)
        self.chunks = kept_chunks
        self.chunk_embeddings = kept_embeddings
        self.bm25_engine.index_chunks(self.chunks)

        # Remove physical file if in KNOWLEDGE_BASE_DIR
        for p in KNOWLEDGE_BASE_DIR.rglob(filename):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass
        return True

    def list_documents(self, category: Optional[str] = None) -> List[DocumentRecord]:
        return self.state_store.list_documents(category=category)

    def get_categories(self) -> List[str]:
        return self.state_store.get_categories()
