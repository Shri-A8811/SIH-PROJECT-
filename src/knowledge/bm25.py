"""
Sparse BM25 Search Engine for Sovereign On-Premise Agentic AI Workbench.
Performs exact keyword and term frequency matching over ingested document chunks,
incorporating headings, hierarchical breadcrumbs, and document names into the corpus.
"""
from typing import List, Tuple
import re
from rank_bm25 import BM25Okapi
from src.knowledge.chunker import DocumentChunk


class BM25SearchEngine:
    """Sparse keyword retrieval using BM25Okapi with structure awareness."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.bm25: BM25Okapi = None
        self.tokenized_corpus: List[List[str]] = []

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def index_chunks(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            self.tokenized_corpus = []
            return

        corpus = []
        for c in chunks:
            breadcrumbs = c.metadata.get("breadcrumbs", [])
            b_text = " ".join(breadcrumbs) if isinstance(breadcrumbs, list) else ""
            composite_text = f"{c.document_name} {c.section_title} {b_text} {c.content}"
            corpus.append(self.tokenize(composite_text))

        self.tokenized_corpus = corpus
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        scored_chunks = [(self.chunks[i], float(scores[i])) for i in range(len(self.chunks))]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
