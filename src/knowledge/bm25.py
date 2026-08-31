"""
Sparse BM25 Search Engine for Sovereign On-Premise Agentic AI Workbench.
Performs exact keyword and term frequency matching over ingested document chunks.
"""
from typing import List, Tuple
import re
from rank_bm25 import BM25Okapi
from src.knowledge.chunker import DocumentChunk


class BM25SearchEngine:
    """Sparse keyword retrieval using BM25Okapi."""

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

        self.tokenized_corpus = [self.tokenize(c.content + " " + c.section_title) for c in chunks]
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
