"""
Tests for Hybrid Knowledge Retrieval & Grounding Status Threshold Gate.
"""
from src.core.state_store import StateStore
from src.knowledge.chunker import DocumentChunker
from src.knowledge.hybrid_retriever import HybridKnowledgeRetriever


def test_structure_aware_chunker():
    markdown_content = """# MRPL SOP-17
## Section 4.0 Minimum Retirement Limits
Standard nominal thickness is 8.00 mm.
Under no circumstances shall residual thickness be below 4.80 mm.
"""
    chunks = DocumentChunker.chunk_markdown_document("MRPL_SOP_17.md", markdown_content)
    assert len(chunks) >= 1
    assert "4.80 mm" in chunks[0].content


def test_hybrid_retriever_matched_and_unmatched_threshold():
    store = StateStore("sqlite:///:memory:")
    retriever = HybridKnowledgeRetriever(store)

    # Ingest mock documents (multi-document corpus for valid BM25 IDF)
    doc1 = """# MRPL SOP-17
## Section 4.2 Retirement Criteria
Crude distillation transfer lines must not operate below 4.80 mm.
Spools below this limit must be replaced immediately.
"""
    doc2 = """# MRPL SOP-04
## Section 3.0 Flange Joint Standards
Class 1500 RTJ flanges must undergo 142 bar hydro-testing.
Zero allowable surface fissuring on gasket ring joints.
"""
    chunks = DocumentChunker.chunk_markdown_document("MRPL_SOP_17.md", doc1)
    chunks.extend(DocumentChunker.chunk_markdown_document("MRPL_SOP_04.md", doc2))

    retriever.chunks = chunks
    retriever.bm25_engine.index_chunks(chunks)
    retriever.chunk_embeddings = [retriever.embedding_engine.get_embedding(c.content) for c in chunks]

    # 1. Matched Query
    matched_res = retriever.search("minimum retirement thickness crude transfer line", project_id="P_RAG")
    assert matched_res["grounding_status"] == "matched"
    assert len(matched_res["results"]) > 0
    assert "E_RET_" in matched_res["results"][0]["evidence_id"]

    # 2. Out-of-Domain Unmatched Query (must trigger grounding_status: unmatched caveat)
    unmatched_res = retriever.search("arbitrary quantum gravity string theory recipe", project_id="P_RAG")
    assert unmatched_res["grounding_status"] == "unmatched"
    assert "caveat" in unmatched_res
