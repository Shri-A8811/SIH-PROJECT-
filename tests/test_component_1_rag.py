"""
Verification Test Suite for Component 1: Document Intelligence & Structure-Aware RAG Engine.
Validates:
1. PDF outline/TOC hierarchy and breadcrumb extraction.
2. Markdown and tabular layout preservation with repeated headers.
3. Local BGE neural embeddings, batched generation, and deterministic fallback.
4. Breadcrumb-boosted hybrid retrieval and cross-encoder precision scoring.
5. Category/folder-scoped isolation and zero cross-contamination.
"""
import os
import pytest
from pathlib import Path
from src.core.state_store import StateStore
from src.knowledge.chunker import DocumentChunk, DocumentChunker
from src.knowledge.embeddings import LocalEmbeddingEngine
from src.knowledge.bm25 import BM25SearchEngine
from src.knowledge.reranker import CrossEncoderReranker
from src.knowledge.hybrid_retriever import HybridKnowledgeRetriever


def test_markdown_chunking_and_table_header_preservation():
    """Validates that markdown tables are preserved as intact tables and headers replicated if split."""
    table_content = """# Turnaround Equipment Inspection

## Section 2.1: Thickness Gauging Log

Here are the ultrasonic thickness measurement results:

| Equipment Tag | Nominal (mm) | Measured (mm) | Design Min (mm) | Status |
|---|---|---|---|---|
| P-104B-B1 | 8.00 | 7.20 | 4.80 | PASS |
| P-104B-B2 | 8.00 | 6.50 | 4.80 | PASS |
| P-104B-B3 | 8.00 | 3.42 | 4.80 | CRITICAL_FAIL |
| P-104B-B4 | 8.00 | 5.10 | 4.80 | PASS |
"""
    chunks = DocumentChunker.chunk_markdown_document("inspection_log.md", table_content, category="Turnaround")
    assert len(chunks) >= 1

    # Find the table chunk
    table_chunk = next((c for c in chunks if c.metadata.get("is_table")), None)
    assert table_chunk is not None, "Table block should be identified with is_table=True"
    assert "Turnaround Equipment Inspection" in table_chunk.metadata.get("breadcrumbs", [])
    assert "Section 2.1: Thickness Gauging Log" in table_chunk.metadata.get("breadcrumbs", [])
    assert "| P-104B-B3 | 8.00 | 3.42 | 4.80 | CRITICAL_FAIL |" in table_chunk.content
    assert "| Equipment Tag | Nominal (mm) |" in table_chunk.content


def test_table_split_replicates_header_rows():
    """Validates that when a large table exceeds max chunk size, column headers are preserved on all parts."""
    header = "| Col A | Col B | Col C |\n|---|---|---|\n"
    rows = [f"| Data {i:03d} | Metric {i*2.5:.2f} | Description {i} with extended technical notes |" for i in range(40)]
    large_table = header + "\n".join(rows)

    chunks = DocumentChunker._split_markdown_table(
        doc_name="large_table.md",
        category="General",
        section_title="Data Table",
        page=1,
        table_text=large_table,
        start_idx=1,
        meta={"breadcrumbs": ["Data Table"]},
    )

    assert len(chunks) > 1, "Large table should be partitioned into multiple chunks"
    for idx, c in enumerate(chunks):
        assert "| Col A | Col B | Col C |" in c.content, f"Chunk {idx+1} must contain the replicated table header"
        assert c.metadata.get("is_table") is True


def test_embedding_engine_batch_and_cache():
    """Validates local embedding engine batching, L2 normalization, and caching."""
    eng = LocalEmbeddingEngine()
    texts = [
        "Crude distillation unit pipe wall thinning",
        "Hydrocracker high-pressure flange RTJ gasket",
        "Diesel hydrotreating unit eddy current inspection",
    ]

    batch_vecs = eng.get_embeddings_batch(texts)
    assert len(batch_vecs) == 3

    for v in batch_vecs:
        assert len(v) == 384
        # Verify L2 norm is 1.0 (within float precision)
        norm = sum(x**2 for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-4

    # Verify cosine similarity is highest for identical text
    sim_self = eng.cosine_similarity(batch_vecs[0], batch_vecs[0])
    sim_other = eng.cosine_similarity(batch_vecs[0], batch_vecs[1])
    assert abs(sim_self - 1.0) < 1e-4
    assert sim_other < 0.99

    # Verify in-memory cache
    cached_vec = eng.get_embedding(texts[0])
    assert cached_vec == batch_vecs[0]


def test_breadcrumb_and_numerical_reranker_boost():
    """Validates that CrossEncoderReranker elevates exact headings and numerical tolerances."""
    reranker = CrossEncoderReranker()
    query = "Find wall thickness measurement 3.42 mm in Chapter 7"

    c1 = DocumentChunk(
        chunk_id="c1",
        document_name="doc1.pdf",
        category="General",
        section_title="General Maintenance Overview",
        page_number=2,
        content="Routine refinery maintenance was scheduled for the upcoming quarter without issue.",
        metadata={"breadcrumbs": ["Maintenance"]},
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        document_name="doc2.pdf",
        category="General",
        section_title="Chapter 7: Ultrasonic Inspection",
        page_number=14,
        content="Bend B-3 showed critical wall thinning with residual thickness measured at 3.42 mm.",
        metadata={"breadcrumbs": ["Chapter 7: Ultrasonic Inspection", "NDT Findings"]},
    )

    candidates = [(c1, 0.40, "bm25"), (c2, 0.40, "vector")]
    reranked = reranker.rerank(query, candidates, top_k=2)

    assert len(reranked) == 2
    top_chunk, top_score, _ = reranked[0]
    assert top_chunk.chunk_id == "c2", "Chunk matching Chapter 7 and 3.42 mm must rank at #1"
    assert top_score > reranked[1][1], "Top chunk score must be higher than distractor chunk"
    assert top_score >= 0.70, "Score should reflect strong heading and numerical alignment"


def test_category_scoped_retrieval_isolation(tmp_path):
    """Validates that category filtering strictly isolates knowledge chunks across folders."""
    store = StateStore()
    retriever = HybridKnowledgeRetriever(store)

    doc_sop = tmp_path / "sop_pipe.md"
    doc_sop.write_text("# SOP-17\n\nPipe thinning inspection requires ultrasonic A-scan.", encoding="utf-8")

    doc_report = tmp_path / "report_turnaround.md"
    doc_report.write_text("# Turnaround Report\n\nPipe thinning was detected on bend B3.", encoding="utf-8")

    retriever.ingest_file(doc_sop, category="SOPs")
    retriever.ingest_file(doc_report, category="Turnaround")

    proj_id = "test_cat_isolation"
    store.create_project(proj_id, "Isolation Test", "Test")

    # Search scoped to SOPs only
    sop_res = retriever.search("Pipe thinning inspection", project_id=proj_id, category="SOPs")
    assert sop_res["grounding_status"] == "matched"
    for r in sop_res["results"]:
        assert r["category"] == "SOPs"
        assert r["document_name"] == "sop_pipe.md"

    # Search scoped to Turnaround only
    turnaround_res = retriever.search("Pipe thinning inspection", project_id=proj_id, category="Turnaround")
    assert turnaround_res["grounding_status"] == "matched"
    for r in turnaround_res["results"]:
        assert r["category"] == "Turnaround"
        assert r["document_name"] == "report_turnaround.md"

    # Search scoped to non-existent category
    empty_res = retriever.search("Pipe thinning", project_id=proj_id, category="NonExistent")
    assert empty_res["grounding_status"] == "unmatched"
    assert "No indexed knowledge chunks found" in empty_res["caveat"]


def test_cisco_report_pdf_outline_extraction():
    """Validates PDF outline extraction on real uploaded artifact if available."""
    pdf_path = Path(r"C:\Users\Shridhar\.gemini\antigravity-ide\brain\bd208ebc-d047-4f53-9408-5cce7f632f76\.user_uploaded\media_1788207425890.pdf")
    if not pdf_path.exists():
        pytest.skip("Sample PDF artifact not available in test environment.")

    chunks = DocumentChunker.chunk_pdf_document("Cisco_Report.pdf", str(pdf_path), category="Internship")
    assert len(chunks) > 50

    # Verify that outline breadcrumbs were populated for Chapter 7 / Project section
    project_chunks = [c for c in chunks if "Customer Review Analytics Project" in c.section_title]
    assert len(project_chunks) >= 1, "Must find chunks with Customer Review Analytics Project outline hierarchy"
    sample_proj_chunk = project_chunks[0]
    assert sample_proj_chunk.page_number in (31, 32, 33, 34, 35)
    assert "Customer Review Analytics Project" in sample_proj_chunk.metadata.get("breadcrumbs", [])
