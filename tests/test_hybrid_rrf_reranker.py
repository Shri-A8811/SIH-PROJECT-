"""
Tests for Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranker.
"""
from src.knowledge.chunker import DocumentChunk
from src.knowledge.reranker import CrossEncoderReranker
from src.knowledge.hybrid_retriever import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_logic():
    """Verifies that RRF combines multi-retriever rankings predictably with k=60."""
    rankings = [
        ["chunk_A", "chunk_B", "chunk_C"],  # BM25 ranking
        ["chunk_B", "chunk_A", "chunk_D"],  # Dense ranking
    ]
    fused = reciprocal_rank_fusion(rankings, k=60)
    
    # Both chunk_A and chunk_B appear in both rankings at ranks (1,2) and (2,1)
    # Their scores should be equal and higher than single-retriever appearances
    fused_dict = dict(fused)
    assert fused_dict["chunk_A"] > fused_dict["chunk_C"]
    assert fused_dict["chunk_B"] > fused_dict["chunk_D"]
    assert abs(fused_dict["chunk_A"] - fused_dict["chunk_B"]) < 1e-6


def test_cross_encoder_reranker_scoring():
    """Verifies that the CrossEncoderReranker rewards exact keyword and number matches."""
    reranker = CrossEncoderReranker()
    chunk1 = DocumentChunk(
        chunk_id="chk1",
        document_name="SOP17.md",
        section_title="Retirement Limits",
        content="Mandatory retirement thickness is 4.80 mm for primary crude lines.",
    )
    chunk2 = DocumentChunk(
        chunk_id="chk2",
        document_name="SOP04.md",
        section_title="General Piping",
        content="Piping systems must be inspected regularly every 24 months.",
    )

    candidates = [(chunk1, 0.5, "hybrid"), (chunk2, 0.5, "hybrid")]
    reranked = reranker.rerank("minimum retirement thickness 4.80 mm", candidates, top_k=2)

    assert len(reranked) == 2
    # chunk1 contains 'retirement', 'thickness', and the exact number '4.80'
    assert reranked[0][0].chunk_id == "chk1"
    assert reranked[0][1] > reranked[1][1]
