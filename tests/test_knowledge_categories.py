"""
Tests for Document Categorization, Folder Scoping, and Multi-Chat Persistence.
"""
from pathlib import Path
import tempfile
import pytest
from src.core.state_store import StateStore
from src.knowledge.hybrid_retriever import HybridKnowledgeRetriever
from src.knowledge.chunker import DocumentChunk, DocumentChunker


def test_document_chunk_category_field():
    """Verifies that DocumentChunk correctly stores and preserves category metadata."""
    chunk = DocumentChunk(
        chunk_id="test_chk_001",
        document_name="MRPL_SOP_017.md",
        category="SOPs",
        section_title="Retirement Limits",
        content="Retirement wall thickness is 4.80 mm.",
    )
    assert chunk.category == "SOPs"
    assert chunk.chunk_id == "test_chk_001"


def test_state_store_document_inventory_crud():
    """Verifies that DocumentRecord CRUD operations function reliably."""
    store = StateStore("sqlite:///:memory:")

    # 1. Upsert document
    doc = store.upsert_document(
        filename="MRPL_SOP_017.md",
        category="SOPs",
        file_path="/data/knowledge_base/SOPs/MRPL_SOP_017.md",
        file_size_bytes=1024,
        chunk_count=5,
    )
    assert doc.filename == "MRPL_SOP_017.md"
    assert doc.category == "SOPs"
    assert doc.chunk_count == 5

    # 2. List documents
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "MRPL_SOP_017.md"

    # 3. Categories list
    cats = store.get_categories()
    assert "SOPs" in cats
    assert "General" in cats

    # 4. Filter by category
    filtered = store.list_documents(category="SOPs")
    assert len(filtered) == 1
    empty_filter = store.list_documents(category="NonExistent")
    assert len(empty_filter) == 0

    # 5. Delete document
    assert store.delete_document("MRPL_SOP_017.md") is True
    assert len(store.list_documents()) == 0


def test_state_store_multi_chat_session_management():
    """Verifies multi-chat session creation, message storage, and thread isolation."""
    store = StateStore("sqlite:///:memory:")

    # 1. Create two independent chat sessions
    sess1 = store.create_chat_session(title="Chat Session 1", knowledge_scope="SOPs")
    sess2 = store.create_chat_session(title="Chat Session 2", knowledge_scope="Inspection Reports")
    assert sess1.id != sess2.id
    assert sess1.knowledge_scope == "SOPs"
    assert sess2.knowledge_scope == "Inspection Reports"

    # 2. Add messages to Session 1
    store.save_chat_message(sess1.id, "user", "What are the retirement limits for CDU-1?")
    store.save_chat_message(sess1.id, "assistant", "Mandatory retirement thickness is 4.80 mm.")

    # 3. Add messages to Session 2
    store.save_chat_message(sess2.id, "user", "Show me flange inspection findings.")

    # 4. Verify message separation between sessions
    msgs1 = store.get_chat_messages(sess1.id)
    msgs2 = store.get_chat_messages(sess2.id)
    assert len(msgs1) == 2
    assert len(msgs2) == 1
    assert msgs1[0].role == "user"
    assert msgs1[1].role == "assistant"
    assert msgs2[0].content == "Show me flange inspection findings."

    # 5. Verify sessions list
    all_sess = store.get_chat_sessions()
    assert len(all_sess) >= 2

    # 6. Delete Session 1 and verify cascade deletion of its messages
    store.delete_chat_session(sess1.id)
    assert store.get_chat_session(sess1.id) is None
    assert len(store.get_chat_messages(sess1.id)) == 0
    # Session 2 should remain untouched
    assert store.get_chat_session(sess2.id) is not None
    assert len(store.get_chat_messages(sess2.id)) == 1


def test_hybrid_retriever_category_scoped_search():
    """Verifies that HybridKnowledgeRetriever correctly filters search results by category."""
    store = StateStore("sqlite:///:memory:")
    retriever = HybridKnowledgeRetriever(store)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create file 1 in category 'SOPs'
        sop_file = tmp_path / "sop17.md"
        sop_file.write_text("# MRPL SOP 17\nMandatory pipe wall retirement limit is 4.80 mm.", encoding="utf-8")
        retriever.ingest_file(sop_file, category="SOPs")

        # Create file 2 in category 'Inspection_Reports'
        insp_file = tmp_path / "insp_report.md"
        insp_file.write_text("# Inspection 2026\nCDU-1 measured wall thickness is 3.42 mm.", encoding="utf-8")
        retriever.ingest_file(insp_file, category="Inspection_Reports")

        assert len(retriever.chunks) >= 2
        categories = retriever.get_categories()
        assert "SOPs" in categories
        assert "Inspection_Reports" in categories

        # Query scoped to 'SOPs'
        res_sop = retriever.search("pipe wall thickness", "PROJ_TEST", top_k=5, category="SOPs")
        assert res_sop["grounding_status"] == "matched"
        assert all(r.get("category") == "SOPs" for r in res_sop["results"])

        # Query scoped to 'Inspection_Reports'
        res_insp = retriever.search("wall thickness", "PROJ_TEST", top_k=5, category="Inspection_Reports")
        assert res_insp["grounding_status"] == "matched"
        assert all(r.get("category") == "Inspection_Reports" for r in res_insp["results"])

        # Query scoped to non-existent folder
        res_empty = retriever.search("wall thickness", "PROJ_TEST", top_k=5, category="NonExistentFolder")
        assert res_empty["grounding_status"] == "unmatched"
        assert len(res_empty["results"]) == 0
