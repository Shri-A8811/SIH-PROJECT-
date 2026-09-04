"""
Structure-Aware Document Chunker for Sovereign On-Premise Agentic AI Workbench.
Splits refinery SOPs, engineering standards, and technical manuals by headings,
sections, tables, and document outlines rather than arbitrary character splits.
Preserves table headers across splits and annotates chunks with hierarchical breadcrumbs.
"""
from typing import Any, Dict, List, Optional, Tuple
import re
from pathlib import Path
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_name: str
    category: str = "General"
    section_title: str
    page_number: int = 1
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunker:
    """Structure-preserving text, markdown, PDF, and docx chunker."""

    MAX_CHUNK_CHARS: int = 1500

    @classmethod
    def chunk_markdown_document(
        cls, doc_name: str, content: str, category: str = "General"
    ) -> List[DocumentChunk]:
        """
        Splits markdown on headers (#, ##, ###) while preserving section hierarchy and tables.
        If a section or table exceeds MAX_CHUNK_CHARS, splits safely without corrupting table rows.
        """
        lines = content.splitlines()
        chunks: List[DocumentChunk] = []

        current_hierarchy: List[str] = ["Overview"]
        current_lines: List[str] = []
        chunk_idx = 1
        current_page = 1

        def flush_chunk(hierarchy: List[str], page: int, text: str, meta: Optional[dict] = None):
            nonlocal chunk_idx, chunks
            text = text.strip()
            if len(text) <= 20:
                return

            sec_title = " > ".join(hierarchy) if hierarchy else "Overview"
            meta = meta or {}
            meta["breadcrumbs"] = list(hierarchy)

            # Check if text is predominantly a markdown table
            is_table = cls._is_markdown_table(text)
            meta["is_table"] = is_table

            if is_table:
                table_chunks = cls._split_markdown_table(
                    doc_name=doc_name,
                    category=category,
                    section_title=sec_title,
                    page=page,
                    table_text=text,
                    start_idx=chunk_idx,
                    meta=meta,
                )
                chunks.extend(table_chunks)
                chunk_idx += len(table_chunks)
            elif len(text) > cls.MAX_CHUNK_CHARS:
                paras = text.split("\n\n")
                sub_text = ""
                for p in paras:
                    if len(sub_text) + len(p) > cls.MAX_CHUNK_CHARS and sub_text:
                        sub_meta = dict(meta)
                        sub_meta["is_part"] = True
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{doc_name}_chk_{chunk_idx:03d}",
                                document_name=doc_name,
                                category=category,
                                section_title=f"{sec_title} (Part)",
                                page_number=page,
                                content=sub_text.strip(),
                                metadata=sub_meta,
                            )
                        )
                        chunk_idx += 1
                        sub_text = p + "\n\n"
                    else:
                        sub_text += p + "\n\n"
                if sub_text.strip():
                    sub_meta = dict(meta)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{doc_name}_chk_{chunk_idx:03d}",
                            document_name=doc_name,
                            category=category,
                            section_title=f"{sec_title} (Part)" if chunk_idx > 1 else sec_title,
                            page_number=page,
                            content=sub_text.strip(),
                            metadata=sub_meta,
                        )
                    )
                    chunk_idx += 1
            else:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_name}_chk_{chunk_idx:03d}",
                        document_name=doc_name,
                        category=category,
                        section_title=sec_title,
                        page_number=page,
                        content=text,
                        metadata=meta,
                    )
                )
                chunk_idx += 1

        current_meta: Dict[str, Any] = {}
        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for page markers e.g. <!-- Page 2 -->
            page_match = re.search(r"<!--\s*Page\s*(\d+)\s*-->", line, re.IGNORECASE)
            if page_match:
                current_page = int(page_match.group(1))

            header_match = re.match(r"^(#{1,4})\s+(.+)$", line)
            if header_match:
                if current_lines:
                    flush_chunk(current_hierarchy, current_page, "\n".join(current_lines), current_meta)
                    current_lines = []
                level = len(header_match.group(1))
                heading_text = header_match.group(2).strip()

                # Adjust hierarchy depth
                current_hierarchy = current_hierarchy[: level - 1]
                current_hierarchy.append(heading_text)
                current_meta = {"header_level": level, "heading": heading_text}
                current_lines.append(line)
            else:
                current_lines.append(line)
            i += 1

        if current_lines:
            flush_chunk(current_hierarchy, current_page, "\n".join(current_lines), current_meta)

        return chunks

    @staticmethod
    def _is_markdown_table(text: str) -> bool:
        """Determines if a block of text is formatted as a Markdown table."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return False
        table_lines = sum(1 for l in lines if l.startswith("|") and l.endswith("|"))
        return table_lines >= 2 and (table_lines / len(lines)) > 0.6

    @classmethod
    def _split_markdown_table(
        cls,
        doc_name: str,
        category: str,
        section_title: str,
        page: int,
        table_text: str,
        start_idx: int,
        meta: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """
        Splits a large Markdown table while preserving header rows across all chunks.
        """
        lines = [l.strip() for l in table_text.splitlines() if l.strip()]
        if len(lines) <= 2:
            return [
                DocumentChunk(
                    chunk_id=f"{doc_name}_chk_{start_idx:03d}",
                    document_name=doc_name,
                    category=category,
                    section_title=section_title,
                    page_number=page,
                    content=table_text,
                    metadata=dict(meta, is_table=True),
                )
            ]

        # Extract header and separator
        header_row = lines[0]
        sep_row = lines[1] if re.match(r"^\|[\s\-:|]+\|$", lines[1]) else "|---|---|"
        data_rows = lines[2:] if re.match(r"^\|[\s\-:|]+\|$", lines[1]) else lines[1:]

        chunks: List[DocumentChunk] = []
        idx = start_idx
        current_rows: List[str] = []
        current_len = len(header_row) + len(sep_row) + 2

        for row in data_rows:
            if current_len + len(row) > cls.MAX_CHUNK_CHARS and current_rows:
                chunk_body = f"{header_row}\n{sep_row}\n" + "\n".join(current_rows)
                chunk_meta = dict(meta)
                chunk_meta.update({"is_table": True, "rows_count": len(current_rows), "table_split": True})
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_name}_chk_{idx:03d}",
                        document_name=doc_name,
                        category=category,
                        section_title=f"{section_title} (Table Part {len(chunks)+1})",
                        page_number=page,
                        content=chunk_body,
                        metadata=chunk_meta,
                    )
                )
                idx += 1
                current_rows = [row]
                current_len = len(header_row) + len(sep_row) + len(row) + 3
            else:
                current_rows.append(row)
                current_len += len(row) + 1

        if current_rows:
            chunk_body = f"{header_row}\n{sep_row}\n" + "\n".join(current_rows)
            chunk_meta = dict(meta)
            chunk_meta.update({"is_table": True, "rows_count": len(current_rows)})
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_name}_chk_{idx:03d}",
                    document_name=doc_name,
                    category=category,
                    section_title=f"{section_title} (Table)" if chunks else section_title,
                    page_number=page,
                    content=chunk_body,
                    metadata=chunk_meta,
                )
            )

        return chunks

    @classmethod
    def _extract_pdf_outline(cls, reader: Any) -> List[Dict[str, Any]]:
        """Walks reader.outline to extract document bookmarks and their corresponding pages."""
        entries: List[Dict[str, Any]] = []

        def _walk(outline_items, level=1):
            for item in outline_items:
                if isinstance(item, list):
                    _walk(item, level=level + 1)
                else:
                    try:
                        dest_page = reader.get_destination_page_number(item) + 1
                        title = getattr(item, "title", str(item)).strip()
                        if title:
                            entries.append({"title": title, "page": dest_page, "level": level})
                    except Exception:
                        pass

        try:
            if getattr(reader, "outline", None):
                _walk(reader.outline, level=1)
        except Exception:
            pass

        return entries

    @classmethod
    def _get_active_outline_hierarchy(
        cls, outline_entries: List[Dict[str, Any]], page_num: int
    ) -> List[str]:
        """Finds the active section hierarchy for a specific page number from the outline."""
        if not outline_entries:
            return []

        valid_items = [e for e in outline_entries if e["page"] <= page_num]
        if not valid_items:
            return [outline_entries[0]["title"]]

        hierarchy: Dict[int, str] = {}
        for item in valid_items:
            lvl = item["level"]
            hierarchy[lvl] = item["title"]
            for k in list(hierarchy.keys()):
                if k > lvl:
                    del hierarchy[k]

        return [hierarchy[k] for k in sorted(hierarchy.keys())]

    @classmethod
    def chunk_pdf_document(
        cls, doc_name: str, file_path: str, category: str = "General"
    ) -> List[DocumentChunk]:
        """
        Extracts and chunks pages from a PDF document using layout mode and document outlines.
        Maps outline chapters and headings directly to chunk breadcrumbs.
        """
        chunks: List[DocumentChunk] = []
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            chunk_idx = 1
            total_pages = len(reader.pages)

            # 1. Extract Outline / Bookmarks
            outline_entries = cls._extract_pdf_outline(reader)

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = ""
                try:
                    page_text = page.extract_text(extraction_mode="layout") or ""
                except Exception:
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        pass

                page_text = page_text.strip()
                if not page_text:
                    continue

                # 2. Determine Section Hierarchy & Title
                outline_hierarchy = cls._get_active_outline_hierarchy(outline_entries, page_num)
                if outline_hierarchy:
                    sec_title = " > ".join(outline_hierarchy)
                    breadcrumbs = list(outline_hierarchy)
                else:
                    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
                    detected_heading = f"Page {page_num} Overview"
                    for line in lines[:4]:
                        if re.match(r"^(?:chapter|section|\d+\.)\s+", line, re.IGNORECASE) or (
                            len(line) < 70 and not line.isdigit() and not line.endswith(".")
                        ):
                            detected_heading = line
                            break
                    sec_title = detected_heading
                    breadcrumbs = [f"Page {page_num}", detected_heading]

                # 3. Check for tabular structures in page
                is_table = cls._detect_tabular_layout(page_text)

                # 4. Split page into coherent paragraph chunks
                paras = [p.strip() for p in re.split(r"\n\s*\n", page_text) if len(p.strip()) > 30]
                if not paras:
                    paras = [page_text]

                for p_idx, para in enumerate(paras):
                    p_is_table = is_table or cls._detect_tabular_layout(para)
                    chunk_meta = {
                        "page": page_num,
                        "total_pages": total_pages,
                        "source": "pdf",
                        "breadcrumbs": breadcrumbs,
                        "is_table": p_is_table,
                    }

                    chunk_sec_title = sec_title if p_idx == 0 else f"{sec_title} (Part {p_idx+1})"

                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{doc_name}_p{page_num:02d}_chk_{chunk_idx:03d}",
                            document_name=doc_name,
                            category=category,
                            section_title=chunk_sec_title,
                            page_number=page_num,
                            content=para,
                            metadata=chunk_meta,
                        )
                    )
                    chunk_idx += 1

        except Exception as e:
            pass

        return chunks

    @staticmethod
    def _detect_tabular_layout(text: str) -> bool:
        """Detects whether text has aligned columns or pipe tables."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return False
        multi_col_lines = 0
        for l in lines:
            if len(re.findall(r"\S+\s{2,}\S+", l)) >= 2 or "|" in l:
                multi_col_lines += 1
        return (multi_col_lines / len(lines)) >= 0.4

    @classmethod
    def chunk_plain_text(
        cls, doc_name: str, content: str, category: str = "General"
    ) -> List[DocumentChunk]:
        """Splits raw text into structured chunks with paragraph boundaries."""
        chunks: List[DocumentChunk] = []
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 20]
        if not paragraphs:
            paragraphs = [content.strip()]

        for i, para in enumerate(paragraphs, start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_name}_chk_{i:03d}",
                    document_name=doc_name,
                    category=category,
                    section_title=f"Section {i}",
                    page_number=1,
                    content=para,
                    metadata={"breadcrumbs": [doc_name, f"Section {i}"]},
                )
            )
        return chunks
