"""
Structure-Aware Document Chunker for Sovereign On-Premise Agentic AI Workbench.
Splits refinery SOPs, engineering standards, and technical manuals by headings,
sections, and tables rather than arbitrary character splits.
"""
from typing import Any, Dict, List
import re
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_name: str
    section_title: str
    page_number: int = 1
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunker:
    """Structure-preserving text and markdown chunker."""

    @staticmethod
    def chunk_markdown_document(doc_name: str, content: str) -> List[DocumentChunk]:
        """Splits markdown on headers (#, ##, ###) while preserving section hierarchy."""
        lines = content.splitlines()
        chunks: List[DocumentChunk] = []
        
        current_section = "General Overview"
        current_lines: List[str] = []
        chunk_idx = 1
        current_page = 1

        for line in lines:
            # Check for page markers e.g. <!-- Page 2 -->
            page_match = re.search(r"<!--\s*Page\s*(\d+)\s*-->", line, re.IGNORECASE)
            if page_match:
                current_page = int(page_match.group(1))

            header_match = re.match(r"^(#{1,4})\s+(.+)$", line)
            if header_match:
                if current_lines:
                    chunk_text = "\n".join(current_lines).strip()
                    if len(chunk_text) > 30:
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{doc_name}_chk_{chunk_idx:03d}",
                                document_name=doc_name,
                                section_title=current_section,
                                page_number=current_page,
                                content=chunk_text,
                                metadata={"header_level": header_match.group(1)},
                            )
                        )
                        chunk_idx += 1
                    current_lines = []
                current_section = header_match.group(2).strip()
                current_lines.append(line)
            else:
                current_lines.append(line)

        # Flush final chunk
        if current_lines:
            chunk_text = "\n".join(current_lines).strip()
            if len(chunk_text) > 30:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_name}_chk_{chunk_idx:03d}",
                        document_name=doc_name,
                        section_title=current_section,
                        page_number=current_page,
                        content=chunk_text,
                    )
                )

        return chunks
    @staticmethod
    def chunk_pdf_document(doc_name: str, file_path: str) -> List[DocumentChunk]:
        """Extracts and chunks pages from a PDF document using pypdf."""
        chunks: List[DocumentChunk] = []
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            chunk_idx = 1
            
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if not page_text:
                    continue
                
                # Identify first non-empty line as section title heuristic
                lines = [l.strip() for l in page_text.splitlines() if l.strip()]
                sec_title = f"Page {page_num} Overview"
                for line in lines[:3]:
                    if len(line) < 80 and not line.isdigit():
                        sec_title = line
                        break

                # If page is long, split by paragraph
                paras = [p.strip() for p in page_text.split("\n\n") if len(p.strip()) > 30]
                if not paras:
                    paras = [page_text]

                for p_idx, para in enumerate(paras):
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{doc_name}_p{page_num:02d}_chk_{chunk_idx:03d}",
                            document_name=doc_name,
                            section_title=sec_title if p_idx == 0 else f"{sec_title} (Cont.)",
                            page_number=page_num,
                            content=para,
                            metadata={"page": page_num, "source": "pdf"},
                        )
                    )
                    chunk_idx += 1
        except Exception as e:
            pass

        return chunks

    @staticmethod
    def chunk_plain_text(doc_name: str, content: str) -> List[DocumentChunk]:
        """Splits raw text into structured chunks based on paragraphs and line breaks."""
        chunks: List[DocumentChunk] = []
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 20]
        if not paragraphs:
            paragraphs = [content.strip()]

        for i, para in enumerate(paragraphs, start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_name}_chk_{i:03d}",
                    document_name=doc_name,
                    section_title=f"Section {i}",
                    page_number=1,
                    content=para,
                )
            )
        return chunks
