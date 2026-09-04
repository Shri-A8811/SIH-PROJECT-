"""
Multimodal Document Extractor for Sovereign On-Premise Agentic AI Workbench.
Processes scanned refinery inspection reports, arbitrary uploaded PDFs, Word docs, and logs.
Supports both structured turnaround NDT extraction and dynamic uploaded Document Q&A.
"""
from typing import Any, Dict, List, Optional
from pathlib import Path
import time
import json
import re
import tempfile
from config.settings import settings
from src.core.state_store import StateStore
from src.models.model_client import ModelClient


class MultimodalDocumentExtractor:
    """Extracts safety-critical findings and answers queries on uploaded documents."""

    def __init__(self, state_store: StateStore, model_client: ModelClient):
        self.state_store = state_store
        self.model_client = model_client

    def parse_document_pages(self, document_path: str) -> List[Dict[str, Any]]:
        """Parses pages and sections from PDF, DOCX, Markdown, or text files."""
        path = Path(document_path)
        if not path.exists():
            return []

        ext = path.suffix.lower()
        pages = []

        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                for idx, p in enumerate(reader.pages):
                    text = p.extract_text() or ""
                    pages.append({
                        "page_number": idx + 1,
                        "text": text,
                        "source": path.name,
                    })
            except Exception as e:
                pages.append({"page_number": 1, "text": f"Error parsing PDF: {e}", "source": path.name})

        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(str(path))
                full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                pages.append({"page_number": 1, "text": full_text, "source": path.name})
            except Exception as e:
                pages.append({"page_number": 1, "text": f"Error parsing DOCX: {e}", "source": path.name})

        else:
            # Plain text / Markdown
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # Split by markdown headers or page comments if available
                sections = re.split(r"(?:\n# |\n<!-- Page \d+ -->)", content)
                for idx, sec in enumerate(sections):
                    if sec.strip():
                        pages.append({
                            "page_number": idx + 1,
                            "text": sec.strip(),
                            "source": path.name,
                        })
            except Exception as e:
                pages.append({"page_number": 1, "text": f"Error reading text file: {e}", "source": path.name})

        return pages

    def answer_uploaded_document_query(
        self,
        document_path: str,
        user_prompt: str,
        project_id: str,
    ) -> Dict[str, Any]:
        """
        Dynamically searches the uploaded document and answers user queries with page grounding.
        """
        pages = self.parse_document_pages(document_path)
        if not pages:
            return {
                "answer": f"Could not read content from uploaded document: {Path(document_path).name}",
                "cited_pages": [],
                "document_name": Path(document_path).name,
            }

        # Score pages by relevance to user_prompt
        query_words = set(re.findall(r"\w+", user_prompt.lower()))
        scored_pages = []
        for p in pages:
            text_lower = p["text"].lower()
            match_count = sum(1 for w in query_words if w in text_lower)
            # Extra boost if exact phrase or chapter/section matches
            if user_prompt.lower() in text_lower:
                match_count += 5
            scored_pages.append((match_count, p))

        scored_pages.sort(key=lambda x: x[0], reverse=True)
        top_pages = [p for score, p in scored_pages[:4] if score > 0]
        if not top_pages:
            top_pages = pages[:3]

        context_blocks = []
        cited_pages = []
        for p in top_pages:
            cited_pages.append(p["page_number"])
            context_blocks.append(f"--- [Page {p['page_number']} of {p['source']}] ---\n{p['text'][:2000]}")

        joined_context = "\n\n".join(context_blocks)

        prompt = f"""You are an expert technical document analyst. Answer the user's question accurately based ONLY on the uploaded document excerpt below.

Document Excerpt:
{joined_context}

User Question: {user_prompt}

Provide a detailed, well-structured answer citing specific page numbers and headings from the document excerpt:"""

        # Generate answer using resident model (qwen2.5vl:3b is fast and accurate)
        res = self.model_client.generate_text(
            model_name="qwen2.5vl:3b",
            prompt=prompt,
            max_tokens=600,
            project_id=project_id,
        )

        answer_text = res.get("response", "")
        if not answer_text or "Inference error" in answer_text:
            # High-fidelity extractive summary fallback directly from parsed pages
            extracted_sections = []
            for p in top_pages:
                lines = [line.strip() for line in p["text"].split("\n") if line.strip()]
                preview = "\n".join(lines[:15])
                extracted_sections.append(f"### 📑 Section Excerpt (Page {p['page_number']})\n{preview}")
            answer_text = (
                f"### 📄 Excerpt Extracted from '{Path(document_path).name}' for: *'{user_prompt}'*\n\n"
                + "\n\n---\n\n".join(extracted_sections)
            )

        return {
            "answer": answer_text,
            "cited_pages": cited_pages,
            "document_name": Path(document_path).name,
            "inference_duration_ms": res.get("inference_duration_ms", 0.0),
        }

    def extract_inspection_report(
        self,
        document_path: str,
        project_id: str,
        task_id: str = "T001",
    ) -> Dict[str, Any]:
        """
        Parses scanned refinery report, extracts equipment anomalies,
        and registers rows in the persistent evidence table with unique evidence_ids.
        """
        doc_file = Path(document_path)
        if not doc_file.is_file():
            return {"status": "error", "error": f"Inspection document not found: {document_path}", "findings": []}
        doc_name = doc_file.name
        start_time = time.time()
        pages = self.parse_document_pages(document_path)
        text_context = "\n\n".join(
            f"[Page {p['page_number']}] {p['text'][:5000]}" for p in pages if p.get("text", "").strip()
        )[:18000]
        prompt = """Extract only findings explicitly supported by this inspection document. Return JSON only:
{"findings":[{"page_number":1,"section":"...","equipment":"...","issue":"...","severity":"Critical|High|Medium|Low","measured_value":"...","nominal_value":"...","threshold_limit":"...","status":"...","raw_text_snippet":"verbatim supporting excerpt"}]}
Do not infer measurements, equipment, standards, or recommendations that are absent from the source."""

        raw_response = ""
        ext = doc_file.suffix.lower()
        image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
        with tempfile.TemporaryDirectory(prefix="workbench_ocr_") as temp_dir:
            image_paths: List[str] = []
            if ext in image_extensions:
                image_paths = [str(doc_file)]
            elif ext == ".pdf" and not text_context:
                try:
                    import fitz  # PyMuPDF, pinned as an offline dependency
                    pdf = fitz.open(str(doc_file))
                    for index, page in enumerate(pdf):
                        if index >= 20:
                            break
                        out_path = Path(temp_dir) / f"page_{index + 1}.png"
                        page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(str(out_path))
                        image_paths.append(str(out_path))
                except Exception as exc:
                    return {"status": "error", "error": f"Scanned PDF requires local PyMuPDF rendering: {exc}", "findings": []}

            if image_paths:
                vision = self.model_client.generate_with_images(
                    settings.models.vision, prompt, image_paths, project_id, task_id
                )
                if vision.get("status") != "ok":
                    return {"status": vision.get("status"), "error": vision.get("error"), "findings": []}
                raw_response = vision.get("response", "")
            elif text_context:
                result = self.model_client.generate_text(
                    settings.models.reasoning, f"{prompt}\n\nSOURCE TEXT:\n{text_context}",
                    max_tokens=1800, project_id=project_id, task_id=task_id,
                )
                raw_response = result.get("response", "")
            else:
                return {"status": "error", "error": "No extractable text or renderable pages found.", "findings": []}

        parsed = self.model_client.extract_json_from_response(raw_response)
        if not parsed or not isinstance(parsed.get("findings"), list):
            return {"status": "error", "error": "Local model did not return valid grounded extraction JSON.", "findings": []}

        extracted_findings = []
        for index, finding in enumerate(parsed["findings"], start=1):
            if not isinstance(finding, dict) or not finding.get("raw_text_snippet"):
                continue
            finding = dict(finding)
            finding["evidence_id"] = f"E_OCR_{task_id}_{index:03d}"
            finding["page_number"] = int(finding.get("page_number", 1))
            extracted_findings.append(finding)

        # Register each finding into the persistent evidence table
        registered_evidence_ids = []
        for f in extracted_findings:
            e_id = f["evidence_id"]
            self.state_store.add_evidence(
                evidence_id=e_id,
                project_id=project_id,
                source_type="local_multimodal_extraction",
                source_document=doc_name,
                page_number=f["page_number"],
                section=f.get("section", "Unsectioned finding"),
                extracted_text=f["raw_text_snippet"],
                structured_data={
                    "equipment": f.get("equipment", "Unidentified equipment"),
                    "issue": f.get("issue", "Unspecified finding"),
                    "severity": f.get("severity", "UNASSESSED"),
                    "measured_value": f.get("measured_value", ""),
                    "nominal_value": f.get("nominal_value", ""),
                    "threshold_limit": f.get("threshold_limit", ""),
                    "status": f.get("status", "UNASSESSED"),
                },
                confidence=0.75,
            )
            registered_evidence_ids.append(e_id)

        used_model = settings.models.vision if image_paths else settings.models.reasoning
        duration_ms = (time.time() - start_time) * 1000
        self.state_store.log_model_activity(
            model_name=used_model,
            action="INFERENCE",
            project_id=project_id,
            task_id=task_id,
            duration_ms=duration_ms,
            details={"findings_count": len(extracted_findings)},
        )

        return {
            "document_name": doc_name,
            "total_findings_extracted": len(extracted_findings),
            "evidence_ids": registered_evidence_ids,
            "findings": extracted_findings,
            "status": "extraction_complete" if extracted_findings else "no_grounded_findings",
        }
