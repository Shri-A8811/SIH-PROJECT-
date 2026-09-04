"""
File Tools for Sovereign On-Premise Agentic AI Workbench.
Provides secure, bounded file read/write operations for internal workspace files.
"""
from typing import Any, Dict, List, Optional
import os
from pathlib import Path
from config.settings import BASE_DIR, DATA_DIR, OUTPUT_DIR


class FileTools:
    """Safe workspace file read and write operations."""

    @staticmethod
    def _is_within(path: Path, roots: List[Path]) -> bool:
        return any(path == root.resolve() or root.resolve() in path.parents for root in roots)

    @staticmethod
    def read_file(file_path: str) -> Dict[str, Any]:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not FileTools._is_within(path, [DATA_DIR, OUTPUT_DIR]):
            raise PermissionError("Read denied: files must be inside the workbench data or outputs directories.")
        if path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("Read denied: file exceeds the 20 MiB tool limit.")
        content = path.read_text(encoding="utf-8", errors="replace")
        return {
            "file_path": str(path),
            "file_name": path.name,
            "size_bytes": path.stat().st_size,
            "content": content,
        }

    @staticmethod
    def write_file(file_path: str, content: str) -> Dict[str, Any]:
        path = Path(file_path).resolve()
        if not FileTools._is_within(path, [OUTPUT_DIR]):
            raise PermissionError("Write denied: agent-generated files must be written under outputs only.")
        if len(content.encode("utf-8")) > 20 * 1024 * 1024:
            raise ValueError("Write denied: content exceeds the 20 MiB tool limit.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "file_path": str(path),
            "file_name": path.name,
            "size_bytes": path.stat().st_size,
            "status": "written_successfully",
        }

    @staticmethod
    def list_files(directory_path: Optional[str] = None) -> List[Dict[str, Any]]:
        target_dir = Path(directory_path).resolve() if directory_path else DATA_DIR
        if not FileTools._is_within(target_dir, [DATA_DIR, OUTPUT_DIR]):
            raise PermissionError("List denied: directories must be inside workbench data or outputs.")
        if not target_dir.exists():
            return []
        
        files_info = []
        for p in target_dir.rglob("*"):
            if p.is_file():
                files_info.append({
                    "name": p.name,
                    "relative_path": str(p.relative_to(BASE_DIR)),
                    "size_bytes": p.stat().st_size,
                })
        return files_info
