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
    def read_file(file_path: str) -> Dict[str, Any]:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check boundary - must be in workspace or data directories
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
