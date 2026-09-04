"""
Settings and Configuration for Sovereign On-Premise Agentic AI Workbench (SIH26117 - MRPL).
Zero cloud dependencies, air-gap strict, single-GPU lifecycle aware.
"""
from pathlib import Path
from pydantic import BaseModel, Field
import os
from urllib.parse import urlparse

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
SAMPLE_INPUTS_DIR = DATA_DIR / "sample_inputs"
TEMPLATES_DIR = DATA_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ModelTags(BaseModel):
    reasoning: str = Field(default="qwen3.5:9b", description="General reasoning, planning, synthesis")
    ocr: str = Field(default="frob/unlimited-ocr:3b", description="Specialized scanned document OCR")
    coding: str = Field(default="qwen2.5-coder:7b", description="Deterministic script & sandbox code generation")
    vision: str = Field(default="qwen2.5vl:3b", description="Multimodal visual reasoning, diagrams, gauges")
    embeddings: str = Field(default="qllama/bge-small-en-v1.5:f16", description="Local vector embeddings")
    reranker: str = Field(default="bbjson/bge-reranker-base:latest", description="Local cross-encoder reranker")

class Settings(BaseModel):
    app_name: str = "Sovereign On-Premise Agentic AI Workbench (MRPL)"
    base_dir: Path = BASE_DIR
    BASE_DIR: Path = BASE_DIR
    # Database configuration: Default to local SQLite for instant zero-dependency start, or PostgreSQL if DATABASE_URL is set
    database_url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR / 'workbench_state.db'}"
    )
    vector_dimension: int = 384
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    
    # Model configuration
    models: ModelTags = ModelTags()
    
    # Never manufacture industrial findings when a model is unavailable. Test-only
    # fixtures may opt in explicitly through WORKBENCH_TEST_MODE.
    enable_deterministic_mock_fallback: bool = os.getenv("WORKBENCH_TEST_MODE", "").lower() in {"1", "true", "yes"}
    
    # Retrieval thresholds
    reranker_min_relevance_score: float = 0.35
    rag_min_relevance_score: float = 0.35
    top_k_retrieval: int = 5
    
    # Sandbox security limits
    sandbox_timeout_seconds: int = 15
    sandbox_max_memory_mb: int = 256
    sandbox_cpu_quota_percent: int = 50
    sandbox_pids_limit: int = 30
    
    # Validation retry limit
    max_validation_retries: int = 3
    
    # Air-gap strictly local
    air_gap_enforced: bool = True

settings = Settings()


def validate_local_ollama_endpoint(endpoint: str) -> str:
    """Reject non-loopback model endpoints before any confidential prompt is sent."""
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError(
            "OLLAMA_BASE_URL must point to a loopback-only local service; remote model endpoints are forbidden."
        )
    return endpoint.rstrip("/")
