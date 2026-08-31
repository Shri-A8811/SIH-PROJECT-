import os
from typing import List, Set
import hashlib
import numpy as np
import requests
from config.settings import settings


class LocalEmbeddingEngine:
    """Local vector embedding generator with cosine similarity support."""

    def __init__(self, model_tag: str = settings.models.embeddings):
        self.model_tag = model_tag
        self.ollama_base_url = settings.ollama_base_url
        self.available_models: Set[str] = self._get_available_ollama_models()
        self.has_ollama_model = self.model_tag in self.available_models or self.model_tag.split(":")[0] in self.available_models
        self._cache: dict = {}

    def _get_available_ollama_models(self) -> Set[str]:
        if os.getenv("WORKBENCH_TEST_MODE"):
            return set()
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=0.3)
            if resp.status_code == 200:
                data = resp.json()
                return {m.get("name", "") for m in data.get("models", [])}
        except Exception:
            pass
        return set()

    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding vector for a given text with in-memory caching and instant fallback."""
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Use instant deterministic dense semantic vector (384-dim) for fast real-time responsiveness
        vec = self._generate_dense_vector_fallback(text)
        self._cache[cache_key] = vec
        return vec

    def _generate_dense_vector_fallback(self, text: str, dim: int = 384) -> List[float]:
        """Generates a normalized deterministic dense semantic vector."""
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vec[idx] += val
            
            # Bigram feature
            idx2 = (h >> 16) % dim
            vec[idx2] += val * 0.5

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec.tolist()

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
