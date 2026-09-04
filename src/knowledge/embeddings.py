import os
from typing import List, Optional, Set
import hashlib
import numpy as np
import requests
import logging
from config.settings import settings, validate_local_ollama_endpoint

logger = logging.getLogger(__name__)


class LocalEmbeddingEngine:
    """Local vector embedding generator with BGE neural embeddings and cosine similarity support."""

    def __init__(self, model_tag: str = settings.models.embeddings):
        self.model_tag = model_tag
        self.ollama_base_url = validate_local_ollama_endpoint(settings.ollama_base_url)
        self.available_models: Set[str] = self._get_available_ollama_models()
        self.has_ollama_model = (
            self.model_tag in self.available_models
            or self.model_tag.split(":")[0] in self.available_models
            or any(self.model_tag.split(":")[0] in m for m in self.available_models)
        )
        self._cache: dict = {}

    def _get_available_ollama_models(self) -> Set[str]:
        if os.getenv("WORKBENCH_TEST_MODE"):
            return set()
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                return {m.get("name", "") for m in data.get("models", [])}
        except Exception:
            pass
        return set()

    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding vector for a single text."""
        return self.get_embeddings_batch([text])[0]

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates embedding vectors for a batch of text chunks using Ollama /api/embed.
        Uses in-memory MD5 cache and falls back gracefully to dense vector simulation.
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache
        for idx, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
            if cache_key in self._cache:
                results[idx] = self._cache[cache_key]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if not uncached_texts:
            return [r for r in results if r is not None]

        # Process uncached in batches via Ollama /api/embed
        ollama_active = self.has_ollama_model and not os.getenv("WORKBENCH_TEST_MODE")

        for i in range(0, len(uncached_texts), batch_size):
            chunk_texts = uncached_texts[i : i + batch_size]
            chunk_indices = uncached_indices[i : i + batch_size]
            batch_vectors: Optional[List[List[float]]] = None

            if ollama_active:
                try:
                    resp = requests.post(
                        f"{self.ollama_base_url}/api/embed",
                        json={"model": self.model_tag, "input": chunk_texts},
                        timeout=15.0,
                    )
                    if resp.status_code == 200:
                        raw_embeddings = resp.json().get("embeddings", [])
                        if len(raw_embeddings) == len(chunk_texts):
                            batch_vectors = []
                            for emb in raw_embeddings:
                                vec = np.array(emb, dtype=np.float32)
                                norm = np.linalg.norm(vec)
                                if norm > 1e-6:
                                    vec = vec / norm
                                batch_vectors.append(vec.tolist())
                except Exception as e:
                    logger.debug(f"Ollama batch embed fallback: {e}")

            # Fallback if Ollama call failed or returned empty
            if not batch_vectors:
                batch_vectors = [self._generate_dense_vector_fallback(t) for t in chunk_texts]

            for orig_idx, text, vec in zip(chunk_indices, chunk_texts, batch_vectors):
                cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
                self._cache[cache_key] = vec
                results[orig_idx] = vec

        return [r for r in results if r is not None]

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
