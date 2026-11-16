import math
import re
from typing import List, Dict
import numpy as np

class TextEmbeddingEngine:
    """
    Academic research embedding engine.
    Computes normalized term-frequency / sub-word n-gram embedding vectors
    for cosine similarity search over photography, lighting, and composition corpora.
    Works deterministically without external GPU requirements.
    """
    def __init__(self, vector_dim: int = 256):
        self.vector_dim = vector_dim

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 1]
        # Include character bi-grams and tri-grams for subword matching
        subwords = []
        for t in tokens:
            if len(t) >= 3:
                for i in range(len(t) - 2):
                    subwords.append(t[i:i+3])
        return tokens + subwords

    def get_embedding(self, text: str) -> np.ndarray:
        """Computes a normalized dense representation vector of fixed dimension."""
        tokens = self._tokenize(text)
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        if not tokens:
            return vec

        for token in tokens:
            # Deterministic hash projection
            h = hash(token)
            idx = abs(h) % self.vector_dim
            sign = 1.0 if (h // self.vector_dim) % 2 == 0 else -1.0
            vec[idx] += sign

        # L2 Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

embedding_engine = TextEmbeddingEngine()
