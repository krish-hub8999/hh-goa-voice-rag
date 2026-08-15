"""Embedding backends with a fast zero-download fallback for local demos."""

import hashlib
import re
from collections.abc import Iterable

import numpy as np


class HashingEmbedder:
    """Deterministic multilingual-ish char/word hashing embedder.

    It is intentionally dependency-light and suitable for smoke tests. For
    quality production retrieval, use SentenceTransformerEmbedder.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _features(self, text: str) -> Iterable[str]:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        words = normalized.split()
        yield from words
        for n in (3, 4, 5):
            yield from (normalized[i : i + n] for i in range(max(0, len(normalized) - n + 1)))

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the optional quality extra for sentence-transformers") from exc
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        prefix = "query: " if query else "passage: "
        vectors = self.model.encode(
            [prefix + text for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
