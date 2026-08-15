"""Persistent local vector store using FAISS when available, NumPy otherwise."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import Chunk
from .schemas import RetrievedChunk


class LocalVectorStore:
    def __init__(self, directory: Path, embedder: Any):
        self.directory = Path(directory)
        self.embedder = embedder
        self.vectors: np.ndarray | None = None
        self.records: list[dict[str, Any]] = []
        self._faiss = None
        self.last_embed_ms = 0.0

    @property
    def loaded(self) -> bool:
        return self.vectors is not None and bool(self.records)

    def build(self, chunks: list[Chunk], *, batch_size: int = 128) -> None:
        self.records = [c.__dict__ for c in chunks]
        vectors = []
        for start in range(0, len(chunks), batch_size):
            vectors.append(self.embedder.encode([c.text for c in chunks[start : start + batch_size]]))
        self.vectors = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 0), dtype=np.float32)
        self._build_faiss()

    def _build_faiss(self) -> None:
        self._faiss = None
        try:
            import faiss

            if self.vectors is not None and len(self.vectors):
                self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
                self._faiss.add(self.vectors)
        except ImportError:
            pass

    def save(self) -> None:
        if self.vectors is None:
            raise ValueError("Cannot save an empty store")
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / "vectors.npy", self.vectors)
        with (self.directory / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        (self.directory / "manifest.json").write_text(
            json.dumps({"count": len(self.records), "dimensions": int(self.vectors.shape[1])}),
            encoding="utf-8",
        )

    def load(self) -> None:
        self.vectors = np.load(self.directory / "vectors.npy", mmap_mode="r")
        with (self.directory / "records.jsonl").open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        self._build_faiss()

    def search(self, query: str, *, top_k: int = 5, candidate_k: int = 24) -> list[RetrievedChunk]:
        if not self.loaded:
            return []
        embed_started = time.perf_counter()
        vector = self.embedder.encode([query], query=True)
        self.last_embed_ms = (time.perf_counter() - embed_started) * 1000
        candidate_k = min(candidate_k, len(self.records))
        if self._faiss is not None:
            scores, indices = self._faiss.search(vector, candidate_k)
            pairs = list(zip(scores[0].tolist(), indices[0].tolist()))
        else:
            scores = (self.vectors @ vector[0]).astype(np.float32)
            indices = np.argpartition(-scores, candidate_k - 1)[:candidate_k]
            pairs = sorted(((float(scores[i]), int(i)) for i in indices), reverse=True)

        query_terms = set(query.lower().split())
        reranked = []
        for vector_score, index in pairs:
            record = self.records[index]
            text_terms = set(str(record["text"]).lower().split())
            lexical = len(query_terms & text_terms) / max(1, len(query_terms))
            selected_boost = 0.015 if record.get("is_selected") else 0.0
            combined = 0.85 * float(vector_score) + 0.15 * lexical + selected_boost
            reranked.append((combined, record))
        reranked.sort(key=lambda item: item[0], reverse=True)

        results: list[RetrievedChunk] = []
        seen_passages: set[tuple[str | int | None, int | None]] = set()
        for score, record in reranked:
            passage_key = (record.get("query_id"), record.get("passage_index"))
            if passage_key in seen_passages and len(results) < top_k - 1:
                continue
            seen_passages.add(passage_key)
            results.append(
                RetrievedChunk(
                    chunk_id=record["chunk_id"],
                    text=record["text"],
                    score=round(float(score), 6),
                    strategy=record["strategy"],
                    language=record["language"],
                    query_id=record.get("query_id"),
                    passage_index=record.get("passage_index"),
                    is_selected=bool(record.get("is_selected")),
                )
            )
            if len(results) >= top_k:
                break
        return results
