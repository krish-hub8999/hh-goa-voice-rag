"""Multilingual chunking strategies used during indexing.

The default hybrid strategy preserves dataset passage boundaries, splits long
passages on sentence/paragraph boundaries, and uses a small overlap only when
a passage is too long. This avoids breaking short answers while still keeping
long web passages searchable.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    strategy: str
    query_id: str | int | None
    language: str
    passage_index: int
    is_selected: bool
    neighbor_index: int = 0


_SENTENCE_RE = re.compile(r"(?<=[.!?।॥！？])\s+|\n+")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _sentences(text: str) -> list[str]:
    return [s for s in (_clean(x) for x in _SENTENCE_RE.split(text)) if s]


def _windows(items: list[str], max_words: int, overlap: int) -> list[str]:
    if not items:
        return []
    result: list[str] = []
    start = 0
    while start < len(items):
        current: list[str] = []
        words = 0
        end = start
        while end < len(items):
            item_words = len(items[end].split())
            if current and words + item_words > max_words:
                break
            current.append(items[end])
            words += item_words
            end += 1
        result.append(" ".join(current))
        if end >= len(items):
            break
        start = max(start + 1, end - overlap)
    return result


def make_chunks(
    text: str,
    *,
    base_id: str,
    query_id: str | int | None,
    language: str,
    passage_index: int,
    is_selected: bool,
    strategy: str = "hybrid",
    max_words: int = 120,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    text = _clean(text)
    if not text:
        return []
    sentences = _sentences(text)
    if strategy == "passage":
        pieces = [text]
    elif strategy == "sentence":
        pieces = sentences
    elif strategy == "sliding":
        pieces = _windows(text.split(), max_words, max(1, max_words // 6))
    elif strategy == "hybrid":
        # Sentence-aware windows, with a sentence overlap to preserve context.
        pieces = _windows(sentences, max_words, overlap_sentences)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return [
        Chunk(
            chunk_id=f"{base_id}:{i}",
            text=piece,
            strategy=strategy,
            query_id=query_id,
            language=language,
            passage_index=passage_index,
            is_selected=is_selected,
            neighbor_index=i,
        )
        for i, piece in enumerate(pieces)
        if piece
    ]
