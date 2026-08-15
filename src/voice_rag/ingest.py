"""MSMARCO-XI dataset normalization and streaming ingestion."""

from collections.abc import Iterable
from typing import Any

from .chunking import Chunk, make_chunks


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def extract_passages(example: dict[str, Any], language: str = "auto") -> list[tuple[str, int, bool]]:
    passages = example.get("passages") or {}
    if not isinstance(passages, dict):
        passages = {}
    translated = _as_list(passages.get("Translated_passages"))
    english = _as_list(passages.get("English_passages"))
    selected = _as_list(passages.get("is_selected"))
    chosen = english if language == "en" else translated or english
    return [
        (str(text), i, bool(selected[i]) if i < len(selected) else False)
        for i, text in enumerate(chosen)
        if str(text or "").strip()
    ]


def example_to_chunks(
    example: dict[str, Any],
    *,
    language: str = "auto",
    strategy: str = "hybrid",
    max_words: int = 120,
) -> list[Chunk]:
    query_id = example.get("query_id")
    actual_language = str(example.get("target_lang") or language)
    chunks: list[Chunk] = []
    for text, passage_index, is_selected in extract_passages(example, language):
        base_id = f"{query_id or 'row'}-{passage_index}"
        chunks.extend(
            make_chunks(
                text,
                base_id=base_id,
                query_id=query_id,
                language=actual_language,
                passage_index=passage_index,
                is_selected=is_selected,
                strategy=strategy,
                max_words=max_words,
            )
        )
    return chunks


def stream_msmarco(
    *,
    language: str,
    split: str = "train",
    max_examples: int | None = None,
    cache_dir: str | None = None,
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional data extra: pip install -e '.[data]'") from exc
    dataset = load_dataset(
        "ai4bharat/MSMARCO-XI",
        language,
        split=split,
        streaming=True,
        cache_dir=cache_dir,
    )
    for i, example in enumerate(dataset):
        if max_examples is not None and i >= max_examples:
            break
        yield dict(example)
