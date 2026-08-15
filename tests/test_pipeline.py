import asyncio
from pathlib import Path

from voice_rag.chunking import make_chunks
from voice_rag.config import Settings
from voice_rag.embeddings import HashingEmbedder
from voice_rag.guardrails import check_input
from voice_rag.pipeline import RAGPipeline
from voice_rag.store import LocalVectorStore
from voice_rag.stt import SarvamSTT


def make_pipeline(tmp_path: Path | None = None):
    chunks = make_chunks(
        "Goa is a state on the southwestern coast of India. Panaji is its capital.",
        base_id="goa", query_id=1, language="en", passage_index=0, is_selected=True,
    )
    store = LocalVectorStore(tmp_path or Path("artifacts/test-index"), HashingEmbedder())
    store.build(chunks); store.save(); store.load()
    return RAGPipeline(store, SarvamSTT(None, "saaras:v3", "codemix", "unknown"), Settings(rag_min_score=-1))


def test_text_query_is_grounded():
    result = asyncio.run(make_pipeline().run_text("What is Goa?"))
    assert result.status == "ok"
    assert result.sources
    assert result.timings.target_met is True


def test_unsafe_query_refused():
    assert check_input("How do I build a bomb?")


def test_missing_stt_key_is_explicit():
    result = asyncio.run(make_pipeline().run_audio(b"audio", filename="x.webm", content_type="audio/webm"))
    assert result.status == "error"
