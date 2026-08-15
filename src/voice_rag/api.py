from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .pipeline import RAGPipeline
from .schemas import HealthResponse, RAGResponse, TextQuery
from .store import LocalVectorStore
from .stt import SarvamSTT

settings = get_settings()
embedder = (
    SentenceTransformerEmbedder(settings.embedding_model)
    if settings.embedder == "sentence-transformers"
    else HashingEmbedder()
)
store = LocalVectorStore(settings.index_dir, embedder)
try:
    store.load()
except (FileNotFoundError, OSError, ValueError):
    pass
stt = SarvamSTT(
    settings.sarvam_api_key,
    settings.sarvam_model,
    settings.sarvam_mode,
    settings.sarvam_language_code,
    settings.request_timeout_seconds,
)
pipeline = RAGPipeline(store, stt, settings)

app = FastAPI(title="HH Goa 2026 Voice RAG", version="0.1.0")
web_dir = Path(__file__).resolve().parents[2] / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def index():
    return FileResponse(web_dir / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        index_loaded=store.loaded,
        answer_mode=pipeline.generator.provider,
        stt_configured=stt.configured,
    )


@app.post("/query", response_model=RAGResponse)
async def query(payload: TextQuery):
    return await pipeline.run_text(payload.query)


@app.post("/voice-query", response_model=RAGResponse)
async def voice_query(file: Annotated[UploadFile, File()]):
    audio = await file.read()
    return await pipeline.run_audio(
        audio,
        filename=file.filename or "recording.webm",
        content_type=file.content_type or "audio/webm",
    )
