from typing import Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    strategy: str
    language: str
    query_id: str | int | None = None
    passage_index: int | None = None
    is_selected: bool = False


class Timing(BaseModel):
    total_ms: float
    stt_ms: float = 0.0
    guardrail_ms: float = 0.0
    embed_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    target_ms: float = 200.0
    target_met: bool


class RAGResponse(BaseModel):
    request_id: str
    status: Literal["ok", "refused", "error"]
    transcript: str | None = None
    answer: str
    refusal_reason: str | None = None
    sources: list[RetrievedChunk] = Field(default_factory=list)
    timings: Timing
    provider: str = "text"


class TextQuery(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    answer_mode: str
    stt_configured: bool
