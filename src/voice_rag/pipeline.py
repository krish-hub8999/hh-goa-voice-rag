"""Structured orchestration for text and voice queries."""

import time
import uuid

from .errors import IndexNotLoadedError, ProviderError
from .generation import ExtractiveGenerator, OpenAICompatibleGenerator
from .guardrails import check_grounding, check_input
from .schemas import RAGResponse, Timing
from .store import LocalVectorStore
from .stt import SarvamSTT


class RAGPipeline:
    def __init__(self, store: LocalVectorStore, stt: SarvamSTT, settings):
        self.store = store
        self.stt = stt
        if settings.answer_mode == "llm" and settings.llm_api_key:
            self.generator = OpenAICompatibleGenerator(
                settings.llm_base_url, settings.llm_api_key, settings.llm_model
            )
        else:
            self.generator = ExtractiveGenerator()
        self.top_k = settings.rag_top_k
        self.min_score = settings.rag_min_score

    async def run_text(self, query: str, *, request_id: str | None = None, transcript: str | None = None, stt_ms: float = 0.0, provider: str = "text") -> RAGResponse:
        started = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())
        guardrail_started = time.perf_counter()
        refusal = check_input(query)
        guardrail_ms = (time.perf_counter() - guardrail_started) * 1000
        if refusal:
            return self._response(request_id, "refused", refusal, [], started, stt_ms, guardrail_ms, provider, refusal)
        if not self.store.loaded:
            raise IndexNotLoadedError("Build an index before querying the API")

        # The store records query-embedding time separately from retrieval time.
        retrieval_started = time.perf_counter()
        retrieved = self.store.search(query, top_k=self.top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        grounding_refusal = check_grounding(retrieved, self.min_score)
        if grounding_refusal:
            return self._response(request_id, "refused", grounding_refusal, retrieved, started, stt_ms, guardrail_ms, provider, grounding_refusal, retrieval_ms=retrieval_ms, embed_ms=self.store.last_embed_ms)
        chunks = [chunk for chunk in retrieved if chunk.score >= self.min_score] or retrieved[:1]

        generation_started = time.perf_counter()
        try:
            answer, _citations = await self.generator.answer(query, chunks)
        except ProviderError:
            # Recovery policy: do not fail closed when the optional LLM is down;
            # return a fully grounded extractive answer instead.
            answer, _citations = await ExtractiveGenerator().answer(query, chunks)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        return RAGResponse(
            request_id=request_id,
            status="ok",
            transcript=transcript,
            answer=answer,
            sources=chunks,
            timings=Timing(
                total_ms=round(total_ms, 3),
                stt_ms=round(stt_ms, 3),
                guardrail_ms=round(guardrail_ms, 3),
                embed_ms=round(self.store.last_embed_ms, 3),
                retrieval_ms=round(retrieval_ms, 3),
                generation_ms=round(generation_ms, 3),
                target_met=total_ms < 200,
            ),
            provider=provider,
        )

    async def run_audio(self, audio: bytes, *, filename: str, content_type: str) -> RAGResponse:
        request_id = str(uuid.uuid4())
        try:
            transcript, stt_ms = await self.stt.transcribe(audio, filename=filename, content_type=content_type)
        except ProviderError as exc:
            return RAGResponse(
                request_id=request_id,
                status="error",
                answer="Voice transcription is unavailable right now. You can use the text box instead.",
                refusal_reason=str(exc),
                timings=Timing(total_ms=0, stt_ms=0, target_met=False),
                provider="sarvam",
            )
        return await self.run_text(transcript, request_id=request_id, transcript=transcript, stt_ms=stt_ms, provider="sarvam")

    @staticmethod
    def _response(request_id, status, answer, chunks, started, stt_ms, guardrail_ms, provider, reason, retrieval_ms=0.0, embed_ms=0.0):
        total_ms = (time.perf_counter() - started) * 1000
        return RAGResponse(
            request_id=request_id,
            status=status,
            answer=answer,
            refusal_reason=reason,
            sources=chunks,
            timings=Timing(
                total_ms=round(total_ms, 3), stt_ms=round(stt_ms, 3),
                guardrail_ms=round(guardrail_ms, 3), retrieval_ms=round(retrieval_ms, 3),
                embed_ms=round(embed_ms, 3),
                target_met=total_ms < 200,
            ),
            provider=provider,
        )
