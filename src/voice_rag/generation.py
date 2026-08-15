"""Grounded answer generators: local extractive default + optional JSON LLM."""

import asyncio
import json
import re
from typing import Any

import httpx

from .errors import ProviderError
from .schemas import RetrievedChunk


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?।॥！？])\s+", text) if s.strip()]


class ExtractiveGenerator:
    provider = "extractive"

    async def answer(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
        query_terms = set(query.lower().split())
        candidates: list[tuple[float, str, str]] = []
        for chunk in chunks:
            for sentence in _sentences(chunk.text) or [chunk.text]:
                terms = set(sentence.lower().split())
                overlap = len(query_terms & terms) / max(1, len(query_terms))
                candidates.append((overlap + chunk.score, sentence, chunk.chunk_id))
        candidates.sort(reverse=True)
        chosen = []
        citations = []
        for _, sentence, chunk_id in candidates:
            if sentence not in chosen:
                chosen.append(sentence)
                if chunk_id not in citations:
                    citations.append(chunk_id)
            if len(chosen) == 2:
                break
        answer = " ".join(chosen) if chosen else chunks[0].text
        return f"{answer} [{', '.join(citations)}]", citations


class OpenAICompatibleGenerator:
    provider = "llm"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 8.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def answer(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
        context = "\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
        prompt = (
            "Answer only from the supplied context. If it is insufficient, say so. "
            "Return JSON with exactly answer and citations, where citations are chunk IDs.\n\n"
            f"Question: {query}\nContext:\n{context}"
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a grounded retrieval assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        last_error = "unknown generation error"
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                parsed: dict[str, Any] = json.loads(content)
                answer = str(parsed.get("answer") or "").strip()
                citations = [str(item) for item in parsed.get("citations", [])]
                valid_ids = {c.chunk_id for c in chunks}
                if answer and citations and set(citations).issubset(valid_ids):
                    return answer + f" [{', '.join(citations)}]", citations
                last_error = "LLM response failed citation validation"
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = str(exc)
            if attempt == 0:
                await asyncio.sleep(0.1)
        raise ProviderError(last_error)
