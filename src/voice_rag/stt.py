"""Sarvam Saaras v3 REST adapter with bounded retries and safe errors."""

import asyncio
import io
import time

import httpx

from .errors import ProviderError


class SarvamSTT:
    endpoint = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, api_key: str | None, model: str, mode: str, language_code: str, timeout: float = 12.0):
        self.api_key = api_key
        self.model = model
        self.mode = mode
        self.language_code = language_code
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def transcribe(self, audio: bytes, *, filename: str = "recording.webm", content_type: str = "audio/webm") -> tuple[str, float]:
        if not self.api_key:
            raise ProviderError("SARVAM_API_KEY is not configured")
        started = time.perf_counter()
        data = {
            "model": self.model,
            "mode": self.mode,
            "language_code": self.language_code,
        }
        files = {"file": (filename, io.BytesIO(audio), content_type)}
        last_error = "unknown provider error"
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.endpoint,
                        headers={"api-subscription-key": self.api_key},
                        data=data,
                        files=files,
                    )
                if response.status_code == 200:
                    payload = response.json()
                    transcript = str(payload.get("transcript") or "").strip()
                    if transcript:
                        return transcript, (time.perf_counter() - started) * 1000
                    last_error = "Sarvam returned an empty transcript"
                else:
                    last_error = f"Sarvam HTTP {response.status_code}: {response.text[:200]}"
                    if response.status_code not in {429, 500, 502, 503, 504}:
                        break
            except (httpx.HTTPError, ValueError) as exc:
                last_error = str(exc)
            if attempt < 2:
                await asyncio.sleep(0.15 * (2**attempt))
        raise ProviderError(last_error)
