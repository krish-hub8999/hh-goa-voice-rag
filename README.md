# HH Goa 2026 Task 2 — Voice-enabled RAG

An end-to-end, multilingual retrieval-augmented QA demo over [AI4Bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI):

```text
browser microphone → Sarvam Saaras v3 STT → guardrails → query embedding
→ FAISS/NumPy local vector retrieval → lexical rerank → grounded answer + citations
```

The default answer mode is local extractive generation. That makes the retrieval path measurable and keeps the demo usable without a second hosted model key. An optional OpenAI-compatible JSON generator is available through environment variables, with citation validation and extractive fallback if the provider fails.

## Run the demo in five minutes

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts\build_demo_index.py
python -m uvicorn voice_rag.api:app --reload
```

Open http://127.0.0.1:8000. The included fixture makes the UI runnable immediately; it is only a smoke-test corpus, not the competition dataset.

## Build an MSMARCO-XI index

Install the dataset extra, then stream a bounded slice while iterating on the demo:

```powershell
pip install -e ".[data,quality]"
voice-rag build-index --language hi --split train --max-examples 5000 `
  --strategy hybrid --embedder sentence-transformers
```

If you use the quality embedder, set `EMBEDDER=sentence-transformers` and the same `EMBEDDING_MODEL` in `.env` before starting the API. The demo fixture and default API use the dependency-light hash embedder so they start without downloading a model.

The dataset card documents language-specific configs and fields such as `query`, `Answer`, `query_id`, and `passages.Translated_passages`; the loader uses translated passages for Indic configs and English passages when `--language en` is selected. The builder indexes passages only, retaining query/language/passage metadata for citations and analysis.

Available chunking strategies:

- `passage`: preserve each source passage as one unit for short passages.
- `sentence`: sentence-level units using multilingual punctuation, including Devanagari danda.
- `sliding`: word windows with overlap for noisy/long passages.
- `hybrid` (default): sentence-aware windows, passage boundaries, and one-sentence overlap when a passage is long.

This is deliberately an index-time choice rather than one hard-coded splitter. Compare strategies by building separate output directories and running the same benchmark.

## Voice configuration

```powershell
Copy-Item .env.example .env
# Set SARVAM_API_KEY in .env
```

The adapter uses Sarvam’s current `POST https://api.sarvam.ai/speech-to-text` multipart endpoint, `saaras:v3`, and `codemix` by default. Change `SARVAM_MODE` to `transcribe`, `translate`, `verbatim`, or `translit` when appropriate. Sarvam REST is intended for short interactive audio; the browser recording path sends WebM audio.

The browser’s “Read answer aloud” button uses the local Web Speech API, so server-side TTS is not required for the demo.

## Latency and benchmark protocol

The hard target is reported for the local text path:

```text
guardrails → embedding → vector search → rerank → extractive answer
```

It excludes the network round trip to Sarvam and excludes an optional hosted LLM. Those external calls are separately reported in each response’s `stt_ms` and `generation_ms` fields. A hosted LLM cannot honestly be claimed to fit a 200 ms end-to-end budget without measuring the deployed region/provider.

Run the benchmark after building the demo or a real index:

```powershell
python scripts\benchmark.py --index artifacts\index --repeats 10
```

It writes `artifacts/benchmark.json` with P50, P70, P100, mean, sample count, target, and scope. P100 here is the maximum observed sample, not a statistical percentile estimate.

Example fixture output from this repository:

```json
{
  "scope": "text -> guardrails -> embedding -> local vector retrieval -> extractive answer",
  "target_ms": 200,
  "p50_ms": "measured by running scripts/benchmark.py",
  "p70_ms": "measured by running scripts/benchmark.py",
  "p100_ms": "measured by running scripts/benchmark.py",
  "note": "Sarvam network STT and optional hosted LLM generation are excluded from this local target."
}
```

Do not replace these values with a synthetic claim. Commit the generated `artifacts/benchmark.json` from the actual machine/environment used in the submission.

## Harness and guardrails

- Structured Pydantic request/response objects with request IDs and per-stage timings.
- Provider retries for transient Sarvam errors (429/5xx) and bounded timeouts.
- Optional LLM retries, strict JSON parsing, and citation-ID validation.
- Safe extractive fallback if optional generation fails.
- Input refusals for unsafe/illegal requests and obvious off-topic requests.
- Grounding refusal when retrieval is empty or the top score is below `RAG_MIN_SCORE`.
- Every successful answer includes the retrieved chunk IDs and source text.

These are intentionally deterministic, reviewable controls. They are not a substitute for a full content-safety service in a production deployment.

## API

- `GET /health`
- `POST /query` with `{ "query": "..." }`
- `POST /voice-query` with multipart field `file`
- `GET /docs` for the generated OpenAPI view

## Repository layout

```text
src/voice_rag/       application, ingestion, chunking, retrieval, adapters
scripts/             demo index builder and latency benchmark
data/                tiny runnable fixture and benchmark questions
web/                 live browser demo
tests/               chunking, guardrail, retrieval, and STT-failure tests
```

## Submission/demo checklist

1. Build an index from the selected MSMARCO-XI language and record its manifest.
2. Run the benchmark on the submission machine and commit `artifacts/benchmark.json`.
3. Set `SARVAM_API_KEY`, deploy the FastAPI container, and test `/health`.
4. Record the two requested videos: architecture/latency walkthrough and live voice demo, including one refused off-topic/unsafe query.
5. Link the GitHub repo, live URL, benchmark output, and videos in the submission form.

## Current official references

- [MSMARCO-XI dataset card](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
- [Sarvam Speech-to-Text REST reference](https://docs.sarvam.ai/api-reference/speech-to-text/transcribe)
- [Sarvam Speech-to-Text overview](https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/overview)
- [ElevenLabs STT reference](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) — not used by this implementation, retained as the alternative allowed provider.
