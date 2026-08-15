"""Benchmark warm local text->retrieval->extractive answer latency."""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_rag.config import Settings
from voice_rag.embeddings import HashingEmbedder
from voice_rag.pipeline import RAGPipeline
from voice_rag.store import LocalVectorStore
from voice_rag.stt import SarvamSTT


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/index")
    parser.add_argument("--queries", default="data/benchmark_queries.jsonl")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    store = LocalVectorStore(Path(args.index), HashingEmbedder())
    store.load()
    settings = Settings(answer_mode="extractive", rag_min_score=-1)
    pipeline = RAGPipeline(store, SarvamSTT(None, "saaras:v3", "codemix", "unknown"), settings)
    queries = [json.loads(line)["query"] for line in Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    timings = []
    for query in queries[: max(1, args.repeats)]:
        for _ in range(args.repeats):
            started = time.perf_counter()
            await pipeline.run_text(query)
            timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    def percentile(p):
        index = min(len(timings) - 1, round((p / 100) * (len(timings) - 1)))
        return round(timings[index], 3)
    result = {
        "scope": "text -> guardrails -> embedding -> local vector retrieval -> extractive answer",
        "queries": len(queries),
        "samples": len(timings),
        "warmup": "none; first request is included, repeat the command for steady-state numbers",
        "target_ms": 200,
        "p50_ms": percentile(50),
        "p70_ms": percentile(70),
        "p100_ms": round(max(timings), 3),
        "mean_ms": round(statistics.mean(timings), 3),
        "measured_on": "local machine at benchmark runtime",
        "note": "Sarvam network STT and optional hosted LLM generation are excluded from this local target.",
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
