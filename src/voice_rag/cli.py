import argparse
import json
from pathlib import Path

from .chunking import Chunk
from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .ingest import example_to_chunks, stream_msmarco
from .store import LocalVectorStore


def build_index(args: argparse.Namespace) -> None:
    chunks: list[Chunk] = []
    for example in stream_msmarco(
        language=args.language,
        split=args.split,
        max_examples=args.max_examples,
        cache_dir=args.cache_dir,
    ):
        chunks.extend(
            example_to_chunks(
                example,
                language=args.language,
                strategy=args.strategy,
                max_words=args.max_words,
            )
        )
    if not chunks:
        raise SystemExit("No chunks produced. Check the dataset language/configuration.")
    embedder = (
        SentenceTransformerEmbedder(args.model)
        if args.embedder == "sentence-transformers"
        else HashingEmbedder()
    )
    store = LocalVectorStore(Path(args.output), embedder)
    store.build(chunks)
    store.save()
    manifest = {
        "dataset": "ai4bharat/MSMARCO-XI",
        "language": args.language,
        "split": args.split,
        "examples_limit": args.max_examples,
        "strategy": args.strategy,
        "chunk_count": len(chunks),
        "embedder": args.embedder,
    }
    Path(args.output).joinpath("build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="voice-rag")
    sub = parser.add_subparsers(required=True)
    build = sub.add_parser("build-index")
    build.add_argument("--language", default="hi", help="MSMARCO-XI config, e.g. hi, en, ta")
    build.add_argument("--split", default="train")
    build.add_argument("--max-examples", type=int, default=5000)
    build.add_argument("--strategy", choices=["hybrid", "sentence", "sliding", "passage"], default="hybrid")
    build.add_argument("--max-words", type=int, default=120)
    build.add_argument("--embedder", choices=["hash", "sentence-transformers"], default="hash")
    build.add_argument("--model", default="intfloat/multilingual-e5-small")
    build.add_argument("--cache-dir")
    build.add_argument("--output", default="artifacts/index")
    build.set_defaults(func=build_index)
    args = parser.parse_args()
    args.func(args)
