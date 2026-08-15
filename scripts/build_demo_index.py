import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_rag.chunking import Chunk, make_chunks
from voice_rag.embeddings import HashingEmbedder
from voice_rag.store import LocalVectorStore

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "demo_corpus.jsonl"
OUTPUT = Path(__file__).resolve().parents[1] / "artifacts" / "index"


def main():
    chunks: list[Chunk] = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        chunks.extend(
            make_chunks(
                row["text"],
                base_id=row["id"],
                query_id=row.get("query_id"),
                language=row.get("language", "en"),
                passage_index=row.get("passage_index", 0),
                is_selected=row.get("is_selected", True),
                strategy="hybrid",
                max_words=80,
            )
        )
    store = LocalVectorStore(OUTPUT, HashingEmbedder())
    store.build(chunks)
    store.save()
    print(f"built {len(chunks)} chunks at {OUTPUT}")


if __name__ == "__main__":
    main()
