from voice_rag.chunking import make_chunks


def test_hybrid_preserves_sentence_boundaries_and_overlap():
    chunks = make_chunks(
        "First sentence. Second sentence. Third sentence. Fourth sentence.",
        base_id="x", query_id=1, language="en", passage_index=0, is_selected=True,
        strategy="hybrid", max_words=3, overlap_sentences=1,
    )
    assert len(chunks) >= 2
    assert all(chunk.text for chunk in chunks)
    assert chunks[0].strategy == "hybrid"


def test_sliding_handles_multilingual_punctuation():
    chunks = make_chunks(
        "यह पहला वाक्य है। यह दूसरा वाक्य है।",
        base_id="x", query_id=1, language="hi-IN", passage_index=0, is_selected=False,
        strategy="sentence",
    )
    assert len(chunks) == 2
