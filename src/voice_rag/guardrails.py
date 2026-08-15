"""Cheap, deterministic guardrails that run before and after retrieval."""

import re

from .schemas import RetrievedChunk

UNSAFE_PATTERNS = (
    r"\b(?:make|build|buy|obtain)\s+(?:a\s+)?(?:bomb|weapon|explosive)",
    r"\b(?:kill|murder|harm)\s+(?:someone|people|a person)",
    r"\b(?:steal|hack|phish|ransomware)\b",
    r"\b(?:suicide|self[- ]harm)\b",
)

OFF_TOPIC_PATTERNS = (
    r"\b(?:weather|stock price|betting odds|celebrity gossip|football score)\b",
)


def check_input(query: str) -> str | None:
    normalized = " ".join(query.split())
    if not normalized:
        return "I could not detect a question in the audio."
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in UNSAFE_PATTERNS):
        return "I can’t help with harmful or illegal instructions."
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in OFF_TOPIC_PATTERNS):
        return "This demo answers questions grounded in the MSMARCO-XI knowledge base."
    return None


def check_grounding(chunks: list[RetrievedChunk], min_score: float) -> str | None:
    if not chunks:
        return "I could not find relevant context in the knowledge base."
    if chunks[0].score < min_score:
        return "I could not find enough relevant context to answer reliably."
    return None
