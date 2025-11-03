"""Chunker: sentence-level splitting and overlapping chunks."""

from __future__ import annotations

from typing import List

import re


def sentence_split(text: str) -> List[str]:
    # Simple sentence split to avoid model dependency here
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def make_overlapping_chunks(sentences: List[str], window: int = 2, overlap: int = 1) -> List[str]:
    chunks: List[str] = []
    if window <= 0:
        return chunks
    step = max(1, window - overlap)
    for i in range(0, len(sentences), step):
        chunk = sentences[i : i + window]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


def retain_relevant_chunks(chunks: List[str], claim: str, entity_hint: str | None = None) -> List[str]:
    kept: List[str] = []
    claim_words = set(w.lower() for w in re.findall(r"\w+", claim))
    for ch in chunks:
        text_low = ch.lower()
        if entity_hint and entity_hint.lower() in text_low:
            kept.append(ch)
            continue
        shared = claim_words.intersection(re.findall(r"\w+", text_low))
        if len(shared) >= max(1, int(0.15 * len(claim_words))):
            kept.append(ch)
    return kept


__all__ = ["sentence_split", "make_overlapping_chunks", "retain_relevant_chunks"]



