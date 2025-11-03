"""Claim extraction module.

This module converts paragraph text into a list of atomic factual claims.
It uses spaCy to split sentences and heuristics to split multi-fact
sentences into atomic claims while filtering subjective phrases.
"""

from __future__ import annotations

from typing import Any, Dict, List

import re
import spacy


_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback: try to download dynamically if not present
            from spacy.cli import download

            download("en_core_web_sm")
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


OPINION_PREFIXES = [
    "i think",
    "i believe",
    "maybe",
    "perhaps",
    "it seems",
    "it appears",
    "i guess",
    "in my opinion",
]


def _is_opinion(sentence_text: str) -> bool:
    normalized = sentence_text.strip().lower()
    return any(normalized.startswith(p) for p in OPINION_PREFIXES)


def _split_by_conjunctions(span_text: str) -> List[str]:
    """Split sentence by coordinating conjunctions to approach atomicity.

    Heuristics:
    - Split on top-level "and", "but", ", and", "; and" where suitable
    - Avoid splitting inside parentheses
    - Keep segments non-empty and with at least one verb-like token later
    """
    # Light-weight approach using regex; spaCy-based could be heavier here
    # Protect parentheses by temporarily replacing content
    protected: List[str] = []
    placeholders: Dict[str, str] = {}

    def protect(match):
        key = f"__PAREN_{len(protected)}__"
        protected.append(match.group(0))
        placeholders[key] = match.group(0)
        return key

    temp = re.sub(r"\([^\)]*\)", protect, span_text)
    # Split on conjunctions; keep commas + and/but
    parts = re.split(r"\b(?:,\s*)?(?:and|but)\b", temp, flags=re.IGNORECASE)
    restored = []
    for part in parts:
        for key, value in placeholders.items():
            part = part.replace(key, value)
        cleaned = part.strip(" ,;:" )
        if cleaned:
            restored.append(cleaned)
    return restored


def _has_predicate_like(doc: "spacy.tokens.Doc") -> bool:
    return any(t.pos_ in {"VERB", "AUX"} for t in doc)


def _make_atomic_claims_from_sentence(sentence_text: str) -> List[str]:
    # Initial split by conjunctions
    segments = _split_by_conjunctions(sentence_text)
    nlp = _get_nlp()
    atomic: List[str] = []
    for seg in segments:
        seg_doc = nlp(seg)
        # If still contains multiple predicates, attempt clause-level split via punctuation
        verb_count = sum(1 for t in seg_doc if t.pos_ in {"VERB", "AUX"})
        if verb_count > 1:
            subparts = re.split(r"[;:]+", seg)
            for sub in subparts:
                sub_doc = nlp(sub)
                if _has_predicate_like(sub_doc):
                    atomic.append(sub.strip())
        else:
            if _has_predicate_like(seg_doc):
                atomic.append(seg.strip())
    return atomic


def extract_claims(text: str) -> List[Dict[str, Any]]:
    """Extract atomic factual claims from free-form text.

    Parameters
    ----------
    text: str
        Input paragraph(s).

    Returns
    -------
    List[Dict[str, Any]]
        Items with keys: text, original_sentence, confidence
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp()
    doc = nlp(text)

    results: List[Dict[str, Any]] = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        if _is_opinion(sent_text):
            continue

        atomic_segments = _make_atomic_claims_from_sentence(sent_text)
        # Basic confidence heuristic: fewer splits -> higher confidence
        base_conf = 0.9 if len(atomic_segments) <= 1 else max(0.6, 0.9 - 0.1 * (len(atomic_segments) - 1))

        for seg in atomic_segments:
            if len(seg) < 4:
                continue
            # Light heuristic to ensure factual pattern (contains a content word)
            if not re.search(r"[A-Za-z]\w+", seg):
                continue
            results.append(
                {
                    "text": seg,
                    "original_sentence": sent_text,
                    "confidence": round(base_conf, 2),
                }
            )

    return results


__all__ = ["extract_claims"]



