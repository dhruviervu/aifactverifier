"""Hard matching scores for entities, numbers, and keywords."""

from __future__ import annotations

from typing import Dict, List

import math
import re


def entity_match_score(claim_entities: List[Dict], evidence_text: str) -> float:
    if not claim_entities:
        return 1.0
    matched = 0
    ev_low = evidence_text.lower()
    for e in claim_entities:
        if e.get("label") in {"PERSON", "GPE", "ORG"}:
            if e.get("text", "").lower() in ev_low:
                matched += 1
    total = sum(1 for e in claim_entities if e.get("label") in {"PERSON", "GPE", "ORG"})
    return (matched / total) if total else 1.0


def _numeric_match_value(claim_val: float, ev_val: float) -> float:
    if claim_val == ev_val:
        return 1.0
    if claim_val == 0:
        return 0.0
    rel = abs(ev_val - claim_val) / abs(claim_val)
    return 0.9 if rel <= 0.05 else 0.0


def numeric_match_score(claim_numbers: List[Dict], evidence_numbers: List[Dict]) -> float:
    if not claim_numbers:
        return 1.0
    if not evidence_numbers:
        return 0.0
    best = 0.0
    for c in claim_numbers:
        cval = c.get("value")
        for e in evidence_numbers:
            eval_ = e.get("value")
            best = max(best, _numeric_match_value(float(cval), float(eval_)))
    return best


def keyword_coverage_score(claim_text: str, evidence_text: str) -> float:
    claim_tokens = set(w.lower() for w in re.findall(r"[A-Za-z0-9]+", claim_text) if len(w) > 2)
    ev_tokens = set(re.findall(r"[A-Za-z0-9]+", evidence_text.lower()))
    if not claim_tokens:
        return 1.0
    covered = len(claim_tokens & ev_tokens)
    return covered / len(claim_tokens)


def hard_score(claim_entities: List[Dict], claim_numbers: List[Dict], claim_text: str, evidence_text: str, evidence_numbers: List[Dict]) -> float:
    e = entity_match_score(claim_entities, evidence_text)
    n = numeric_match_score(claim_numbers, evidence_numbers)
    k = keyword_coverage_score(claim_text, evidence_text)
    return 0.45 * e + 0.30 * n + 0.25 * k


__all__ = [
    "entity_match_score",
    "numeric_match_score",
    "keyword_coverage_score",
    "hard_score",
]



