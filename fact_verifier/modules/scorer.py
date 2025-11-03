"""Final scoring and verdict rules."""

from __future__ import annotations

from typing import Dict


def compute_candidate_score(similarity: float, hard_score: float, entailment_prob: float, source_credibility: float) -> float:
    return 0.25 * similarity + 0.45 * hard_score + 0.20 * entailment_prob + 0.10 * source_credibility


def decide_verdict(entailment_prob: float, contradiction_prob: float, hard_score: float) -> str:
    if contradiction_prob >= 0.60 and (contradiction_prob - entailment_prob) >= 0.30:
        return "Refuted"
    if entailment_prob >= 0.65 and hard_score >= 0.60:
        return "Supported"
    return "Not Enough Evidence"


__all__ = ["compute_candidate_score", "decide_verdict"]



