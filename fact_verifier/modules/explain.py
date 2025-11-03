"""Explainability utilities: highlight evidence and produce justification."""

from __future__ import annotations

from typing import Dict, List

import html
import re


def _highlight(text: str, terms: List[str]) -> str:
    escaped = [re.escape(t) for t in terms if t]
    if not escaped:
        return html.escape(text)
    pattern = re.compile(r"(" + "|".join(escaped) + r")", re.I)
    return pattern.sub(r"<mark>\\1</mark>", html.escape(text))


def build_explanation(claim_text: str, evidence_text: str, entities: List[Dict], numbers: List[Dict]) -> Dict[str, str]:
    entity_terms = [e.get("text") for e in entities]
    number_terms = [str(n.get("value")) for n in numbers]
    keywords = list({*entity_terms, *number_terms})
    highlighted = _highlight(evidence_text, [t for t in keywords if t])
    explanation = (
        f"Entity matched: {', '.join([e for e in entity_terms if e])}; "
        f"Numbers referenced: {', '.join([n for n in number_terms if n])}."
    )
    return {"highlighted_evidence": highlighted, "explanation": explanation}


__all__ = ["build_explanation"]



