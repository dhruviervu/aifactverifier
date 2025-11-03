"""Wikipedia retriever.

Identifies main entity heuristic and fetches Wikipedia page text.
"""

from __future__ import annotations

from typing import Dict, Optional

import re
import wikipedia


def _select_entity(entities: list[dict], claim_text: str) -> Optional[str]:
    priority = ["GPE", "ORG", "PERSON"]
    for label in priority:
        for e in entities:
            if e.get("label") == label:
                return e.get("text")
    # fallback: first capitalized noun-like token span
    m = re.search(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b", claim_text)
    return m.group(1) if m else None


def retrieve_wikipedia_text(entities: list[dict], claim_text: str) -> Optional[str]:
    entity = _select_entity(entities, claim_text)
    if not entity:
        return None
    try:
        page_title = wikipedia.search(entity, results=1)
        if not page_title:
            return None
        page = wikipedia.page(page_title[0], auto_suggest=False)
        content = page.content or ""
        # drop infobox-like lines (naive)
        cleaned = re.sub(r"^\s*\{\{[\s\S]*?\}\}\s*$", "", content, flags=re.M)
        return cleaned.strip()
    except Exception:
        return None


__all__ = ["retrieve_wikipedia_text"]



