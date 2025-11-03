"""Entity and numeric parser.

Extracts entities (PERSON, GPE, ORG, DATE, etc.) and normalizes numbers/units.
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
            from spacy.cli import download

            download("en_core_web_sm")
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


UNIT_ALIASES = {
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "m": "m",
    "kilometer": "km",
    "kilometers": "km",
    "km": "km",
    "foot": "ft",
    "feet": "ft",
    "ft": "ft",
}


def _normalize_number(token_text: str) -> float | None:
    text = token_text.lower()
    if text in NUMBER_WORDS:
        return float(NUMBER_WORDS[text])
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _extract_numbers_with_units(text: str) -> List[Dict[str, Any]]:
    pattern = re.compile(r"(\d+[\d,]*(?:\.\d+)?)\s*(km|kilometers|kilometer|m|meters|meter|ft|feet|foot)\b", re.I)
    results: List[Dict[str, Any]] = []
    for match in pattern.finditer(text):
        value = float(match.group(1).replace(",", ""))
        unit_raw = match.group(2).lower()
        unit = UNIT_ALIASES.get(unit_raw, unit_raw)
        results.append({"value": value, "unit": unit, "text": match.group(0)})
    return results


def parse_entities_and_numbers(text: str) -> Dict[str, Any]:
    nlp = _get_nlp()
    doc = nlp(text)

    ents: List[Dict[str, Any]] = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "GPE", "ORG", "DATE", "CARDINAL", "QUANTITY"}:
            ents.append({"text": ent.text, "label": ent.label_})

    numbers: List[Dict[str, Any]] = []
    for t in doc:
        val = _normalize_number(t.text)
        if val is not None:
            numbers.append({"value": val, "text": t.text})

    numbers_with_units = _extract_numbers_with_units(text)

    return {
        "entities": ents,
        "numbers": numbers,
        "numbers_with_units": numbers_with_units,
    }


__all__ = ["parse_entities_and_numbers"]



