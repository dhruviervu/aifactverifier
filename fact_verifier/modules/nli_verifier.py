"""NLI verifier using facebook/bart-large-mnli."""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


_MODEL = None
_TOKENIZER = None


def _get_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        name = "facebook/bart-large-mnli"
        _TOKENIZER = AutoTokenizer.from_pretrained(name)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(name)
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def nli_scores(premise: str, hypothesis: str) -> Dict[str, float]:
    tokenizer, model = _get_model()
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    # BART MNLI labels: contradiction, neutral, entailment
    return {
        "contradiction": float(probs[0]),
        "neutral": float(probs[1]),
        "entailment": float(probs[2]),
    }


__all__ = ["nli_scores"]



