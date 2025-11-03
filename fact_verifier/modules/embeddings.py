"""Embeddings and similarity utilities."""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer, util


_MODEL = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _MODEL


def embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def cosine_similarities(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return util.cos_sim(query_vec, matrix).cpu().numpy().ravel()


__all__ = ["embed_texts", "cosine_similarities"]



