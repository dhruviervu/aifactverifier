"""FastAPI for Fact Verifier."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure project root is on sys.path when executed from arbitrary CWD
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fact_verifier.modules.claim_extractor import extract_claims
from fact_verifier.modules.entity_numeric_parser import parse_entities_and_numbers
from fact_verifier.modules.wiki_retriever import retrieve_wikipedia_text
from fact_verifier.modules.web_retriever import web_search_fallback
from fact_verifier.modules.chunker import sentence_split, make_overlapping_chunks, retain_relevant_chunks
from fact_verifier.modules.embeddings import embed_texts, cosine_similarities
from fact_verifier.modules.hard_matcher import hard_score
from fact_verifier.modules.nli_verifier import nli_scores
from fact_verifier.modules.scorer import compute_candidate_score, decide_verdict
from fact_verifier.modules.explain import build_explanation

import asyncio


class VerifyRequest(BaseModel):
    text: str


class ClaimOut(BaseModel):
    text: str
    original_sentence: str
    confidence: float


class EvidenceOut(BaseModel):
    text: str
    similarity: float
    hard_score: float
    entailment: float
    contradiction: float
    source: str
    candidate_score: float


class ClaimVerdictOut(ClaimOut):
    verdict: str
    final_score: float
    best_evidence_text: Optional[str] = None
    highlighted_evidence: Optional[str] = None
    explanation: Optional[str] = None
    evidence_candidates: List[EvidenceOut] = []


class VerifyResponse(BaseModel):
    claims: List[ClaimVerdictOut]


app = FastAPI(title="Fact Verifier API", version="0.1.0")


def _gather_chunks_for_claim(claim_text: str, entity_hint: Optional[str], wiki_text: Optional[str], web_texts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Return list of (chunk_text, source_label)."""
    chunks: List[Tuple[str, str]] = []
    if wiki_text:
        sents = sentence_split(wiki_text)
        cks = make_overlapping_chunks(sents, window=2, overlap=1)
        rel = retain_relevant_chunks(cks, claim_text, entity_hint)
        chunks.extend([(r, "wikipedia") for r in rel])
    for url, text in web_texts:
        sents = sentence_split(text)
        cks = make_overlapping_chunks(sents, window=2, overlap=1)
        rel = retain_relevant_chunks(cks, claim_text, entity_hint)
        chunks.extend([(r, "web") for r in rel])
    return chunks


def run_verification(text: str) -> List[Dict[str, Any]]:
    claims = extract_claims(text)
    results: List[Dict[str, Any]] = []
    if not claims:
        return results

    for c in claims:
        claim_text = c["text"]
        parsed_claim = parse_entities_and_numbers(claim_text)
        entities = parsed_claim["entities"]
        numbers = parsed_claim["numbers"]
        entity_hint = None
        for e in entities:
            if e.get("label") in {"GPE", "ORG", "PERSON"}:
                entity_hint = e.get("text")
                break

        wiki_text = retrieve_wikipedia_text(entities, claim_text)
        web_texts: List[Tuple[str, str]] = []
        if not wiki_text:
            try:
                web_results = asyncio.run(web_search_fallback(claim_text))
                for item in web_results:
                    web_texts.append((item.get("url", ""), item.get("text", "")))
            except Exception:
                web_texts = []

        chunks = _gather_chunks_for_claim(claim_text, entity_hint, wiki_text, web_texts)
        if not chunks:
            # No evidence
            results.append(
                {
                    **c,
                    "verdict": "Not Enough Evidence",
                    "final_score": 0.0,
                    "best_evidence_text": None,
                    "highlighted_evidence": None,
                    "explanation": "No relevant evidence retrieved.",
                    "evidence_candidates": [],
                }
            )
            continue

        claim_vec = embed_texts([claim_text])[0]
        chunk_texts = [ct for ct, _ in chunks]
        chunk_vecs = embed_texts(chunk_texts)
        sims = cosine_similarities(claim_vec, chunk_vecs)

        indexed = [
            (i, chunk_texts[i], chunks[i][1], float(sims[i])) for i in range(len(chunk_texts))
        ]
        indexed = [x for x in indexed if x[3] >= 0.30]
        indexed.sort(key=lambda x: x[3], reverse=True)
        top = indexed[:10]

        evidence_candidates: List[EvidenceOut] = []
        best: Tuple[float, EvidenceOut, Dict[str, float]] | None = None
        for _, ev_text, source, sim in top:
            ev_parsed = parse_entities_and_numbers(ev_text)
            hs = hard_score(entities, numbers, claim_text, ev_text, ev_parsed["numbers"])
            nli = nli_scores(ev_text, claim_text)
            src_cred = 0.9 if source == "wikipedia" else 0.6
            cand = compute_candidate_score(sim, hs, nli["entailment"], src_cred)
            ev_out = EvidenceOut(
                text=ev_text,
                similarity=round(sim, 3),
                hard_score=round(hs, 3),
                entailment=round(nli["entailment"], 3),
                contradiction=round(nli["contradiction"], 3),
                source=source,
                candidate_score=round(cand, 3),
            )
            evidence_candidates.append(ev_out)
            if best is None or cand > best[0]:
                best = (cand, ev_out, nli)

        if best is None:
            results.append(
                {
                    **c,
                    "verdict": "Not Enough Evidence",
                    "final_score": 0.0,
                    "best_evidence_text": None,
                    "highlighted_evidence": None,
                    "explanation": "No sufficiently similar evidence.",
                    "evidence_candidates": [e.model_dump() for e in evidence_candidates],
                }
            )
            continue

        best_cand_score, best_ev_out, best_nli = best
        verdict = decide_verdict(best_nli["entailment"], best_nli["contradiction"], best_ev_out.hard_score)
        exp = build_explanation(claim_text, best_ev_out.text, entities, numbers)
        results.append(
            {
                **c,
                "verdict": verdict,
                "final_score": round(float(best_cand_score), 3),
                "best_evidence_text": best_ev_out.text,
                "highlighted_evidence": exp["highlighted_evidence"],
                "explanation": exp["explanation"],
                "evidence_candidates": [e.model_dump() for e in evidence_candidates],
            }
        )

    return results


@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest) -> VerifyResponse:
    outputs = run_verification(req.text)
    claim_models = [ClaimVerdictOut(**o) for o in outputs]
    return VerifyResponse(claims=claim_models)


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"status": "ok", "message": "Fact Verifier API"}


