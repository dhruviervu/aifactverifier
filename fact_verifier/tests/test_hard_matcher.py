from fact_verifier.modules.hard_matcher import (
    entity_match_score,
    numeric_match_score,
    keyword_coverage_score,
    hard_score,
)


def test_hard_matcher_scores():
    claim_entities = [{"text": "Eiffel Tower", "label": "ORG"}, {"text": "Paris", "label": "GPE"}]
    evidence_text = "The Eiffel Tower is an iron tower in Paris, France. It was completed in 1889."
    e = entity_match_score(claim_entities, evidence_text)
    assert 0.9 <= e <= 1.0

    claim_numbers = [{"value": 1889}]
    evidence_numbers = [{"value": 1889}]
    n = numeric_match_score(claim_numbers, evidence_numbers)
    assert n == 1.0

    k = keyword_coverage_score("Eiffel Tower completed in 1889", evidence_text)
    assert 0.4 <= k <= 1.0

    hs = hard_score(claim_entities, claim_numbers, "Eiffel Tower completed in 1889", evidence_text, evidence_numbers)
    assert 0.6 <= hs <= 1.0



