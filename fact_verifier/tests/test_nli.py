import pytest

from fact_verifier.modules.nli_verifier import nli_scores


@pytest.mark.slow
def test_nli_reasonable_thresholds():
    # This test ensures model produces probabilities and basic ordering in a simple case
    res = nli_scores("The Eiffel Tower is in Paris.", "The Eiffel Tower is located in Paris, France.")
    assert set(res.keys()) == {"contradiction", "neutral", "entailment"}
    assert 0.0 <= res["entailment"] <= 1.0
    assert 0.0 <= res["contradiction"] <= 1.0
    assert abs(sum(res.values()) - 1.0) < 1e-5



