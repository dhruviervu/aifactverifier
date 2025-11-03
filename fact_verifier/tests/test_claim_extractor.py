from fact_verifier.modules.claim_extractor import extract_claims


def test_claim_extraction_basic():
    text = (
        "The Eiffel Tower is located in Paris, France. "
        "It was completed in 1889 and stands 330 meters tall."
    )
    claims = extract_claims(text)
    assert isinstance(claims, list)
    assert any("Eiffel Tower is located in Paris" in c["text"] or "located in Paris" in c["text"] for c in claims)
    assert any("completed in 1889" in c["text"] for c in claims)
    assert any("330" in c["text"] and "meter" in c["text"].lower() for c in claims)


def test_filters_opinion():
    text = "I think the moon is made of cheese."
    claims = extract_claims(text)
    assert len(claims) == 0



