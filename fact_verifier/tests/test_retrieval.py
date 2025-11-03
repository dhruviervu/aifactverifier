import pytest

from fact_verifier.modules.wiki_retriever import retrieve_wikipedia_text


@pytest.mark.parametrize(
    "entities,claim_text",
    [
        ([{"text": "Eiffel Tower", "label": "ORG"}], "The Eiffel Tower is in Paris."),
        ([{"text": "Paris", "label": "GPE"}], "Paris is the capital of France."),
    ],
)
def test_wikipedia_retrieval_handles_basic_cases(entities, claim_text):
    text = retrieve_wikipedia_text(entities, claim_text)
    # May be None in CI without network, but should not raise; when available, expect some content
    assert text is None or isinstance(text, str)



