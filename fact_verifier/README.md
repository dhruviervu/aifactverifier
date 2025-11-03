# Production-Grade Fact Verifier

A reliable, explainable fact verification system with Streamlit UI and FastAPI API.

## Features
- Extracts atomic claims from paragraphs
- Retrieves evidence from Wikipedia with web fallback
- Sentence-level retrieval and ranking via embeddings
- Hard consistency checks (entities, numbers, keywords)
- Soft semantic checks (embedding similarity + NLI)
- Verdicts: Supported, Refuted, Not Enough Evidence
- Transparent justifications and score breakdowns

## Quickstart

### 1) Clone and setup
```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r fact_verifier/requirements.txt
python -m spacy download en_core_web_sm  # redundant if wheel installed
```

### 2) Run API and UI (local)
- API: `uvicorn fact_verifier.app.api:app --reload --port 8000`
- UI: `streamlit run fact_verifier/app/main_streamlit.py`

### 3) Docker
```bash
docker build -t fact-verifier -f fact_verifier/Dockerfile .
docker run -p 8000:8000 -p 8501:8501 fact-verifier
```
Then open UI at `http://localhost:8501` and API at `http://localhost:8000/docs`.

## Configuration
- Set `SERPER_API_KEY` or `BING_API_KEY` in environment for web fallback.

## Tests
```bash
pytest -q fact_verifier/tests
```

## Architecture
- `fact_verifier/modules/` contains pipeline components:
  - `claim_extractor.py`: sentence/claim splitting
  - `entity_numeric_parser.py`: NER + number normalization
  - `wiki_retriever.py`: Wikipedia retrieval
  - `web_retriever.py`: web fallback using Serper/Bing
  - `chunker.py`: sentence -> overlapping chunks
  - `embeddings.py`: embedding + cosine similarity
  - `hard_matcher.py`: entity/number/keyword hard checks
  - `nli_verifier.py`: NLI (BART MNLI)
  - `scorer.py`: final scoring + verdict
  - `explain.py`: highlights + explanation payload
  - `cache.py`: simple caching layer
- `fact_verifier/app/` contains UI and API entrypoints.

## Example
Input:
"The Eiffel Tower is located in Paris, France. It was completed in 1889 and stands 330 meters tall."

Expected claims and verdicts:
- Claim 1: Supported (location)
- Claim 2: Supported (year)
- Claim 3: Supported (height)

## Notes
- The first implementation favors robustness over perfect linguistic decomposition.
- You can swap retrievers or models by editing module configs.



