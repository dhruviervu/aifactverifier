FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install
COPY fact_verifier/requirements.txt /app/fact_verifier/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/fact_verifier/requirements.txt

# Copy source
COPY fact_verifier /app/fact_verifier

# Download spaCy model (wheel is listed, but ensure cached)
RUN python -m spacy download en_core_web_sm || true

EXPOSE 8000 8501

CMD bash -lc "\
  uvicorn fact_verifier.app.api:app --host 0.0.0.0 --port 8000 & \
  streamlit run fact_verifier/app/main_streamlit.py --server.port=8501 --server.address=0.0.0.0 \
"



