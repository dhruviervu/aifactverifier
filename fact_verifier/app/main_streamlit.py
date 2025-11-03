"""Streamlit UI for the Fact Verifier."""

from __future__ import annotations

import json
import os
import sys
import streamlit as st

# Ensure project root is on sys.path when run from arbitrary CWD
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fact_verifier.app.api import run_verification


st.set_page_config(page_title="Fact Verifier", layout="wide")
st.title("AI Output Fact Verifier")

user_text = st.text_area("Enter text to verify", height=200)

if st.button("Verify"):
    with st.spinner("Running verification pipeline..."):
        results = run_verification(user_text)
    if not results:
        st.info("No claims detected or no evidence found.")
    else:
        for idx, r in enumerate(results, 1):
            verdict = r.get("verdict", "Not Enough Evidence")
            color = {
                "Supported": "#16a34a",
                "Refuted": "#dc2626",
                "Not Enough Evidence": "#ca8a04",
            }.get(verdict, "#4b5563")
            st.markdown(f"**Claim {idx}:** {r['text']}")
            st.markdown(f"<span style='color:{color};font-weight:700'>Verdict: {verdict}</span>", unsafe_allow_html=True)
            st.caption(f"Final score: {r.get('final_score', 0.0)} | Confidence: {r.get('confidence', 0.0)}")
            if r.get("highlighted_evidence"):
                st.markdown("**Best Evidence:**", unsafe_allow_html=True)
                st.markdown(r["highlighted_evidence"], unsafe_allow_html=True)
            if r.get("explanation"):
                st.caption(r["explanation"])
            with st.expander("Show Full JSON Trace"):
                st.code(json.dumps(r, indent=2))

st.caption("This is a scaffold. Full pipeline wiring will enrich the UI with verdicts and evidence.")


