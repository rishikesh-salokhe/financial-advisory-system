"""
Streamlit entry point.

Run with:
    streamlit run dashboard/app.py

Individual feature pages live under ``dashboard/pages/`` and are auto-discovered
by Streamlit. This file is a lightweight landing screen + system health probe.
"""
from __future__ import annotations

# ─── sys.path bootstrap ───────────────────────────────────────────────────
# Streamlit only puts the script's own folder on sys.path, so absolute imports
# like ``from dashboard.X import Y`` fail out of the box. Add the project root
# (the parent of this file's directory) before any project imports.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────

import streamlit as st

from dashboard.components.api_client import APIError, get_client

st.set_page_config(
    page_title="Financial Advisory & Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Financial Advisory & Risk Intelligence")
st.caption("AI-powered forecasting, sentiment, risk profiling, and RAG-based research.")

st.markdown(
    """
    Use the sidebar to navigate between modules:

    * **📈 Forecasting** — ARIMA / LSTM price forecasts
    * **💬 Advisor Chat** — RAG-based Q&A over filings & news
    * **📰 Sentiment** — News-driven sentiment for a ticker
    * **⚖️ Risk Profile** — Questionnaire-based risk scoring
    * **💼 Portfolio** — Mean-variance optimization

    > ⚠️ This system is for research and educational use only. It is not investment advice.
    """
)

with st.expander("System status", expanded=False):
    client = get_client()
    try:
        health = client.health()
        cols = st.columns(4)
        cols[0].metric("Status", health["status"])
        cols[1].metric("Environment", health["environment"])
        cols[2].metric("Version", health["version"])
        cols[3].metric("MongoDB", "✅" if health["mongo_ok"] else "❌")
    except APIError as exc:
        st.error(f"Backend unreachable: {exc.message}")
