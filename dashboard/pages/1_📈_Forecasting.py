"""
Forecasting page — pulls a forecast from the FastAPI backend and renders a
Plotly chart with the historical close prices, the point forecast, and a
shaded confidence band.

This page is auto-discovered by Streamlit because it lives under
``dashboard/pages/``. The leading ``1_`` controls sidebar sort order; the
emoji controls the label.
"""
from __future__ import annotations

# ─── sys.path bootstrap ───────────────────────────────────────────────────
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.api_client import APIError, get_client
from ml_engine.data.yahoo_finance import fetch_close_series

st.set_page_config(page_title="Forecasting", page_icon="📈", layout="wide")

st.title("📈 Price Forecasting")
st.caption(
    "ARIMA-based price forecasts with confidence intervals. "
    "Train a model first with `python scripts/train_forecaster.py --ticker AAPL`."
)


# ─── Sidebar controls ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Forecast settings")
    ticker = st.text_input("Ticker", value="AAPL", max_chars=10).strip().upper()
    horizon_days = st.slider("Horizon (trading days)", 5, 90, 30, step=5)
    model_type = st.selectbox(
        "Model", options=["auto", "arima", "lstm"], index=0,
        help="'auto' picks the best available trained model.",
    )
    confidence = st.slider("Confidence interval", 0.50, 0.99, 0.95, step=0.01)
    history_days = st.slider("History to display (days)", 90, 1095, 365, step=30)
    run = st.button("Run Forecast", type="primary", use_container_width=True)


# ─── Main view ────────────────────────────────────────────────────────────

if not run:
    st.info("👈 Configure parameters in the sidebar and click **Run Forecast** to begin.")
    st.stop()

if not ticker:
    st.error("Please enter a ticker symbol.")
    st.stop()


# ─── 1. Call backend ──────────────────────────────────────────────────────

client = get_client()
with st.spinner(f"Forecasting {ticker}..."):
    try:
        payload = client.forecast({
            "ticker": ticker,
            "horizon_days": horizon_days,
            "model_type": model_type,
            "confidence_interval": confidence,
        })
    except APIError as exc:
        if exc.status_code == 503:
            st.warning(
                f"⚠️ No trained model found for **{ticker}**.\n\n"
                f"Train one by running this in a terminal:\n```\n"
                f"python scripts/train_forecaster.py --ticker {ticker}\n```"
            )
        else:
            st.error(f"Backend error ({exc.status_code}): {exc.message}")
        st.stop()


points = payload["points"]
if not points:
    st.warning("Forecast returned no points.")
    st.stop()

forecast_df = pd.DataFrame(points)
forecast_df["date"] = pd.to_datetime(forecast_df["date"])
forecast_df = forecast_df.set_index("date")


# ─── 2. Fetch history for the chart ───────────────────────────────────────

with st.spinner(f"Loading {history_days}d history..."):
    try:
        history = fetch_close_series(ticker, lookback_days=history_days)
    except Exception as exc:  # noqa: BLE001 — surface any yfinance error to the user
        st.error(f"Failed to fetch history for {ticker}: {exc}")
        st.stop()


# ─── 3. Plot ──────────────────────────────────────────────────────────────

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=history.index, y=history.values,
    mode="lines", name="History",
    line=dict(color="#1f77b4", width=2),
))

fig.add_trace(go.Scatter(
    x=forecast_df.index, y=forecast_df["predicted"],
    mode="lines", name=f"Forecast ({payload['model_type'].upper()})",
    line=dict(color="#ff7f0e", width=2, dash="dash"),
))

# Confidence band (upper trace first with fill='tonexty' on the lower).
if forecast_df["upper"].notna().any() and forecast_df["lower"].notna().any():
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["upper"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(255, 127, 14, 0.18)",
        name=f"{int(confidence * 100)}% CI",
        hoverinfo="skip",
    ))

fig.update_layout(
    title=f"{ticker} — {horizon_days}-day forecast",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    height=540,
    margin=dict(l=20, r=20, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)


# ─── 4. Metrics + tabular forecast ────────────────────────────────────────

st.subheader("Model metrics")
metrics = payload.get("metrics") or {}
if metrics:
    cols = st.columns(min(4, len(metrics)))
    for col, (label, value) in zip(cols, metrics.items()):
        try:
            col.metric(label.upper(), f"{float(value):.4f}")
        except (TypeError, ValueError):
            col.metric(label.upper(), str(value))
else:
    st.caption("No in-sample metrics returned by the backend.")

with st.expander("Forecast table"):
    display_df = forecast_df.copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    st.dataframe(
        display_df.style.format({"predicted": "{:.4f}", "lower": "{:.4f}", "upper": "{:.4f}"}),
        use_container_width=True,
    )

st.caption(
    f"Forecast generated at {payload['generated_at']} • "
    f"model: {payload['model_type']} • "
    f"ticker: {payload['ticker']}"
)
