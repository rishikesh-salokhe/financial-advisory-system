"""
Sentiment page — pulls recent headlines for a ticker, scores them with FinBERT
on the backend, and renders:
    * three KPI tiles (bullish / neutral / bearish counts)
    * an overall-polarity gauge in [-1, +1]
    * a Plotly timeline (x = published time, y = polarity, color = label)
    * a clickable headlines table

Auto-discovered by Streamlit because it lives under ``dashboard/pages/``. The
leading ``2_`` controls sidebar order; the emoji controls the label.
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

st.set_page_config(page_title="Sentiment", page_icon="📰", layout="wide")

st.title("📰 News Sentiment")
st.caption(
    "FinBERT-scored sentiment on the latest Yahoo Finance headlines. "
    "First run downloads the ~440MB FinBERT model — subsequent runs are fast."
)


# ─── Sidebar controls ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Sentiment settings")
    ticker = st.text_input("Ticker", value="AAPL", max_chars=10).strip().upper()
    max_articles = st.slider("Max headlines", 5, 100, 25, step=5)
    run = st.button("Analyze", type="primary", use_container_width=True)


# ─── Main view ────────────────────────────────────────────────────────────

if not run:
    st.info("👈 Enter a ticker and click **Analyze** to begin.")
    st.stop()

if not ticker:
    st.error("Please enter a ticker symbol.")
    st.stop()


# ─── 1. Call backend ──────────────────────────────────────────────────────

client = get_client()
with st.spinner(f"Fetching headlines and scoring sentiment for {ticker}..."):
    try:
        payload = client.sentiment({"ticker": ticker, "max_articles": max_articles})
    except APIError as exc:
        st.error(f"Backend error ({exc.status_code}): {exc.message}")
        st.stop()


articles = payload.get("articles", [])
if not articles:
    st.warning(
        f"No recent headlines found for **{ticker}**. "
        f"Yahoo Finance occasionally returns empty news lists for lower-volume "
        f"tickers — try a major name like AAPL, MSFT, or TSLA."
    )
    st.stop()


# ─── 2. KPI tiles + overall gauge ─────────────────────────────────────────

distribution = payload.get("distribution", {})
overall_label = payload.get("overall_label", "neutral")
overall_score = float(payload.get("overall_score", 0.0))

col1, col2, col3, col4 = st.columns([1, 1, 1, 1.6])

col1.metric("Bullish 🟢", int(distribution.get("positive", 0)))
col2.metric("Neutral ⚪", int(distribution.get("neutral", 0)))
col3.metric("Bearish 🔴", int(distribution.get("negative", 0)))

# Overall sentiment gauge
gauge_color = (
    "#2ca02c" if overall_label == "positive"
    else ("#d62728" if overall_label == "negative" else "#7f7f7f")
)
gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=overall_score,
    number={"valueformat": ".2f"},
    title={"text": f"Overall: {overall_label.title()}", "font": {"size": 14}},
    gauge={
        "axis": {"range": [-1, 1], "tickwidth": 1},
        "bar": {"color": gauge_color},
        "steps": [
            {"range": [-1, -0.15], "color": "#fde7e7"},
            {"range": [-0.15, 0.15], "color": "#eeeeee"},
            {"range": [0.15, 1], "color": "#e7f5e7"},
        ],
        "threshold": {
            "line": {"color": "#333", "width": 3},
            "thickness": 0.8,
            "value": overall_score,
        },
    },
))
gauge.update_layout(height=170, margin=dict(l=10, r=10, t=30, b=10))
col4.plotly_chart(gauge, use_container_width=True)


# ─── 3. Sentiment timeline ────────────────────────────────────────────────

df = pd.DataFrame(articles)
if "published_at" in df.columns:
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.sort_values("published_at")

st.subheader("Sentiment over time")

label_colors = {"positive": "#2ca02c", "neutral": "#7f7f7f", "negative": "#d62728"}
fig = go.Figure()
for label, color in label_colors.items():
    sub = df[df["label"] == label] if "label" in df.columns else df.iloc[0:0]
    if sub.empty:
        continue
    fig.add_trace(go.Scatter(
        x=sub["published_at"], y=sub["score"],
        mode="markers",
        name=label.title(),
        marker=dict(size=11, color=color, line=dict(width=1, color="#222")),
        text=sub["headline"],
        hovertemplate="<b>%{text}</b><br>%{x|%Y-%m-%d %H:%M}<br>polarity: %{y:.3f}<extra></extra>",
    ))

# Reference lines for the ±0.15 neutral zone
fig.add_hline(y=0.15, line_dash="dot", line_color="#aaaaaa", line_width=1)
fig.add_hline(y=-0.15, line_dash="dot", line_color="#aaaaaa", line_width=1)
fig.add_hline(y=0, line_dash="dash", line_color="#cccccc", line_width=1)

fig.update_layout(
    height=420,
    xaxis_title="Published",
    yaxis_title="Polarity (-1 bearish … +1 bullish)",
    yaxis=dict(range=[-1.05, 1.05]),
    hovermode="closest",
    margin=dict(l=20, r=20, t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)


# ─── 4. Headlines table ───────────────────────────────────────────────────

st.subheader("Headlines")

# Build a display-friendly dataframe with clickable URLs.
def _format_label(lbl: str) -> str:
    emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}.get(lbl, "")
    return f"{emoji} {lbl.title()}"

display_df = df.copy()
if "published_at" in display_df.columns:
    display_df["Published"] = display_df["published_at"].dt.strftime("%Y-%m-%d %H:%M")
else:
    display_df["Published"] = ""
display_df["Sentiment"] = display_df["label"].map(_format_label)
display_df["Score"] = display_df["score"].astype(float)
display_df["Headline"] = display_df["headline"]
display_df["Source"] = display_df.get("url", "")

st.dataframe(
    display_df[["Published", "Sentiment", "Score", "Headline", "Source"]]
    .sort_values("Published", ascending=False)
    .reset_index(drop=True),
    use_container_width=True,
    column_config={
        "Score": st.column_config.NumberColumn(format="%.3f"),
        "Source": st.column_config.LinkColumn("Source", display_text="open ↗"),
    },
    hide_index=True,
)

st.caption(
    f"{len(articles)} headlines • overall polarity {overall_score:+.3f} "
    f"({overall_label}) • model: FinBERT (ProsusAI)"
)
