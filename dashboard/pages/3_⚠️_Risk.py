"""
Risk analytics page — asset-level Sharpe/Sortino/VaR/drawdown/beta vs benchmark.

Auto-discovered by Streamlit because it lives under ``dashboard/pages/``. The
leading ``3_`` controls sidebar order; the emoji controls the label.
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

st.set_page_config(page_title="Risk", page_icon="⚠️", layout="wide")

st.title("⚠️ Asset Risk Analytics")
st.caption(
    "Sharpe, Sortino, historical VaR / CVaR, max drawdown, and market beta — "
    "computed on log returns over the selected lookback window."
)


# ─── Sidebar controls ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Risk settings")
    ticker = st.text_input("Ticker", value="AAPL", max_chars=10).strip().upper()
    benchmark = st.text_input("Benchmark", value="SPY", max_chars=10).strip().upper()
    lookback_days = st.slider("Lookback window (days)", 90, 3650, 730, step=30)
    risk_free_rate = st.slider(
        "Risk-free rate (annualized)",
        0.0, 0.10, 0.04, step=0.005,
        format="%.3f",
    )
    run = st.button("Analyze", type="primary", use_container_width=True)


# ─── Main view ────────────────────────────────────────────────────────────

if not run:
    st.info("👈 Configure parameters in the sidebar and click **Analyze** to begin.")
    st.stop()

if not ticker:
    st.error("Please enter a ticker symbol.")
    st.stop()


# ─── 1. Call backend ──────────────────────────────────────────────────────

client = get_client()
with st.spinner(f"Computing risk metrics for {ticker} vs {benchmark}..."):
    try:
        payload = client.asset_risk({
            "ticker": ticker,
            "benchmark": benchmark,
            "lookback_days": lookback_days,
            "risk_free_rate": risk_free_rate,
        })
    except APIError as exc:
        st.error(f"Backend error ({exc.status_code}): {exc.message}")
        st.stop()


metrics = payload["metrics"]


# ─── 2. Top KPI row ───────────────────────────────────────────────────────

def _fmt_pct(x: float | None, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:.{decimals}f}%"


def _fmt_num(x: float | None, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Annualized Return", _fmt_pct(metrics["annualized_return"]))
k2.metric("Annualized Volatility", _fmt_pct(metrics["annualized_volatility"]))
k3.metric("Sharpe Ratio", _fmt_num(metrics["sharpe"]))
k4.metric("Sortino Ratio", _fmt_num(metrics["sortino"]))
k5.metric("Max Drawdown", _fmt_pct(metrics["max_drawdown"]))


# ─── 3. VaR / CVaR + Market relationship tables ──────────────────────────

c_left, c_right = st.columns(2)

with c_left:
    st.subheader("Tail Risk")
    var_df = pd.DataFrame({
        "Confidence": ["95%", "99%"],
        "VaR (daily loss)": [_fmt_pct(metrics["var_95"]), _fmt_pct(metrics["var_99"])],
        "CVaR (expected shortfall)": [_fmt_pct(metrics["cvar_95"]), _fmt_pct(metrics["cvar_99"])],
    })
    st.dataframe(var_df, use_container_width=True, hide_index=True)
    st.caption(
        "Historical VaR. *VaR* = loss exceeded on the worst α-fraction of days. "
        "*CVaR* = average loss in that tail."
    )

with c_right:
    st.subheader(f"Market Relationship (vs {payload['benchmark']})")
    market_df = pd.DataFrame({
        "Metric": ["Beta", "Alpha (annualized)", "R-squared"],
        "Value": [
            _fmt_num(metrics["beta"], 3),
            _fmt_pct(metrics["alpha_annual"]),
            _fmt_num(metrics["r_squared"], 3),
        ],
    })
    st.dataframe(market_df, use_container_width=True, hide_index=True)
    st.caption(
        "OLS of asset log-returns on benchmark log-returns. "
        "*Beta* = market sensitivity (1.0 = same as benchmark). "
        "*Alpha* = excess return after adjusting for beta-implied return."
    )


# ─── 4. Underwater drawdown chart ─────────────────────────────────────────

st.subheader("Drawdown (Underwater Equity)")

dd_df = pd.DataFrame(payload["drawdown_curve"])
dd_df["date"] = pd.to_datetime(dd_df["date"])

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=dd_df["date"],
    y=dd_df["drawdown_pct"] * 100,
    mode="lines",
    fill="tozeroy",
    fillcolor="rgba(214, 39, 40, 0.25)",
    line=dict(color="#d62728", width=1.5),
    name="Drawdown",
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
))

# Annotate the max drawdown point
if metrics["max_drawdown_trough_date"]:
    fig_dd.add_annotation(
        x=metrics["max_drawdown_trough_date"],
        y=metrics["max_drawdown"] * 100,
        text=f"Max DD: {_fmt_pct(metrics['max_drawdown'])}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-30,
        font=dict(color="#d62728", size=11),
    )

fig_dd.update_layout(
    height=320,
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    margin=dict(l=20, r=20, t=20, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig_dd, use_container_width=True)

# Drawdown timing details
mdd_caption_parts = []
if metrics["max_drawdown_peak_date"]:
    mdd_caption_parts.append(f"Peak: {metrics['max_drawdown_peak_date']}")
if metrics["max_drawdown_trough_date"]:
    mdd_caption_parts.append(f"Trough: {metrics['max_drawdown_trough_date']}")
if metrics["max_drawdown_recovery_date"]:
    mdd_caption_parts.append(f"Recovery: {metrics['max_drawdown_recovery_date']}")
else:
    mdd_caption_parts.append("Recovery: not yet")
if metrics["max_drawdown_duration_days"] is not None:
    mdd_caption_parts.append(f"Peak→Trough: {metrics['max_drawdown_duration_days']} days")
st.caption(" • ".join(mdd_caption_parts))


# ─── 5. Rolling volatility ────────────────────────────────────────────────

st.subheader("Rolling 30-Day Volatility (Annualized)")

rv_df = pd.DataFrame(payload["rolling_volatility"])
rv_df["date"] = pd.to_datetime(rv_df["date"])

fig_rv = go.Figure()
fig_rv.add_trace(go.Scatter(
    x=rv_df["date"],
    y=rv_df["annualized_vol"] * 100,
    mode="lines",
    line=dict(color="#1f77b4", width=2),
    name="Rolling Vol (annualized)",
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
))
# Reference line at full-period annualized volatility
fig_rv.add_hline(
    y=metrics["annualized_volatility"] * 100,
    line_dash="dash",
    line_color="#7f7f7f",
    annotation_text=f"Full-period: {_fmt_pct(metrics['annualized_volatility'])}",
    annotation_position="top right",
)
fig_rv.update_layout(
    height=300,
    xaxis_title="Date",
    yaxis_title="Annualized Volatility (%)",
    margin=dict(l=20, r=20, t=20, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig_rv, use_container_width=True)


# ─── 6. Return distribution with VaR markers ──────────────────────────────

st.subheader("Daily Log-Return Distribution")

hist_df = pd.DataFrame(payload["return_histogram"])
hist_df["center"] = (hist_df["bin_left"] + hist_df["bin_right"]) / 2
hist_df["width"] = hist_df["bin_right"] - hist_df["bin_left"]

# Color bins in the tail (below -VaR_95) red, others blue.
var_95_threshold = -metrics["var_95"]
bar_colors = [
    "#d62728" if center <= var_95_threshold else "#1f77b4"
    for center in hist_df["center"]
]

fig_hist = go.Figure()
fig_hist.add_trace(go.Bar(
    x=hist_df["center"] * 100,
    y=hist_df["count"],
    width=hist_df["width"] * 100,
    marker_color=bar_colors,
    hovertemplate="bin: %{x:.3f}%<br>count: %{y}<extra></extra>",
    name="Returns",
))

# Vertical lines at VaR thresholds
for label, value, color in [
    ("VaR 95%", -metrics["var_95"], "#ff7f0e"),
    ("VaR 99%", -metrics["var_99"], "#d62728"),
]:
    fig_hist.add_vline(
        x=value * 100,
        line_dash="dash",
        line_color=color,
        annotation_text=label,
        annotation_position="top",
    )

fig_hist.update_layout(
    height=320,
    xaxis_title="Daily log return (%)",
    yaxis_title="Count",
    margin=dict(l=20, r=20, t=20, b=40),
    showlegend=False,
    bargap=0,
)
st.plotly_chart(fig_hist, use_container_width=True)


# ─── 7. Footer ────────────────────────────────────────────────────────────

st.caption(
    f"{payload['n_observations']} trading-day observations • "
    f"{payload['start_date']} → {payload['end_date']} • "
    f"benchmark: {payload['benchmark']} • "
    f"risk-free rate: {payload['risk_free_rate']:.2%}"
)
