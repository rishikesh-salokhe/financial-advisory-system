"""
Portfolio optimization page — mean-variance optimization, efficient frontier,
and whole-share allocation given a dollar budget.

Auto-discovered by Streamlit (lives under ``dashboard/pages/``). The leading
``5_`` controls sidebar order; the emoji controls the label.
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

st.set_page_config(page_title="Portfolio", page_icon="📈", layout="wide")

st.title("📈 Portfolio Optimization")
st.caption(
    "Mean-variance optimization with Ledoit-Wolf shrinkage covariance. "
    "Plots the full efficient frontier, marks the optimal portfolio for your "
    "chosen objective, and translates weights into whole-share allocations "
    "given a dollar budget. Long-only, no leverage."
)


# ─── Sidebar controls ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Portfolio settings")

    tickers_text = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL,MSFT,GOOGL,NVDA,AMZN",
        help="At least two. Yahoo Finance symbols, e.g. AAPL,MSFT,GOOGL.",
    )
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    lookback_days = st.slider("Lookback window (days)", 90, 3650, 365, step=30)

    objective = st.selectbox(
        "Objective",
        options=["max_sharpe", "min_volatility", "efficient_return"],
        format_func=lambda x: {
            "max_sharpe": "Maximize Sharpe ratio",
            "min_volatility": "Minimize volatility",
            "efficient_return": "Target a specific return",
        }[x],
    )

    target_return: float | None = None
    if objective == "efficient_return":
        target_return = st.slider(
            "Target annualized return",
            0.01, 0.50, 0.15, step=0.005,
            format="%.3f",
            help="Solver will minimize volatility subject to this expected return.",
        )

    risk_free_rate = st.slider(
        "Risk-free rate (annualized)",
        0.0, 0.10, 0.04, step=0.005,
        format="%.3f",
    )

    budget = st.number_input(
        "Investment budget ($)",
        min_value=100.0,
        max_value=10_000_000.0,
        value=10_000.0,
        step=500.0,
    )

    include_frontier = st.toggle(
        "Plot efficient frontier",
        value=True,
        help="Adds ~1–2s. Disable for a faster response if you only want weights.",
    )

    run = st.button("Optimize", type="primary", use_container_width=True)


# ─── Guards ───────────────────────────────────────────────────────────────

if not run:
    st.info("👈 Configure your tickers and click **Optimize** to build the portfolio.")
    st.stop()

if len(tickers) < 2:
    st.error("Please enter at least two tickers.")
    st.stop()


# ─── Call backend ─────────────────────────────────────────────────────────

client = get_client()
payload_req = {
    "tickers": tickers,
    "lookback_days": lookback_days,
    "objective": objective,
    "risk_free_rate": risk_free_rate,
    "budget": budget,
    "include_frontier": include_frontier,
    "frontier_points": 40,
}
if target_return is not None:
    payload_req["target_return"] = target_return

with st.spinner(f"Solving {objective.replace('_', ' ')} for {len(tickers)} assets..."):
    try:
        payload = client.portfolio_optimize(payload_req)
    except APIError as exc:
        st.error(f"Backend error ({exc.status_code}): {exc.message}")
        if exc.details:
            with st.expander("Details"):
                st.json(exc.details)
        st.stop()


# ─── Format helpers ───────────────────────────────────────────────────────

def _fmt_pct(x: float | None, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:.{decimals}f}%"


def _fmt_num(x: float | None, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


def _fmt_dollar(x: float, decimals: int = 2) -> str:
    return f"${x:,.{decimals}f}"


# ─── 1. KPI tiles (optimal vs baseline) ───────────────────────────────────

baseline = payload["baseline"]

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Expected Return",
    _fmt_pct(payload["expected_return"]),
    delta=f"{(payload['expected_return'] - baseline['expected_return']) * 100:+.2f}pp vs equal-weight",
)
k2.metric(
    "Volatility",
    _fmt_pct(payload["volatility"]),
    delta=f"{(payload['volatility'] - baseline['volatility']) * 100:+.2f}pp vs equal-weight",
    delta_color="inverse",  # lower vol = good
)
k3.metric(
    "Sharpe Ratio",
    _fmt_num(payload["sharpe_ratio"]),
    delta=f"{payload['sharpe_ratio'] - baseline['sharpe_ratio']:+.2f} vs equal-weight",
)
k4.metric(
    "Diversification",
    f"{sum(1 for w in payload['weights'].values() if w > 0.001)} / {len(payload['tickers_used'])} assets",
    help="Number of assets that received non-trivial weight (>0.1%).",
)

if payload.get("tickers_failed"):
    st.warning(
        f"⚠️  These tickers failed to fetch and were excluded: "
        f"`{', '.join(payload['tickers_failed'])}`"
    )


# ─── 2. Efficient frontier ────────────────────────────────────────────────

if payload["efficient_frontier"]:
    st.subheader("Efficient Frontier")

    frontier_df = pd.DataFrame(payload["efficient_frontier"])

    fig_ef = go.Figure()

    # Frontier curve, colored by Sharpe
    fig_ef.add_trace(go.Scatter(
        x=frontier_df["volatility"] * 100,
        y=frontier_df["expected_return"] * 100,
        mode="markers+lines",
        marker=dict(
            size=6,
            color=frontier_df["sharpe_ratio"],
            colorscale="Viridis",
            colorbar=dict(title="Sharpe", thickness=12),
            showscale=True,
        ),
        line=dict(color="#1f77b4", width=1),
        name="Efficient frontier",
        hovertemplate="vol: %{x:.2f}%<br>return: %{y:.2f}%<extra></extra>",
    ))

    # Optimal portfolio (star)
    fig_ef.add_trace(go.Scatter(
        x=[payload["volatility"] * 100],
        y=[payload["expected_return"] * 100],
        mode="markers",
        marker=dict(symbol="star", size=22, color="#d62728", line=dict(color="white", width=2)),
        name=f"Optimal ({objective.replace('_', ' ')})",
        hovertemplate="<b>Optimal</b><br>vol: %{x:.2f}%<br>return: %{y:.2f}%<extra></extra>",
    ))

    # Equal-weight baseline (diamond)
    fig_ef.add_trace(go.Scatter(
        x=[baseline["volatility"] * 100],
        y=[baseline["expected_return"] * 100],
        mode="markers",
        marker=dict(symbol="diamond", size=14, color="#ff7f0e", line=dict(color="white", width=2)),
        name="Equal-weight (1/N)",
        hovertemplate="<b>Equal-weight</b><br>vol: %{x:.2f}%<br>return: %{y:.2f}%<extra></extra>",
    ))

    # Capital Market Line: tangent line from rf through max-Sharpe point
    # Drawn whenever we have a max-Sharpe-style optimum; visually shows the
    # risk-return tradeoff under leverage (informational only — long-only here).
    if objective == "max_sharpe" and payload["volatility"] > 0:
        rf_pct = risk_free_rate * 100
        slope = (payload["expected_return"] * 100 - rf_pct) / (payload["volatility"] * 100)
        x_max = float(frontier_df["volatility"].max()) * 100 * 1.05
        fig_ef.add_trace(go.Scatter(
            x=[0, x_max],
            y=[rf_pct, rf_pct + slope * x_max],
            mode="lines",
            line=dict(color="#7f7f7f", dash="dash", width=1),
            name="Capital Market Line",
            hoverinfo="skip",
        ))

    fig_ef.update_layout(
        height=480,
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Expected Annualized Return (%)",
        margin=dict(l=20, r=20, t=20, b=40),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ef, use_container_width=True)
    st.caption(
        "Each frontier point is the minimum-volatility portfolio achieving its expected return. "
        "Color = Sharpe ratio. The red star marks your optimized portfolio; the orange diamond is the 1/N baseline."
    )


# ─── 3. Allocation breakdown ──────────────────────────────────────────────

st.subheader("Allocation Breakdown")

c_pie, c_table = st.columns([1, 1])

# Keep only non-trivial weights for the pie
nonzero_weights = {k: v for k, v in payload["weights"].items() if v > 0.001}

with c_pie:
    fig_pie = go.Figure(data=[go.Pie(
        labels=list(nonzero_weights.keys()),
        values=list(nonzero_weights.values()),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
        hole=0.4,
    )])
    fig_pie.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with c_table:
    disc = payload["discrete_allocation"]
    weights_table = pd.DataFrame([
        {
            "Ticker": t,
            "Weight": _fmt_pct(payload["weights"].get(t, 0.0)),
            "Shares": disc["shares"].get(t, 0),
            "Price": _fmt_dollar(disc["latest_prices"].get(t, 0.0)),
            "Value": _fmt_dollar(
                disc["shares"].get(t, 0) * disc["latest_prices"].get(t, 0.0)
            ),
        }
        for t in payload["tickers_used"]
    ])
    st.dataframe(weights_table, use_container_width=True, hide_index=True)
    st.caption(
        f"**Invested:** {_fmt_dollar(disc['total_invested'])} • "
        f"**Leftover cash:** {_fmt_dollar(disc['leftover_cash'])} "
        f"(of {_fmt_dollar(budget)} budget)"
    )


# ─── 4. Optimal vs Baseline comparison table ─────────────────────────────

st.subheader("Optimal vs Equal-Weight Baseline")

comparison_df = pd.DataFrame({
    "Metric": ["Expected return", "Volatility", "Sharpe ratio"],
    "Optimal": [
        _fmt_pct(payload["expected_return"]),
        _fmt_pct(payload["volatility"]),
        _fmt_num(payload["sharpe_ratio"]),
    ],
    "Equal-weight (1/N)": [
        _fmt_pct(baseline["expected_return"]),
        _fmt_pct(baseline["volatility"]),
        _fmt_num(baseline["sharpe_ratio"]),
    ],
    "Difference": [
        f"{(payload['expected_return'] - baseline['expected_return']) * 100:+.2f}pp",
        f"{(payload['volatility'] - baseline['volatility']) * 100:+.2f}pp",
        f"{payload['sharpe_ratio'] - baseline['sharpe_ratio']:+.3f}",
    ],
})
st.dataframe(comparison_df, use_container_width=True, hide_index=True)
st.caption(
    "Equal-weight (1/N) portfolios are a humbling benchmark — research has repeatedly shown "
    "they're hard to beat out-of-sample once estimation error is accounted for. "
    "If the optimizer's edge is thin, that's a useful signal."
)


# ─── 5. Footer ────────────────────────────────────────────────────────────

st.caption(
    f"{payload['n_observations']} trading-day observations • "
    f"{payload['start_date']} → {payload['end_date']} • "
    f"objective: `{payload['objective']}` • "
    f"risk-free rate: {payload['risk_free_rate']:.2%} • "
    f"covariance: Ledoit-Wolf shrinkage"
)
