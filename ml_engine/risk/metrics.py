"""
Asset-level risk metrics — pure functions over a univariate price series.

Conventions
-----------
* **Log returns** for risk math. They're additive across time (so cumulative
  returns sum cleanly), symmetric around zero, and are the standard in
  academic finance.
* **252 trading days** per year for annualization.
* **Historical VaR / CVaR** — no distributional assumption. Each is reported
  as a *positive* magnitude representing the expected loss.
* **Drawdown** is computed from the running peak in *price space*: every
  point's drawdown is ``(current - running_peak) / running_peak``.
* **Beta** is the ordinary-least-squares slope of asset log-returns regressed
  on benchmark log-returns. Alpha is Jensen's alpha (annualized).

Each function takes a ``pandas.Series`` and emits either a scalar, a Series,
or a small dataclass. Functions are stateless and side-effect-free so they're
straightforward to unit-test and compose.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.04   # ~4% short-term Treasury yield in mid-2026


# ─── Returns ──────────────────────────────────────────────────────────────


def simple_returns(prices: pd.Series) -> pd.Series:
    """Daily simple returns: r_t = P_t / P_{t-1} - 1."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns: r_t = log(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def cumulative_returns(log_rets: pd.Series) -> pd.Series:
    """Cumulative growth path starting at 0 (so ``+0.10`` means +10%)."""
    return np.exp(log_rets.cumsum()) - 1.0


# ─── Volatility ───────────────────────────────────────────────────────────


def annualized_return(log_rets: pd.Series) -> float:
    """Mean daily log-return scaled to annual (≈ geometric annual return)."""
    if log_rets.empty:
        return float("nan")
    return float(log_rets.mean() * TRADING_DAYS_PER_YEAR)


def annualized_volatility(log_rets: pd.Series) -> float:
    """Standard deviation of daily log-returns × √252."""
    if log_rets.empty:
        return float("nan")
    return float(log_rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def rolling_volatility(log_rets: pd.Series, window: int = 30) -> pd.Series:
    """Rolling annualized volatility over a trailing ``window`` of days."""
    return log_rets.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )


# ─── Risk-adjusted return ─────────────────────────────────────────────────


def sharpe_ratio(log_rets: pd.Series, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
    """Annualized Sharpe = (annual excess return) / (annual volatility)."""
    if log_rets.empty:
        return float("nan")
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = log_rets - daily_rf
    ann_excess = excess.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = log_rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(ann_excess / ann_vol) if ann_vol > 0 else float("nan")


def sortino_ratio(log_rets: pd.Series, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
    """Annualized Sortino = (annual excess return) / (annual downside deviation).

    Downside deviation uses only returns below the daily risk-free rate.
    """
    if log_rets.empty:
        return float("nan")
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = log_rets - daily_rf
    ann_excess = excess.mean() * TRADING_DAYS_PER_YEAR

    downside = excess[excess < 0]
    if downside.empty:
        return float("inf")  # No down days — meaningless ratio
    downside_dev = float(np.sqrt((downside**2).mean())) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(ann_excess / downside_dev) if downside_dev > 0 else float("nan")


# ─── Tail risk ────────────────────────────────────────────────────────────


def historical_var(log_rets: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value-at-Risk at the given confidence level.

    Returned as a *positive* daily loss magnitude. For example, ``0.025`` means
    "on the worst 5% of days, daily loss was at least 2.5%".
    """
    if log_rets.empty:
        return float("nan")
    alpha = 1.0 - confidence
    quantile = float(np.quantile(log_rets, alpha))
    return float(-quantile) if quantile < 0 else 0.0


def historical_cvar(log_rets: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall): mean of returns in the worst α tail."""
    if log_rets.empty:
        return float("nan")
    alpha = 1.0 - confidence
    threshold = np.quantile(log_rets, alpha)
    tail = log_rets[log_rets <= threshold]
    if tail.empty:
        return float("nan")
    return float(-tail.mean()) if tail.mean() < 0 else 0.0


# ─── Drawdown ─────────────────────────────────────────────────────────────


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Underwater equity curve in [-1, 0]: 0 at new peaks, -0.20 at 20% below peak."""
    running_peak = prices.cummax()
    return (prices - running_peak) / running_peak


@dataclass
class DrawdownResult:
    max_drawdown: float                 # negative number, e.g. -0.34
    peak_date: date | None
    trough_date: date | None
    recovery_date: date | None          # None if never recovered yet
    duration_days: int | None           # peak → trough in calendar days


def max_drawdown(prices: pd.Series) -> DrawdownResult:
    """Compute the worst peak-to-trough decline + timing metadata."""
    if prices.empty:
        return DrawdownResult(
            max_drawdown=float("nan"),
            peak_date=None,
            trough_date=None,
            recovery_date=None,
            duration_days=None,
        )

    dd = drawdown_series(prices)
    trough_idx = dd.idxmin()
    mdd = float(dd.loc[trough_idx])

    # The peak preceding the trough is the running max up to and including the trough.
    peak_idx = prices.loc[:trough_idx].idxmax()
    peak_price = float(prices.loc[peak_idx])

    # Recovery = first date AFTER the trough where price ≥ peak price.
    after_trough = prices.loc[trough_idx:]
    recovered = after_trough[after_trough >= peak_price]
    recovery_idx = recovered.index[0] if not recovered.empty else None

    return DrawdownResult(
        max_drawdown=mdd,
        peak_date=pd.Timestamp(peak_idx).date(),
        trough_date=pd.Timestamp(trough_idx).date(),
        recovery_date=pd.Timestamp(recovery_idx).date() if recovery_idx is not None else None,
        duration_days=int((pd.Timestamp(trough_idx) - pd.Timestamp(peak_idx)).days),
    )


# ─── Market relationship ─────────────────────────────────────────────────


@dataclass
class BetaAlphaResult:
    beta: float
    alpha_annual: float
    r_squared: float
    n_obs: int


def beta_alpha(
    asset_log_rets: pd.Series,
    benchmark_log_rets: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> BetaAlphaResult:
    """Estimate β and Jensen's α via OLS on aligned daily log-returns."""
    aligned = pd.concat(
        [asset_log_rets.rename("asset"), benchmark_log_rets.rename("bench")],
        axis=1,
        join="inner",
    ).dropna()
    n = len(aligned)
    if n < 30:
        return BetaAlphaResult(
            beta=float("nan"), alpha_annual=float("nan"), r_squared=float("nan"), n_obs=n
        )

    a = aligned["asset"].to_numpy()
    b = aligned["bench"].to_numpy()

    cov = np.cov(a, b, ddof=1)
    bench_var = cov[1, 1]
    if bench_var <= 0:
        return BetaAlphaResult(
            beta=float("nan"), alpha_annual=float("nan"), r_squared=float("nan"), n_obs=n
        )

    beta = float(cov[0, 1] / bench_var)
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    alpha_daily = (float(a.mean()) - daily_rf) - beta * (float(b.mean()) - daily_rf)
    alpha_annual = alpha_daily * TRADING_DAYS_PER_YEAR

    corr = float(np.corrcoef(a, b)[0, 1])
    r_squared = corr * corr

    return BetaAlphaResult(beta=beta, alpha_annual=alpha_annual, r_squared=r_squared, n_obs=n)


# ─── Distribution helper (for the UI histogram) ──────────────────────────


def return_histogram(log_rets: pd.Series, n_bins: int = 50) -> list[dict[str, float | int]]:
    """Build histogram bins suitable for a frontend bar chart."""
    if log_rets.empty:
        return []
    counts, edges = np.histogram(log_rets.values, bins=n_bins)
    out: list[dict[str, float | int]] = []
    for i, c in enumerate(counts):
        out.append({
            "bin_left": float(edges[i]),
            "bin_right": float(edges[i + 1]),
            "count": int(c),
        })
    return out
