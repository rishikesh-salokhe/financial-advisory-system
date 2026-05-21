"""
Mean-variance portfolio optimization wrapper around PyPortfolioOpt.

Design notes
------------
- **Long-only**: weight bounds are ``(0, 1)``. Short selling is intentionally
  disabled for a retail-style advisor.
- **Risk model**: Ledoit-Wolf shrinkage covariance rather than the raw sample
  covariance. Shrinkage pulls the covariance estimate toward a structured
  target, which is much more stable on the ~1-year windows we typically use.
- **Expected returns**: simple mean of historical daily returns, annualized by
  252. This is a deliberately humble estimator — fancier estimators like CAPM
  or Black-Litterman just shift bias without improving out-of-sample Sharpe in
  most studies.
- **PyPortfolioOpt is stateful**: each ``EfficientFrontier`` instance can only
  be solved once. To trace the frontier (50+ points), we build a fresh EF per
  target return. This is the documented pattern in their FAQ.

Public surface
--------------
- ``fetch_price_panel(tickers, lookback_days)`` — aligned close-price DataFrame
- ``optimize(prices, objective, ...)`` — solve a single optimization problem
- ``compute_frontier(prices, n_points, ...)`` — sweep the efficient frontier
- ``discrete_allocate(weights, prices, budget)`` — whole-share allocation
- ``equal_weight_baseline(prices, ...)`` — comparison baseline
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from ml_engine.data.yahoo_finance import fetch_close_series


Objective = Literal["max_sharpe", "min_volatility", "efficient_return"]


# ─── Result containers ────────────────────────────────────────────────────


@dataclass
class OptimizedPortfolio:
    """Result of a single optimization run."""

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


@dataclass
class FrontierPoint:
    """One point on the efficient frontier (for plotting)."""

    volatility: float
    expected_return: float
    sharpe_ratio: float


@dataclass
class DiscreteAllocationResult:
    """Whole-share allocation given a dollar budget."""

    shares: dict[str, int]
    leftover_cash: float
    total_invested: float
    latest_prices: dict[str, float]


@dataclass
class EqualWeightBaseline:
    """Equal-weight (1/N) portfolio metrics — sanity-check reference."""

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


# ─── Price fetching ───────────────────────────────────────────────────────


def fetch_price_panel(tickers: list[str], lookback_days: int = 365) -> pd.DataFrame:
    """Return aligned adjusted-close prices for ``tickers`` as a wide DataFrame.

    One column per ticker, indexed by date, NaNs dropped via inner join. If a
    ticker has insufficient overlap with the others it will simply have fewer
    rows after the join.

    Raises ``ValueError`` if no rows survive the join (e.g. completely
    non-overlapping date ranges, or all tickers are invalid).
    """
    if len(tickers) < 2:
        raise ValueError(f"Need at least 2 tickers, got {len(tickers)}")

    series: list[pd.Series] = []
    failed: list[str] = []
    for t in tickers:
        try:
            s = fetch_close_series(t, lookback_days=lookback_days)
            series.append(s)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Ticker {t} failed to fetch: {exc}")
            failed.append(t)

    if len(series) < 2:
        raise ValueError(
            f"Fewer than 2 tickers fetched successfully (failed: {failed}). "
            "Cannot build a portfolio."
        )

    panel = pd.concat(series, axis=1).dropna(how="any")
    if panel.empty:
        raise ValueError(
            "After aligning dates, no overlapping price history remains across the tickers."
        )

    logger.info(
        f"Built price panel: {panel.shape[0]} rows × {panel.shape[1]} tickers "
        f"({panel.index[0].date()} → {panel.index[-1].date()})"
        + (f" — failed: {failed}" if failed else "")
    )
    return panel


# ─── Internal helpers ─────────────────────────────────────────────────────


def _build_ef(prices: pd.DataFrame):
    """Build a fresh EfficientFrontier for ``prices``.

    Imported lazily so unit tests that don't exercise optimization don't pay
    the import cost (pypfopt pulls in cvxpy which is slow).
    """
    from pypfopt import EfficientFrontier, expected_returns, risk_models

    mu = expected_returns.mean_historical_return(prices, frequency=252)
    S = risk_models.CovarianceShrinkage(prices, frequency=252).ledoit_wolf()
    return EfficientFrontier(mu, S, weight_bounds=(0.0, 1.0)), mu, S


def _portfolio_metrics(
    weights: dict[str, float],
    mu: pd.Series,
    S: pd.DataFrame,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    """Compute (annualized return, annualized vol, Sharpe) for an arbitrary weight vector.

    We compute this manually rather than using ``EfficientFrontier.portfolio_performance``
    because that method only works on the optimized portfolio the EF object
    just solved — not on arbitrary weights like 1/N.
    """
    tickers = list(mu.index)
    w = np.array([weights.get(t, 0.0) for t in tickers])
    ret = float(np.dot(w, mu.values))
    vol = float(np.sqrt(np.dot(w, np.dot(S.values, w))))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


# ─── Public optimization API ──────────────────────────────────────────────


def optimize(
    prices: pd.DataFrame,
    objective: Objective = "max_sharpe",
    target_return: float | None = None,
    risk_free_rate: float = 0.02,
) -> OptimizedPortfolio:
    """Solve one optimization problem and return the optimal weights + metrics."""
    ef, _mu, _S = _build_ef(prices)

    if objective == "max_sharpe":
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    elif objective == "min_volatility":
        ef.min_volatility()
    elif objective == "efficient_return":
        if target_return is None:
            raise ValueError("target_return is required when objective='efficient_return'")
        ef.efficient_return(target_return=target_return)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    cleaned = ef.clean_weights(cutoff=1e-4)
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    return OptimizedPortfolio(
        weights=dict(cleaned),
        expected_return=float(ret),
        volatility=float(vol),
        sharpe_ratio=float(sharpe),
    )


def compute_frontier(
    prices: pd.DataFrame,
    n_points: int = 50,
    risk_free_rate: float = 0.02,
) -> list[FrontierPoint]:
    """Trace the efficient frontier for plotting.

    Strategy: find the achievable return range by solving min-vol (lower bound)
    and max-return (upper bound, which is just the asset with highest mu since
    we're long-only with no leverage). Sample ``n_points`` target returns in
    between and solve ``efficient_return`` for each.

    Solves that fail (target slightly above achievable max, infeasible edge
    cases) are skipped silently — better to return n-1 points than crash.
    """
    _, mu, _ = _build_ef(prices)
    max_ret = float(mu.max())

    # Min-vol return — this is the lower-bound return on the frontier
    min_vol = optimize(prices, objective="min_volatility", risk_free_rate=risk_free_rate)
    min_ret = min_vol.expected_return

    # Sweep from min-vol return up to (just shy of) max single-asset return
    targets = np.linspace(min_ret, max_ret * 0.999, n_points)

    points: list[FrontierPoint] = []
    for tgt in targets:
        try:
            res = optimize(
                prices,
                objective="efficient_return",
                target_return=float(tgt),
                risk_free_rate=risk_free_rate,
            )
            points.append(FrontierPoint(
                volatility=res.volatility,
                expected_return=res.expected_return,
                sharpe_ratio=res.sharpe_ratio,
            ))
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Frontier point at target_return={tgt:.4f} failed: {exc}")

    logger.info(f"Frontier: {len(points)}/{n_points} points computed successfully")
    return points


def discrete_allocate(
    weights: dict[str, float],
    prices: pd.DataFrame,
    budget: float,
) -> DiscreteAllocationResult:
    """Convert continuous weights into whole-share counts given a dollar budget.

    Uses PyPortfolioOpt's ``DiscreteAllocation`` with LP solver (falls back to
    greedy if LP fails — which happens for very small budgets).
    """
    from pypfopt.discrete_allocation import DiscreteAllocation

    latest = prices.iloc[-1]
    # DiscreteAllocation expects a plain dict for weights but a pd.Series for prices.
    da = DiscreteAllocation(weights, latest, total_portfolio_value=budget)
    try:
        alloc, leftover = da.lp_portfolio()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"LP allocator failed ({exc}); falling back to greedy")
        alloc, leftover = da.greedy_portfolio()

    # Coerce keys to plain str (could be numpy strings)
    shares = {str(k): int(v) for k, v in alloc.items()}
    invested = sum(shares[t] * float(latest[t]) for t in shares)
    return DiscreteAllocationResult(
        shares=shares,
        leftover_cash=float(leftover),
        total_invested=float(invested),
        latest_prices={str(t): float(latest[t]) for t in latest.index},
    )


def equal_weight_baseline(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> EqualWeightBaseline:
    """1/N portfolio metrics — a humbling baseline that often beats optimization out-of-sample."""
    _, mu, S = _build_ef(prices)
    n = len(mu)
    weights = {t: 1.0 / n for t in mu.index}
    ret, vol, sharpe = _portfolio_metrics(weights, mu, S, risk_free_rate)
    return EqualWeightBaseline(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
    )


# ─── Convenience: panel metadata ──────────────────────────────────────────


def panel_date_range(prices: pd.DataFrame) -> tuple[date, date]:
    return pd.Timestamp(prices.index[0]).date(), pd.Timestamp(prices.index[-1]).date()
