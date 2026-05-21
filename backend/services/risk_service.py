"""
Risk service — three responsibilities:

    1. ``profile(req)``          — Questionnaire-driven user risk tolerance.
    2. ``analyze_asset(req)``    — Asset-level risk analytics (Sharpe, VaR,
                                   drawdown, beta vs benchmark, distribution).
    3. ``optimize_portfolio()``  — Stub for the upcoming Phase 5 (mean-variance
                                   optimization via PyPortfolioOpt).

``analyze_asset`` does CPU-heavy numpy work on the event loop, so it dispatches
to a worker thread via ``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.exceptions import ExternalServiceError, ModelNotReadyError, ValidationError
from backend.schemas.risk import (
    AssetRiskMetrics,
    AssetRiskRequest,
    AssetRiskResponse,
    BaselineDTO,
    DiscreteAllocationDTO,
    DrawdownPoint,
    FrontierPointDTO,
    PortfolioAllocation,
    PortfolioOptimizationRequest,
    ReturnHistogramBin,
    RiskProfileRequest,
    RiskProfileResponse,
    RollingVolPoint,
)
from ml_engine.data.yahoo_finance import fetch_close_series
from ml_engine.portfolio import optimizer as PO
from ml_engine.risk import metrics as M
from ml_engine.risk.profiler import ProfileInputs, compute_score


class RiskService:
    """Coordinates the questionnaire profiler, asset analytics, and portfolio optimizer."""

    COLLECTION = "risk_profiles"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._collection = db[self.COLLECTION]

    # ─── User-centric risk profile ────────────────────────────────────────

    async def profile(self, req: RiskProfileRequest) -> RiskProfileResponse:
        """Map questionnaire answers to a 0–100 score + allocation."""
        logger.info(f"Risk profile (age={req.age}, horizon={req.time_horizon})")
        inputs = ProfileInputs(
            age=req.age,
            annual_income=req.annual_income,
            net_worth=req.net_worth,
            investment_experience_years=req.investment_experience_years,
            loss_tolerance_pct=req.loss_tolerance_pct,
            time_horizon=req.time_horizon,
            dependents=req.dependents,
            has_emergency_fund=req.has_emergency_fund,
        )
        result = compute_score(inputs)
        return RiskProfileResponse(
            score=result.score,
            tolerance=result.tolerance,
            recommended_equity_pct=result.equity_pct,
            recommended_bond_pct=result.bond_pct,
            recommended_cash_pct=result.cash_pct,
            rationale=result.rationale,
        )

    # ─── Asset-level analytics ────────────────────────────────────────────

    async def analyze_asset(self, req: AssetRiskRequest) -> AssetRiskResponse:
        ticker = req.ticker.upper()
        benchmark = req.benchmark.upper()
        logger.info(
            f"Asset risk {ticker} vs {benchmark} "
            f"lookback={req.lookback_days}d rf={req.risk_free_rate:.2%}"
        )
        return await asyncio.to_thread(
            self._analyze_asset_sync,
            ticker=ticker,
            benchmark=benchmark,
            lookback_days=req.lookback_days,
            risk_free_rate=req.risk_free_rate,
        )

    def _analyze_asset_sync(
        self,
        ticker: str,
        benchmark: str,
        lookback_days: int,
        risk_free_rate: float,
    ) -> AssetRiskResponse:
        # 1. Fetch prices ------------------------------------------------
        try:
            asset_prices = fetch_close_series(ticker, lookback_days=lookback_days)
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(
                f"Failed to fetch prices for {ticker}: {exc}",
                details={"ticker": ticker},
            ) from exc

        if len(asset_prices) < 30:
            raise ModelNotReadyError(
                f"Insufficient history for {ticker} ({len(asset_prices)} days); need at least 30",
                details={"ticker": ticker, "available_days": len(asset_prices)},
            )

        try:
            bench_prices = fetch_close_series(benchmark, lookback_days=lookback_days)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Benchmark {benchmark} fetch failed: {exc}; beta/alpha will be NaN")
            bench_prices = pd.Series(dtype=float)

        # 2. Compute log returns ----------------------------------------
        asset_log = M.log_returns(asset_prices)
        bench_log = M.log_returns(bench_prices) if not bench_prices.empty else pd.Series(dtype=float)

        # 3. Scalar metrics ---------------------------------------------
        mdd_res = M.max_drawdown(asset_prices)
        ba = M.beta_alpha(asset_log, bench_log, risk_free_rate=risk_free_rate)

        scalar = AssetRiskMetrics(
            annualized_return=M.annualized_return(asset_log),
            annualized_volatility=M.annualized_volatility(asset_log),
            sharpe=M.sharpe_ratio(asset_log, risk_free_rate=risk_free_rate),
            sortino=M.sortino_ratio(asset_log, risk_free_rate=risk_free_rate),
            var_95=M.historical_var(asset_log, confidence=0.95),
            var_99=M.historical_var(asset_log, confidence=0.99),
            cvar_95=M.historical_cvar(asset_log, confidence=0.95),
            cvar_99=M.historical_cvar(asset_log, confidence=0.99),
            max_drawdown=mdd_res.max_drawdown,
            max_drawdown_peak_date=mdd_res.peak_date,
            max_drawdown_trough_date=mdd_res.trough_date,
            max_drawdown_recovery_date=mdd_res.recovery_date,
            max_drawdown_duration_days=mdd_res.duration_days,
            beta=ba.beta,
            alpha_annual=ba.alpha_annual,
            r_squared=ba.r_squared,
        )

        # 4. Time series ------------------------------------------------
        dd_series = M.drawdown_series(asset_prices)
        drawdown_curve = [
            DrawdownPoint(date=pd.Timestamp(idx).date(), drawdown_pct=float(val))
            for idx, val in dd_series.items()
            if pd.notna(val)
        ]

        rv = M.rolling_volatility(asset_log, window=30).dropna()
        rolling_vol = [
            RollingVolPoint(date=pd.Timestamp(idx).date(), annualized_vol=float(val))
            for idx, val in rv.items()
        ]

        histogram = [ReturnHistogramBin(**b) for b in M.return_histogram(asset_log, n_bins=50)]

        # 5. Assemble response ------------------------------------------
        return AssetRiskResponse(
            ticker=ticker,
            benchmark=benchmark,
            lookback_days=lookback_days,
            risk_free_rate=risk_free_rate,
            n_observations=len(asset_log),
            start_date=pd.Timestamp(asset_prices.index[0]).date(),
            end_date=pd.Timestamp(asset_prices.index[-1]).date(),
            metrics=scalar,
            drawdown_curve=drawdown_curve,
            rolling_volatility=rolling_vol,
            return_histogram=histogram,
        )

    # ─── Portfolio optimization ───────────────────────────────────────────

    async def optimize_portfolio(
        self, req: PortfolioOptimizationRequest
    ) -> PortfolioAllocation:
        """Mean-variance optimization via PyPortfolioOpt (Ledoit-Wolf covariance).

        CPU-bound (cvxpy solves several QPs for the frontier sweep), so we
        dispatch to a worker thread to keep the event loop responsive.
        """
        tickers = [t.upper() for t in req.tickers]
        logger.info(
            f"Portfolio optimization: {tickers} objective={req.objective} "
            f"lookback={req.lookback_days}d budget=${req.budget:,.0f}"
        )

        if req.objective == "efficient_return" and req.target_return is None:
            raise ValidationError(
                "target_return is required when objective='efficient_return'.",
                details={"objective": req.objective},
            )

        return await asyncio.to_thread(
            self._optimize_portfolio_sync,
            tickers=tickers,
            lookback_days=req.lookback_days,
            objective=req.objective,
            target_return=req.target_return,
            risk_free_rate=req.risk_free_rate,
            budget=req.budget,
            include_frontier=req.include_frontier,
            frontier_points=req.frontier_points,
        )

    def _optimize_portfolio_sync(
        self,
        tickers: list[str],
        lookback_days: int,
        objective: str,
        target_return: float | None,
        risk_free_rate: float,
        budget: float,
        include_frontier: bool,
        frontier_points: int,
    ) -> PortfolioAllocation:
        # 1. Fetch + align price panel ----------------------------------
        try:
            prices = PO.fetch_price_panel(tickers, lookback_days=lookback_days)
        except ValueError as exc:
            # Domain-level failure (e.g. < 2 tickers usable, no overlap)
            raise ModelNotReadyError(
                str(exc),
                details={"tickers": tickers, "lookback_days": lookback_days},
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(
                f"Failed to fetch price data: {exc}",
                details={"tickers": tickers},
            ) from exc

        tickers_used = [str(c) for c in prices.columns]
        tickers_failed = [t for t in tickers if t not in tickers_used]

        # 2. Solve the requested objective ------------------------------
        try:
            opt = PO.optimize(
                prices,
                objective=objective,  # type: ignore[arg-type]
                target_return=target_return,
                risk_free_rate=risk_free_rate,
            )
        except ValueError as exc:
            # Most commonly: target_return outside the achievable range
            raise ValidationError(
                f"Optimization infeasible: {exc}",
                details={
                    "objective": objective,
                    "target_return": target_return,
                    "hint": "Try a target return between min-vol return and max single-asset return.",
                },
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ModelNotReadyError(
                f"Solver failed: {exc}",
                details={"objective": objective},
            ) from exc

        # 3. Discrete share allocation ----------------------------------
        disc = PO.discrete_allocate(opt.weights, prices, budget=budget)

        # 4. Equal-weight baseline (cheap; always compute) --------------
        baseline = PO.equal_weight_baseline(prices, risk_free_rate=risk_free_rate)

        # 5. Efficient frontier sweep (optional) ------------------------
        frontier: list[FrontierPointDTO] = []
        if include_frontier:
            for pt in PO.compute_frontier(
                prices,
                n_points=frontier_points,
                risk_free_rate=risk_free_rate,
            ):
                frontier.append(FrontierPointDTO(
                    volatility=pt.volatility,
                    expected_return=pt.expected_return,
                    sharpe_ratio=pt.sharpe_ratio,
                ))

        # 6. Assemble response ------------------------------------------
        start_d, end_d = PO.panel_date_range(prices)
        return PortfolioAllocation(
            weights=opt.weights,
            expected_return=opt.expected_return,
            volatility=opt.volatility,
            sharpe_ratio=opt.sharpe_ratio,
            discrete_allocation=DiscreteAllocationDTO(
                shares=disc.shares,
                leftover_cash=disc.leftover_cash,
                total_invested=disc.total_invested,
                latest_prices=disc.latest_prices,
            ),
            efficient_frontier=frontier,
            baseline=BaselineDTO(
                weights=baseline.weights,
                expected_return=baseline.expected_return,
                volatility=baseline.volatility,
                sharpe_ratio=baseline.sharpe_ratio,
            ),
            objective=objective,
            tickers_used=tickers_used,
            tickers_failed=tickers_failed,
            n_observations=len(prices),
            lookback_days=lookback_days,
            risk_free_rate=risk_free_rate,
            start_date=start_d,
            end_date=end_d,
        )
