"""Request/response models for risk profiling, asset analytics, and portfolio optimization."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RiskTolerance = Literal["conservative", "moderate", "aggressive"]
TimeHorizon = Literal["short", "medium", "long"]


class RiskProfileRequest(BaseModel):
    """Questionnaire-driven inputs for assigning a risk tolerance."""

    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0)
    net_worth: float = Field(..., ge=0)
    investment_experience_years: int = Field(..., ge=0, le=80)
    loss_tolerance_pct: float = Field(..., ge=0, le=100, description="Max drawdown user accepts")
    time_horizon: TimeHorizon
    dependents: int = Field(0, ge=0)
    has_emergency_fund: bool = True


class RiskProfileResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=100.0)
    tolerance: RiskTolerance
    recommended_equity_pct: float = Field(..., ge=0, le=100)
    recommended_bond_pct: float = Field(..., ge=0, le=100)
    recommended_cash_pct: float = Field(..., ge=0, le=100)
    rationale: str


class PortfolioOptimizationRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, max_length=50)
    lookback_days: int = Field(365, ge=60, le=3650)
    objective: Literal["max_sharpe", "min_volatility", "efficient_return"] = "max_sharpe"
    target_return: float | None = Field(
        None,
        description="Annualized target return; required when objective='efficient_return'",
    )
    risk_free_rate: float = 0.02


class PortfolioAllocation(BaseModel):
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


# ─── Asset-level risk analytics ───────────────────────────────────────────


class AssetRiskRequest(BaseModel):
    """Inputs for the per-asset risk analytics endpoint."""

    # Pydantic v2 reserves the ``model_`` namespace for its internal attrs.
    # Our existing forecasting schemas use it freely too — we silence the
    # warning here for consistency.
    model_config = ConfigDict(protected_namespaces=())

    ticker: str = Field(..., min_length=1, max_length=10)
    lookback_days: int = Field(730, ge=60, le=3650, description="Days of history to analyze")
    benchmark: str = Field("SPY", min_length=1, max_length=10, description="Comparison index ticker")
    risk_free_rate: float = Field(
        0.04,
        ge=0.0,
        le=0.20,
        description="Annualized risk-free rate used for Sharpe / Sortino / Alpha",
    )


class DrawdownPoint(BaseModel):
    date: date
    drawdown_pct: float = Field(..., description="Underwater % from running peak (≤0)")


class RollingVolPoint(BaseModel):
    date: date
    annualized_vol: float


class ReturnHistogramBin(BaseModel):
    bin_left: float
    bin_right: float
    count: int


class AssetRiskMetrics(BaseModel):
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    max_drawdown_peak_date: date | None
    max_drawdown_trough_date: date | None
    max_drawdown_recovery_date: date | None
    max_drawdown_duration_days: int | None
    beta: float
    alpha_annual: float
    r_squared: float


class AssetRiskResponse(BaseModel):
    ticker: str
    benchmark: str
    lookback_days: int
    risk_free_rate: float
    n_observations: int
    start_date: date
    end_date: date
    metrics: AssetRiskMetrics
    drawdown_curve: list[DrawdownPoint]
    rolling_volatility: list[RollingVolPoint]
    return_histogram: list[ReturnHistogramBin]
