"""Request/response models for risk profiling and portfolio optimization."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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
