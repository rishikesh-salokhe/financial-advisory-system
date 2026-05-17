"""
Rule-based risk profiler.

The initial implementation uses a transparent point-scoring system so we can
explain results to users. We can later replace ``compute_score`` with a learned
model trained on the synthetic-user dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RiskTolerance = Literal["conservative", "moderate", "aggressive"]


@dataclass
class ProfileInputs:
    age: int
    annual_income: float
    net_worth: float
    investment_experience_years: int
    loss_tolerance_pct: float
    time_horizon: Literal["short", "medium", "long"]
    dependents: int
    has_emergency_fund: bool


@dataclass
class ProfileResult:
    score: float           # 0–100
    tolerance: RiskTolerance
    equity_pct: float
    bond_pct: float
    cash_pct: float
    rationale: str


def compute_score(inputs: ProfileInputs) -> ProfileResult:
    """Map inputs to a 0–100 score and an allocation.

    The buckets here are intentionally simple — replace with a calibrated model.
    """
    score = 0.0

    # Age: younger investors can take more risk.
    if inputs.age < 35:
        score += 25
    elif inputs.age < 50:
        score += 18
    elif inputs.age < 65:
        score += 10
    else:
        score += 5

    # Horizon
    score += {"short": 5, "medium": 12, "long": 20}[inputs.time_horizon]

    # Experience
    score += min(inputs.investment_experience_years, 15)

    # Loss tolerance
    score += min(inputs.loss_tolerance_pct / 2, 20)

    # Financial cushion
    if inputs.has_emergency_fund:
        score += 10
    if inputs.dependents == 0:
        score += 5
    if inputs.net_worth >= 5 * inputs.annual_income:
        score += 10

    score = max(0.0, min(100.0, score))

    if score < 40:
        tolerance: RiskTolerance = "conservative"
        equity, bond, cash = 25.0, 60.0, 15.0
    elif score < 70:
        tolerance = "moderate"
        equity, bond, cash = 55.0, 35.0, 10.0
    else:
        tolerance = "aggressive"
        equity, bond, cash = 80.0, 15.0, 5.0

    rationale = (
        f"Score={score:.1f}/100 → {tolerance}. "
        f"Allocation: equities {equity:.0f}%, bonds {bond:.0f}%, cash {cash:.0f}%."
    )

    return ProfileResult(
        score=score,
        tolerance=tolerance,
        equity_pct=equity,
        bond_pct=bond,
        cash_pct=cash,
        rationale=rationale,
    )
