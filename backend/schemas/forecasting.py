"""Request/response models for the forecasting module."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


ModelType = Literal["arima", "lstm", "auto"]


class ForecastRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    horizon_days: int = Field(30, ge=1, le=365, description="Forecast horizon in trading days")
    model_type: ModelType = "auto"
    lookback_days: int = Field(730, ge=60, le=3650, description="Days of history to train on")
    confidence_interval: float = Field(0.95, ge=0.5, lt=1.0)


class ForecastPoint(BaseModel):
    date: date
    predicted: float
    lower: float | None = None
    upper: float | None = None


class ForecastResponse(BaseModel):
    ticker: str
    model_type: ModelType
    generated_at: date
    horizon_days: int
    points: list[ForecastPoint]
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Backtest metrics (rmse, mape, mae) if available",
    )
