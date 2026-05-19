"""
Forecasting service — orchestrates artifact loading, prediction, and response
shaping. Route handlers should call only this layer.

The actual ARIMA / LSTM implementations live under ``ml_engine.forecasting``;
this service is responsible for the higher-level workflow.

Artifact discovery convention:
    models/forecasting/{TICKER}_{MODEL}.joblib

Training is performed offline via ``scripts/train_forecaster.py``. Forecasts
are computed on-demand from the loaded artifact — fast (<50ms per call) and
stateless, so re-training without restarting the API is safe.
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.config import PROJECT_ROOT
from backend.core.exceptions import ModelNotReadyError
from backend.schemas.forecasting import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    ModelType,
)
from ml_engine.forecasting.arima_model import ARIMAForecaster
from ml_engine.forecasting.base import Forecaster
# LSTMForecaster is imported lazily inside _load_artifact() to keep TensorFlow's
# multi-second import cost off the FastAPI startup path.


MODEL_DIR: Path = PROJECT_ROOT / "models" / "forecasting"


class ForecastingService:
    """Coordinates the forecasting workflow."""

    COLLECTION = "forecasts"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._collection = db[self.COLLECTION]

    # ─── Public API ───────────────────────────────────────────────────────

    async def forecast(self, req: ForecastRequest) -> ForecastResponse:
        ticker = req.ticker.upper()
        model_type = self._resolve_model_type(req.model_type, ticker)
        logger.info(f"Forecast {ticker} model={model_type} horizon={req.horizon_days}d")

        # ML work is CPU-bound — push it off the event loop so the API stays responsive.
        return await asyncio.to_thread(
            self._forecast_sync,
            ticker=ticker,
            model_type=model_type,
            horizon=req.horizon_days,
            confidence=req.confidence_interval,
        )

    async def list_available_models(self, ticker: str) -> list[str]:
        """Return which model types have trained artifacts for ``ticker``."""
        ticker = ticker.upper()
        return [m for m in ("arima", "lstm") if self._artifact_path(ticker, m).exists()]

    # ─── Sync core (called via to_thread) ─────────────────────────────────

    def _forecast_sync(
        self,
        ticker: str,
        model_type: ModelType,
        horizon: int,
        confidence: float,
    ) -> ForecastResponse:
        path = self._artifact_path(ticker, model_type)
        if not path.exists():
            raise ModelNotReadyError(
                f"No trained '{model_type}' model for '{ticker}'. "
                f"Train one with:  python scripts/train_forecaster.py --ticker {ticker}",
                details={"ticker": ticker, "model_type": model_type, "expected_path": str(path)},
            )

        forecaster = self._load_artifact(path, model_type)
        mean, lower, upper = forecaster.predict(horizon=horizon, confidence=confidence)

        # Forecast index: business days after the model's last training observation.
        anchor = forecaster.last_training_date or date.today()
        forecast_dates = _business_days_after(anchor, horizon)

        points = [
            ForecastPoint(date=d, predicted=float(m), lower=float(lo), upper=float(hi))
            for d, m, lo, hi in zip(forecast_dates, mean, lower, upper, strict=True)
        ]

        return ForecastResponse(
            ticker=ticker,
            model_type=model_type,
            generated_at=date.today(),
            horizon_days=horizon,
            points=points,
            metrics=forecaster.metrics,
        )

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _artifact_path(ticker: str, model_type: str) -> Path:
        return MODEL_DIR / f"{ticker.upper()}_{model_type}.joblib"

    @staticmethod
    def _load_artifact(path: Path, model_type: str) -> Forecaster:
        if model_type == "arima":
            return ARIMAForecaster.load(path)
        if model_type == "lstm":
            # Lazy import — TensorFlow takes seconds to import, no reason to pay
            # that cost on every backend startup if no one requests an LSTM forecast.
            from ml_engine.forecasting.lstm_model import LSTMForecaster
            return LSTMForecaster.load(path)
        raise ModelNotReadyError(
            f"Unknown model type '{model_type}'",
            details={"model_type": model_type},
        )

    def _resolve_model_type(self, requested: ModelType, ticker: str) -> ModelType:
        """Resolve ``"auto"`` against artifacts on disk; pass through otherwise."""
        if requested != "auto":
            return requested
        # Prefer LSTM if both exist (more capacity); fall back to ARIMA.
        for candidate in ("lstm", "arima"):
            if self._artifact_path(ticker, candidate).exists():
                return candidate  # type: ignore[return-value]
        raise ModelNotReadyError(
            f"No trained models on disk for '{ticker}'. "
            f"Train one with:  python scripts/train_forecaster.py --ticker {ticker}",
            details={"ticker": ticker, "model_type": "auto"},
        )


# ─── Utility: business-day index ──────────────────────────────────────────


def _business_days_after(start: date, n: int) -> list[date]:
    """Return the next ``n`` weekdays (Mon–Fri) strictly after ``start``.

    Doesn't account for market holidays — fine for forecast display. For
    trading-execution code use ``pandas_market_calendars``.
    """
    start_ts = pd.Timestamp(start) + pd.Timedelta(days=1)
    bdays = pd.bdate_range(start=start_ts, periods=n)
    return [d.date() for d in bdays]
