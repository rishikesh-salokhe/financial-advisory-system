"""
Forecasting service — orchestrates data fetching, model selection, training,
caching, and persistence. Route handlers should call only this layer.

The actual ARIMA / LSTM implementations live under ``ml_engine.forecasting``;
this service is responsible for the higher-level workflow.
"""
from __future__ import annotations

from datetime import date

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.exceptions import ModelNotReadyError
from backend.schemas.forecasting import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
)


class ForecastingService:
    """Coordinates the forecasting workflow.

    Currently a stub — the real implementation will:
      1. Look up a cached forecast in MongoDB (TTL ≈ 1 day for daily horizons).
      2. Fetch historical OHLCV from ``ml_engine.data.yahoo_finance``.
      3. Route to the chosen model (ARIMA / LSTM) via a Forecaster registry.
      4. Persist the result and return it.
    """

    COLLECTION = "forecasts"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._collection = db[self.COLLECTION]

    async def forecast(self, req: ForecastRequest) -> ForecastResponse:
        logger.info(f"Forecast requested: {req.model_dump()}")

        # TODO: cache lookup
        # cached = await self._collection.find_one({"ticker": req.ticker, ...})

        # TODO: implement once ml_engine.forecasting is wired in.
        raise ModelNotReadyError(
            f"Forecasting models are not yet trained for ticker '{req.ticker}'. "
            "Train models via scripts/train_forecaster.py first.",
            details={"ticker": req.ticker, "model_type": req.model_type},
        )

    async def list_available_models(self, ticker: str) -> list[str]:
        """Stub — returns the model types we have artifacts for on disk."""
        logger.debug(f"Listing models for {ticker}")
        return []

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _empty_response(req: ForecastRequest) -> ForecastResponse:
        """Used by tests to build a deterministic zero-point response."""
        return ForecastResponse(
            ticker=req.ticker,
            model_type=req.model_type,
            generated_at=date.today(),
            horizon_days=req.horizon_days,
            points=[],
            metrics={},
        )

    @staticmethod
    def _points_from_arrays(
        dates: list[date],
        preds: list[float],
        lower: list[float] | None = None,
        upper: list[float] | None = None,
    ) -> list[ForecastPoint]:
        lower = lower or [None] * len(preds)  # type: ignore[list-item]
        upper = upper or [None] * len(preds)  # type: ignore[list-item]
        return [
            ForecastPoint(date=d, predicted=p, lower=lo, upper=hi)
            for d, p, lo, hi in zip(dates, preds, lower, upper, strict=True)
        ]
