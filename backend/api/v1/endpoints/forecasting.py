"""Forecasting endpoints (ARIMA / LSTM)."""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.deps import ForecastingSvc
from backend.schemas.common import APIResponse
from backend.schemas.forecasting import ForecastRequest, ForecastResponse

router = APIRouter(prefix="/forecasting", tags=["forecasting"])


@router.post(
    "/predict",
    response_model=APIResponse[ForecastResponse],
    summary="Generate a price forecast for a ticker",
)
async def predict(req: ForecastRequest, service: ForecastingSvc) -> APIResponse[ForecastResponse]:
    result = await service.forecast(req)
    return APIResponse(data=result)


@router.get(
    "/models/{ticker}",
    response_model=APIResponse[list[str]],
    summary="List trained models available for a ticker",
)
async def list_models(ticker: str, service: ForecastingSvc) -> APIResponse[list[str]]:
    models = await service.list_available_models(ticker.upper())
    return APIResponse(data=models)
