"""Sentiment analysis endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.deps import SentimentSvc
from backend.schemas.common import APIResponse
from backend.schemas.sentiment import SentimentRequest, SentimentSummary

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.post(
    "/analyze",
    response_model=APIResponse[SentimentSummary],
    summary="Run sentiment analysis on news for a ticker",
)
async def analyze(req: SentimentRequest, service: SentimentSvc) -> APIResponse[SentimentSummary]:
    result = await service.analyze(req)
    return APIResponse(data=result)
