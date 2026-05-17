"""Aggregator for all v1 API routes."""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.v1.endpoints import forecasting, health, rag, risk, sentiment

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(forecasting.router)
api_router.include_router(rag.router)
api_router.include_router(sentiment.router)
api_router.include_router(risk.router)
