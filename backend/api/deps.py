"""
Shared FastAPI dependencies.

Service instances are constructed once per request via these dependencies so
that tests can override them with ``app.dependency_overrides``.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.db.mongodb import get_db
from backend.services.forecasting_service import ForecastingService
from backend.services.rag_service import RAGService
from backend.services.risk_service import RiskService
from backend.services.sentiment_service import SentimentService


# ─── Database ─────────────────────────────────────────────────────────────


def db_dependency() -> AsyncIOMotorDatabase:
    return get_db()


DB = Annotated[AsyncIOMotorDatabase, Depends(db_dependency)]


# ─── Services ─────────────────────────────────────────────────────────────


def forecasting_service(db: DB) -> ForecastingService:
    return ForecastingService(db=db)


def rag_service(db: DB) -> RAGService:
    return RAGService(db=db)


def sentiment_service(db: DB) -> SentimentService:
    return SentimentService(db=db)


def risk_service(db: DB) -> RiskService:
    return RiskService(db=db)


ForecastingSvc = Annotated[ForecastingService, Depends(forecasting_service)]
RAGSvc = Annotated[RAGService, Depends(rag_service)]
SentimentSvc = Annotated[SentimentService, Depends(sentiment_service)]
RiskSvc = Annotated[RiskService, Depends(risk_service)]
