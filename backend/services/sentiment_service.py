"""Sentiment analysis service stub."""
from __future__ import annotations

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.exceptions import ModelNotReadyError
from backend.schemas.sentiment import SentimentRequest, SentimentSummary


class SentimentService:
    COLLECTION = "sentiment_runs"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._collection = db[self.COLLECTION]

    async def analyze(self, req: SentimentRequest) -> SentimentSummary:
        logger.info(f"Sentiment analysis requested for {req.ticker}")
        # TODO: implement once ml_engine.sentiment.analyzer is wired in.
        raise ModelNotReadyError(
            "Sentiment pipeline is not initialized. "
            "Configure NEWS_API_KEY and load the FinBERT pipeline.",
            details={"ticker": req.ticker},
        )
