"""Request/response models for sentiment analysis."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


Sentiment = Literal["positive", "neutral", "negative"]


class SentimentRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    start_date: date | None = None
    end_date: date | None = None
    max_articles: int = Field(50, ge=1, le=500)


class ArticleSentiment(BaseModel):
    headline: str
    url: str | None = None
    published_at: datetime | None = None
    label: Sentiment
    score: float = Field(..., ge=-1.0, le=1.0, description="Polarity in [-1, 1]")


class SentimentSummary(BaseModel):
    ticker: str
    overall_label: Sentiment
    overall_score: float = Field(..., ge=-1.0, le=1.0)
    article_count: int
    distribution: dict[Sentiment, int]
    articles: list[ArticleSentiment]
