"""
Sentiment analysis service — fetches headlines, scores them with FinBERT, and
aggregates results into a ``SentimentSummary``.

The FinBERT pipeline (~440MB) is loaded lazily on the first request and cached
on the service instance, so subsequent calls in the same backend process are
fast. Scoring is dispatched off the FastAPI event loop via ``asyncio.to_thread``
because tokenization + inference are CPU-bound.

If yfinance returns no headlines for the requested ticker, the service returns
a neutral, empty summary rather than raising — empty news is a valid result,
not an error.
"""
from __future__ import annotations

import asyncio
from collections import Counter

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.schemas.sentiment import (
    ArticleSentiment,
    Sentiment,
    SentimentRequest,
    SentimentSummary,
)
from ml_engine.data.news_fetcher import fetch_news
from ml_engine.sentiment.analyzer import FinBERTAnalyzer, SentimentScore


class SentimentService:
    """Coordinates the news-fetch → FinBERT → aggregation workflow."""

    COLLECTION = "sentiment_runs"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._collection = db[self.COLLECTION]
        # Lazy-initialized FinBERT pipeline. The first analyze() call pays the
        # 440MB download + load cost; later calls reuse the cached instance.
        self._analyzer: FinBERTAnalyzer | None = None

    # ─── Public API ───────────────────────────────────────────────────────

    async def analyze(self, req: SentimentRequest) -> SentimentSummary:
        ticker = req.ticker.upper()
        days_back = self._compute_days_back(req)
        logger.info(
            f"Sentiment {ticker} max_articles={req.max_articles} days_back={days_back}"
        )

        # 1. Fetch headlines (network I/O, run on a thread so we don't block).
        articles = await asyncio.to_thread(
            fetch_news,
            ticker=ticker,
            max_articles=req.max_articles,
            days_back=days_back,
        )

        # 2. Filter by date range if requested.
        if req.start_date or req.end_date:
            articles = self._filter_by_date(articles, req.start_date, req.end_date)

        if not articles:
            logger.info(f"No headlines for {ticker} — returning neutral empty summary")
            return SentimentSummary(
                ticker=ticker,
                overall_label="neutral",
                overall_score=0.0,
                article_count=0,
                distribution={"positive": 0, "neutral": 0, "negative": 0},
                articles=[],
            )

        # 3. Score via FinBERT (CPU-bound — push off the event loop).
        headlines = [a["title"] for a in articles]
        scores = await asyncio.to_thread(self._score_batch, headlines)

        # 4. Pair article metadata with scores.
        per_article = [
            ArticleSentiment(
                headline=a["title"],
                url=a.get("url"),
                published_at=a.get("published_at"),
                label=s.label,
                score=s.score,
            )
            for a, s in zip(articles, scores, strict=True)
        ]

        # 5. Aggregate.
        overall_score, overall_label, distribution = self._aggregate(scores)
        return SentimentSummary(
            ticker=ticker,
            overall_label=overall_label,
            overall_score=overall_score,
            article_count=len(per_article),
            distribution=distribution,
            articles=per_article,
        )

    # ─── Sync core (called via to_thread) ─────────────────────────────────

    def _score_batch(self, headlines: list[str]) -> list[SentimentScore]:
        """Lazy-init the analyzer and score all headlines in one call."""
        if self._analyzer is None:
            logger.info("Loading FinBERT pipeline (first request — may take ~30s on cold cache)")
            self._analyzer = FinBERTAnalyzer()
        return self._analyzer.score_batch(headlines)

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_days_back(req: SentimentRequest) -> int:
        """Derive a sensible ``days_back`` for the fetcher from the request.

        If the caller specified ``start_date`` we honor it; otherwise default
        to 30 days, which gives FinBERT enough headlines to produce a meaningful
        aggregate without overwhelming the free-tier quotas.
        """
        from datetime import date as _date

        if req.start_date is not None:
            delta = (_date.today() - req.start_date).days
            return max(1, min(delta, 365))  # clamp to [1, 365]
        return 30

    @staticmethod
    def _filter_by_date(articles: list[dict], start, end) -> list[dict]:
        """Keep only articles whose ``published_at`` falls in [start, end]."""
        kept = []
        for art in articles:
            pub = art.get("published_at")
            if pub is None:
                continue
            pub_date = pub.date() if hasattr(pub, "date") else pub
            if start and pub_date < start:
                continue
            if end and pub_date > end:
                continue
            kept.append(art)
        return kept

    @staticmethod
    def _aggregate(
        scores: list[SentimentScore],
    ) -> tuple[float, Sentiment, dict[Sentiment, int]]:
        """Average polarity, derive overall label, count label distribution."""
        if not scores:
            return 0.0, "neutral", {"positive": 0, "neutral": 0, "negative": 0}

        overall_score = sum(s.score for s in scores) / len(scores)

        # Same thresholds VADER uses for compound scores — works well empirically
        # for the [-1, 1] polarity FinBERTAnalyzer emits.
        if overall_score >= 0.15:
            overall_label: Sentiment = "positive"
        elif overall_score <= -0.15:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        counts = Counter(s.label for s in scores)
        distribution: dict[Sentiment, int] = {
            "positive": int(counts.get("positive", 0)),
            "neutral": int(counts.get("neutral", 0)),
            "negative": int(counts.get("negative", 0)),
        }
        return overall_score, overall_label, distribution
