"""
Sentiment analyzer interface and default pipeline.

The default pipeline uses HuggingFace's FinBERT (``ProsusAI/finbert``) for
financial-domain sentiment scoring. A lighter VADER baseline (NLTK) is provided
for offline development.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


Sentiment = Literal["positive", "neutral", "negative"]


@dataclass
class SentimentScore:
    label: Sentiment
    score: float  # signed polarity in [-1, 1]


class SentimentAnalyzer(Protocol):
    def score(self, text: str) -> SentimentScore: ...
    def score_batch(self, texts: list[str]) -> list[SentimentScore]: ...


# ─── VADER baseline (NLTK) ────────────────────────────────────────────────


class VaderAnalyzer:
    """Quick lexicon-based baseline. Good for smoke tests; weak on finance."""

    def __init__(self) -> None:
        import nltk

        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
        except LookupError:
            nltk.download("vader_lexicon")
            from nltk.sentiment import SentimentIntensityAnalyzer

        self._sia = SentimentIntensityAnalyzer()

    def score(self, text: str) -> SentimentScore:
        compound: float = self._sia.polarity_scores(text)["compound"]
        if compound >= 0.05:
            label: Sentiment = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return SentimentScore(label=label, score=compound)

    def score_batch(self, texts: list[str]) -> list[SentimentScore]:
        return [self.score(t) for t in texts]


# ─── FinBERT pipeline ─────────────────────────────────────────────────────


class FinBERTAnalyzer:
    """FinBERT (ProsusAI) — domain-tuned financial sentiment.

    Loaded lazily on first call; cache the analyzer at the service layer to
    avoid per-request model loads.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self) -> None:
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline

        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            truncation=True,
        )

    def score(self, text: str) -> SentimentScore:
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str]) -> list[SentimentScore]:
        self._ensure_loaded()
        outputs = self._pipeline(texts)  # type: ignore[misc]
        results: list[SentimentScore] = []
        for out in outputs:
            label = out["label"].lower()
            assert label in ("positive", "neutral", "negative")
            # Convert categorical confidence into signed polarity.
            polarity = out["score"] if label == "positive" else (-out["score"] if label == "negative" else 0.0)
            results.append(SentimentScore(label=label, score=polarity))  # type: ignore[arg-type]
        return results
