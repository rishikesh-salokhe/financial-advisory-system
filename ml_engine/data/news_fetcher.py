"""
Cached Yahoo Finance news fetcher.

Pulls the most recent headlines for a ticker via ``yfinance.Ticker.news`` and
normalizes them to a stable schema regardless of the upstream yfinance version.
Headlines are cached as parquet for one hour so a Streamlit user refreshing
the sentiment page doesn't hammer Yahoo on every click.

Two layers:
    1. ``_download_news``  — raw yfinance call wrapped in tenacity retry.
    2. ``fetch_news``      — public API with a 1-hour parquet cache keyed by
                             ticker. Cache lives under ``data/cache/news/``.

The normalized record shape returned by ``fetch_news``:
    {
        "title":        str,                # headline text
        "url":          str | None,         # canonical article URL
        "publisher":    str | None,         # source name
        "published_at": pandas.Timestamp,   # tz-naive UTC
        "uuid":         str | None,         # de-dup key
    }
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import settings


# ─── Constants ────────────────────────────────────────────────────────────

CACHE_DIR: Path = settings.vectorstore_dir.parent / "cache" / "news"
CACHE_TTL = timedelta(hours=1)


# ─── Internal download (no cache) ─────────────────────────────────────────


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
def _download_news(ticker: str) -> list[dict[str, Any]]:
    """Raw, un-cached fetch. Internal — public callers should use ``fetch_news``."""
    import yfinance as yf

    logger.debug(f"yfinance news fetch for {ticker}")
    raw: list[dict[str, Any]] = yf.Ticker(ticker).news or []
    if not raw:
        # Yahoo sometimes returns an empty list for valid tickers if it has no
        # current headlines; that's a soft error, not a retry-worthy one.
        logger.info(f"No news returned by yfinance for {ticker}")
        return []
    return raw


# ─── Schema normalization ─────────────────────────────────────────────────


def _normalize_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one yfinance news record to our stable schema.

    yfinance has shipped at least two different shapes for the news payload
    over recent versions; we accept both:

    * Legacy flat shape: ``{title, publisher, link, providerPublishTime, ...}``
    * Newer nested shape: ``{id, content: {title, provider, canonicalUrl, pubDate, ...}}``

    Returns ``None`` if the record lacks a usable title.
    """
    # Newer nested shape ---------------------------------------------------
    content = rec.get("content") if isinstance(rec.get("content"), dict) else None
    if content:
        title = (content.get("title") or "").strip()
        if not title:
            return None
        url = None
        if isinstance(content.get("canonicalUrl"), dict):
            url = content["canonicalUrl"].get("url")
        elif isinstance(content.get("clickThroughUrl"), dict):
            url = content["clickThroughUrl"].get("url")
        publisher = None
        if isinstance(content.get("provider"), dict):
            publisher = content["provider"].get("displayName")
        published_raw = content.get("pubDate") or content.get("displayTime")
        published_at = _coerce_timestamp(published_raw)
        return {
            "title": title,
            "url": url,
            "publisher": publisher,
            "published_at": published_at,
            "uuid": rec.get("id") or content.get("id"),
        }

    # Legacy flat shape ----------------------------------------------------
    title = (rec.get("title") or "").strip()
    if not title:
        return None
    published_at = _coerce_timestamp(rec.get("providerPublishTime"))
    return {
        "title": title,
        "url": rec.get("link"),
        "publisher": rec.get("publisher"),
        "published_at": published_at,
        "uuid": rec.get("uuid"),
    }


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    """Convert yfinance's various time formats into a tz-naive UTC Timestamp."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            ts = pd.to_datetime(value, unit="s", utc=True)
        else:
            ts = pd.to_datetime(str(value), utc=True)
        return ts.tz_convert(None)
    except (ValueError, TypeError):
        return None


# ─── Cache helpers ────────────────────────────────────────────────────────


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}_{datetime.now():%Y%m%d_%H}.parquet"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < CACHE_TTL


# ─── Public API ───────────────────────────────────────────────────────────


def fetch_news(
    ticker: str,
    *,
    max_articles: int = 50,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Return up to ``max_articles`` normalized news records for ``ticker``.

    Records are sorted newest-first. The list may be empty if yfinance has no
    headlines for the ticker (which it sometimes returns even for liquid names).
    """
    cache_file = _cache_path(ticker)

    if use_cache and _cache_is_fresh(cache_file):
        logger.debug(f"News cache hit: {cache_file.name}")
        cached = pd.read_parquet(cache_file)
        return cached.head(max_articles).to_dict(orient="records")

    raw = _download_news(ticker)
    normalized: list[dict[str, Any]] = []
    for rec in raw:
        norm = _normalize_record(rec)
        if norm is not None:
            normalized.append(norm)

    # Sort newest-first; entries without a timestamp sink to the bottom.
    normalized.sort(
        key=lambda r: r["published_at"] if r["published_at"] is not None else pd.Timestamp.min,
        reverse=True,
    )

    if use_cache and normalized:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(normalized).to_parquet(cache_file)
        logger.debug(f"News cache write: {cache_file.name} ({len(normalized)} rows)")

    return normalized[:max_articles]
