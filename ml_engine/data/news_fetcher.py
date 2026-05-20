"""
Multi-source news fetcher with automatic provider fallback.

Provider chain (built dynamically based on configured API keys):
    1. NewsAPI     — newsapi.org /v2/everything; up to 100 articles/req on the
                     free tier, broad coverage.
    2. Finnhub     — finnhub.io /company-news; ticker-keyed financial news.
    3. yfinance    — yf.Ticker(t).news; always available, ~10 headlines max.

Each provider implements the same minimal contract (``fetch(ticker, ...)``
returning a list of normalized records) and is tried in order until one
returns data. If a provider raises or returns empty, we fall through to the
next one — *individual provider failures never propagate*.

Records are normalized to a stable schema regardless of source:
    {
        "title":        str,
        "url":          str | None,
        "publisher":    str | None,
        "published_at": pandas.Timestamp,   # tz-naive UTC
        "uuid":         str | None,
        "source":       str,                # which provider returned this record
    }

Parquet caching layer: 1-hour TTL keyed by (ticker, max_articles, days_back).
The active provider's name is recorded in the cache file too so we can tell
which source served the last successful response.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import settings


# ─── Constants ────────────────────────────────────────────────────────────

CACHE_DIR: Path = settings.vectorstore_dir.parent / "cache" / "news"
CACHE_TTL = timedelta(hours=1)
DEFAULT_DAYS_BACK = 30
HTTP_TIMEOUT = 10  # seconds


# ─── Provider protocol ────────────────────────────────────────────────────


class NewsSource(Protocol):
    """All providers expose the same minimal contract."""

    name: str

    def fetch(
        self, ticker: str, *, max_articles: int, days_back: int
    ) -> list[dict[str, Any]]: ...


# ─── Provider 1: NewsAPI ──────────────────────────────────────────────────


class NewsAPISource:
    """``newsapi.org`` /v2/everything endpoint."""

    name = "newsapi"
    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @retry(reraise=True, stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
    def fetch(self, ticker: str, *, max_articles: int, days_back: int) -> list[dict[str, Any]]:
        page_size = min(max_articles, 100)  # NewsAPI free tier caps at 100
        params = {
            "q": ticker,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "from": (date.today() - timedelta(days=days_back)).isoformat(),
            "searchIn": "title,description",
        }
        headers = {"X-Api-Key": self._api_key}
        logger.debug(f"NewsAPI fetch ticker={ticker} pageSize={page_size} days_back={days_back}")

        resp = requests.get(self.BASE_URL, params=params, headers=headers, timeout=HTTP_TIMEOUT)
        if resp.status_code == 401:
            raise PermissionError("NewsAPI rejected the API key (401)")
        if resp.status_code == 429:
            raise RuntimeError("NewsAPI rate-limited (429) — daily quota exhausted")
        resp.raise_for_status()

        payload = resp.json()
        if payload.get("status") != "ok":
            raise RuntimeError(f"NewsAPI error: {payload.get('message', 'unknown')}")

        articles = payload.get("articles", [])
        normalized: list[dict[str, Any]] = []
        for art in articles:
            title = (art.get("title") or "").strip()
            if not title or title == "[Removed]":
                continue
            normalized.append({
                "title": title,
                "url": art.get("url"),
                "publisher": (art.get("source") or {}).get("name"),
                "published_at": _coerce_timestamp(art.get("publishedAt")),
                "uuid": art.get("url"),  # NewsAPI has no stable ID; URL is the closest thing
                "source": self.name,
            })
        return normalized


# ─── Provider 2: Finnhub ──────────────────────────────────────────────────


class FinnhubSource:
    """``finnhub.io`` /api/v1/company-news endpoint."""

    name = "finnhub"
    BASE_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @retry(reraise=True, stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
    def fetch(self, ticker: str, *, max_articles: int, days_back: int) -> list[dict[str, Any]]:
        today = date.today()
        params = {
            "symbol": ticker,
            "from": (today - timedelta(days=days_back)).isoformat(),
            "to": today.isoformat(),
            "token": self._api_key,
        }
        logger.debug(f"Finnhub fetch ticker={ticker} days_back={days_back}")

        resp = requests.get(self.BASE_URL, params=params, timeout=HTTP_TIMEOUT)
        if resp.status_code == 401:
            raise PermissionError("Finnhub rejected the API key (401)")
        if resp.status_code == 429:
            raise RuntimeError("Finnhub rate-limited (429)")
        resp.raise_for_status()

        articles = resp.json()
        if not isinstance(articles, list):
            raise RuntimeError(f"Finnhub returned unexpected payload type: {type(articles)}")

        # Finnhub returns newest-first by default but we'll re-sort to be sure.
        normalized: list[dict[str, Any]] = []
        for art in articles[:max_articles]:
            title = (art.get("headline") or "").strip()
            if not title:
                continue
            normalized.append({
                "title": title,
                "url": art.get("url"),
                "publisher": art.get("source"),
                "published_at": _coerce_timestamp(art.get("datetime")),
                "uuid": str(art.get("id")) if art.get("id") is not None else None,
                "source": self.name,
            })
        return normalized


# ─── Provider 3: yfinance (fallback, always available) ───────────────────


class YFinanceSource:
    """``yfinance.Ticker.news`` — no API key, ~10 headlines max."""

    name = "yfinance"

    @retry(reraise=True, stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
    def fetch(self, ticker: str, *, max_articles: int, days_back: int) -> list[dict[str, Any]]:
        import yfinance as yf

        logger.debug(f"yfinance fetch ticker={ticker}")
        raw: list[dict[str, Any]] = yf.Ticker(ticker).news or []
        normalized: list[dict[str, Any]] = []
        for rec in raw:
            norm = self._normalize(rec)
            if norm is not None:
                normalized.append(norm)
        return normalized[:max_articles]

    def _normalize(self, rec: dict[str, Any]) -> dict[str, Any] | None:
        """Handle both the legacy flat shape and the newer nested shape."""
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
            return {
                "title": title,
                "url": url,
                "publisher": publisher,
                "published_at": _coerce_timestamp(content.get("pubDate") or content.get("displayTime")),
                "uuid": rec.get("id") or content.get("id"),
                "source": self.name,
            }
        # Legacy flat shape
        title = (rec.get("title") or "").strip()
        if not title:
            return None
        return {
            "title": title,
            "url": rec.get("link"),
            "publisher": rec.get("publisher"),
            "published_at": _coerce_timestamp(rec.get("providerPublishTime")),
            "uuid": rec.get("uuid"),
            "source": self.name,
        }


# ─── Provider-chain assembly ──────────────────────────────────────────────


def _build_provider_chain() -> list[NewsSource]:
    """Construct the ordered list of providers based on configured API keys."""
    chain: list[NewsSource] = []
    if settings.news_api_key:
        chain.append(NewsAPISource(settings.news_api_key))
    if settings.finnhub_api_key:
        chain.append(FinnhubSource(settings.finnhub_api_key))
    chain.append(YFinanceSource())  # always available as last resort
    return chain


# ─── Timestamp coercion (shared helper) ───────────────────────────────────


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    """Convert any of the source-specific time formats into a tz-naive UTC Timestamp."""
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


# ─── Caching ──────────────────────────────────────────────────────────────


def _cache_path(ticker: str, max_articles: int, days_back: int) -> Path:
    key = f"{ticker.upper()}_n{max_articles}_d{days_back}_{datetime.now():%Y%m%d_%H}.parquet"
    return CACHE_DIR / key


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
    days_back: int = DEFAULT_DAYS_BACK,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Return up to ``max_articles`` normalized headlines for ``ticker``.

    Tries providers in order (NewsAPI → Finnhub → yfinance) and returns the
    first non-empty result. Individual provider failures are logged and
    swallowed; the only way to get an empty list back is if **every** provider
    failed or returned no data — which is informational, not an error.
    """
    cache_file = _cache_path(ticker, max_articles, days_back)

    if use_cache and _cache_is_fresh(cache_file):
        logger.debug(f"News cache hit: {cache_file.name}")
        return pd.read_parquet(cache_file).to_dict(orient="records")

    providers = _build_provider_chain()
    if not any(isinstance(p, (NewsAPISource, FinnhubSource)) for p in providers):
        logger.info(
            "No premium news provider configured (NEWS_API_KEY, FINNHUB_API_KEY both unset). "
            "Falling back to yfinance — expect ~10 headlines max."
        )

    last_error: Exception | None = None
    for provider in providers:
        try:
            articles = provider.fetch(ticker, max_articles=max_articles, days_back=days_back)
        except Exception as exc:  # noqa: BLE001 — provider-level isolation is intentional
            logger.warning(f"News provider {provider.name!r} failed: {exc}")
            last_error = exc
            continue

        if not articles:
            logger.info(f"News provider {provider.name!r} returned 0 articles for {ticker}")
            continue

        # Sort newest-first; entries without a timestamp sink to the bottom.
        articles.sort(
            key=lambda r: r["published_at"] if r["published_at"] is not None else pd.Timestamp.min,
            reverse=True,
        )

        logger.info(
            f"Got {len(articles)} headlines for {ticker} from {provider.name!r} "
            f"(days_back={days_back})"
        )

        if use_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(articles).to_parquet(cache_file)
            logger.debug(f"News cache write: {cache_file.name} ({len(articles)} rows)")

        return articles[:max_articles]

    # If we got here, every provider failed or returned empty.
    if last_error is not None:
        logger.error(f"All news providers failed; last error: {last_error}")
    return []
