"""
Cached Yahoo Finance fetcher for historical OHLCV data.

Two layers:
    1. ``_download_history``  — raw yfinance call wrapped in tenacity retry.
    2. ``fetch_history``      — public API with on-disk parquet cache keyed by
                                (ticker, interval, end-date). Cache TTL is one
                                trading day so training scripts can be re-run
                                cheaply during development.

The cache lives under ``data/cache/yfinance/`` and is fully reproducible — if
you delete the folder the next call rebuilds it. The directory is gitignored
so artifacts never pollute the repo.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import settings


# ─── Constants ────────────────────────────────────────────────────────────

CACHE_DIR: Path = settings.vectorstore_dir.parent / "cache" / "yfinance"
CACHE_TTL = timedelta(hours=20)   # ~1 trading day; tweak per environment


# ─── Internal download (no cache) ─────────────────────────────────────────


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
def _download_history(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: str,
) -> pd.DataFrame:
    """Raw, un-cached fetch. Internal — public callers should use ``fetch_history``."""
    import yfinance as yf

    logger.debug(f"yfinance download {ticker} {start:%Y-%m-%d}→{end:%Y-%m-%d} ({interval})")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    # yfinance can return a MultiIndex on columns when multiple tickers are
    # requested. Flatten it here so downstream code never has to care.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df.columns = [str(c).lower() for c in df.columns]
    return df


# ─── Cache helpers ────────────────────────────────────────────────────────


def _cache_path(ticker: str, interval: str, end_dt: datetime) -> Path:
    """Compute the parquet cache file path for a (ticker, interval, end-date) key."""
    key = f"{ticker.upper()}_{interval}_{end_dt:%Y%m%d}.parquet"
    return CACHE_DIR / key


def _cache_is_fresh(path: Path) -> bool:
    """True when the cache file exists and is within ``CACHE_TTL``."""
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < CACHE_TTL


# ─── Public API ───────────────────────────────────────────────────────────


def fetch_history(
    ticker: str,
    *,
    lookback_days: int = 730,
    end: date | None = None,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return OHLCV history for ``ticker`` with transparent on-disk caching.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol (e.g. ``"AAPL"``).
    lookback_days:
        How many calendar days back from ``end`` to fetch.
    end:
        End date (inclusive). Defaults to today.
    interval:
        yfinance interval string (``"1d"``, ``"1h"``, ...).
    use_cache:
        Set ``False`` to bypass the parquet cache for this call (e.g. in tests).

    Returns
    -------
    pandas.DataFrame
        OHLCV dataframe with lowercase columns and a tz-naive DatetimeIndex.
    """
    end_dt = datetime.combine(end or date.today(), datetime.min.time())
    start_dt = end_dt - timedelta(days=lookback_days)
    cache_file = _cache_path(ticker, interval, end_dt)

    if use_cache and _cache_is_fresh(cache_file):
        logger.debug(f"Cache hit: {cache_file.name}")
        return pd.read_parquet(cache_file)

    df = _download_history(ticker, start_dt, end_dt, interval)

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        logger.debug(f"Cache write: {cache_file.name} ({len(df)} rows)")

    return df


def fetch_close_series(ticker: str, lookback_days: int = 730) -> pd.Series:
    """Convenience: return just the adjusted-close series, named after the ticker."""
    df = fetch_history(ticker, lookback_days=lookback_days)
    return df["close"].rename(ticker.upper())
