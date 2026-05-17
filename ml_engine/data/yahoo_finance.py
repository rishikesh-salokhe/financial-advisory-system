"""
Thin wrapper around yfinance for historical OHLCV data.

Retried with tenacity because the free Yahoo endpoint is flaky. Returns a
pandas DataFrame with a DatetimeIndex (UTC normalized).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
def fetch_history(
    ticker: str,
    *,
    lookback_days: int = 730,
    end: date | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Return OHLCV history for ``ticker``.

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
    """
    import yfinance as yf

    end_dt = datetime.combine(end or date.today(), datetime.min.time())
    start_dt = end_dt - timedelta(days=lookback_days)

    logger.debug(f"Fetching {ticker} from {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}")
    df = yf.download(
        ticker,
        start=start_dt,
        end=end_dt,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def fetch_close_series(ticker: str, lookback_days: int = 730) -> pd.Series:
    """Convenience: return just the adjusted-close series."""
    df = fetch_history(ticker, lookback_days=lookback_days)
    return df["close"].rename(ticker.upper())
