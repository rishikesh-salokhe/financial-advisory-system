"""
Train and persist a forecasting model for a single ticker.

Usage:
    python scripts/train_forecaster.py --ticker AAPL
    python scripts/train_forecaster.py --ticker AAPL --order 5,1,0
    python scripts/train_forecaster.py --ticker AAPL --lookback-days 1095 --order auto

The trained artifact is saved to:
    models/forecasting/{TICKER}_{MODEL}.joblib

The forecasting API picks up artifacts from that directory automatically on
the next request — no server restart required (the service re-checks on each
call). For production, you'd front this with a registry + versioning.
"""
from __future__ import annotations

# ─── sys.path bootstrap ───────────────────────────────────────────────────
# Allow ``python scripts/train_forecaster.py`` to find the project packages
# without the user having to set PYTHONPATH.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────

import argparse

from loguru import logger

from backend.core.config import PROJECT_ROOT
from backend.core.logging import configure_logging
from ml_engine.data.yahoo_finance import fetch_close_series
from ml_engine.forecasting.arima_model import ARIMAForecaster


MODEL_DIR = PROJECT_ROOT / "models" / "forecasting"


def _parse_order(raw: str) -> tuple[int, int, int] | str:
    """Parse ``--order`` value: either the literal "auto" or "p,d,q"."""
    if raw == "auto":
        return "auto"
    try:
        parts = [int(x.strip()) for x in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--order must be 'auto' or 'p,d,q' integers; got {raw!r}"
        ) from exc
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--order must have exactly three values: p,d,q")
    return tuple(parts)  # type: ignore[return-value]


def train(ticker: str, lookback_days: int, order: tuple[int, int, int] | str) -> Path:
    """Fetch history, fit ARIMA, persist artifact. Returns the artifact path."""
    logger.info(f"Training ARIMA for {ticker} (lookback={lookback_days}d, order={order})")

    series = fetch_close_series(ticker, lookback_days=lookback_days)
    if series.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    logger.info(f"Loaded {len(series)} close prices ({series.index[0].date()} → {series.index[-1].date()})")

    forecaster = ARIMAForecaster(order=order)
    forecaster.fit(series)

    artifact_path = MODEL_DIR / f"{ticker.upper()}_arima.joblib"
    forecaster.save(artifact_path)

    metrics = forecaster.metrics
    logger.info(
        f"Done. order={forecaster.order} "
        f"rmse={metrics.get('rmse', float('nan')):.4f} "
        f"mae={metrics.get('mae', float('nan')):.4f} "
        f"mape={metrics.get('mape', float('nan')):.2f}%"
    )
    return artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ARIMA forecaster for one ticker.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=730,
        help="Days of history to train on (default: 730 ≈ 2 years)",
    )
    parser.add_argument(
        "--order",
        type=_parse_order,
        default="auto",
        help='ARIMA order: "auto" for AIC grid search, or "p,d,q" e.g. "5,1,0"',
    )
    args = parser.parse_args()

    configure_logging()
    train(args.ticker.upper(), args.lookback_days, args.order)


if __name__ == "__main__":
    main()
