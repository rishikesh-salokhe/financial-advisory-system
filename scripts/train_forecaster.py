"""
Train and persist a forecasting model for a single ticker.

Usage:
    # ARIMA (default)
    python scripts/train_forecaster.py --ticker AAPL
    python scripts/train_forecaster.py --ticker AAPL --order 5,1,0
    python scripts/train_forecaster.py --ticker AAPL --lookback-days 1095 --order auto

    # LSTM
    python scripts/train_forecaster.py --ticker AAPL --model lstm
    python scripts/train_forecaster.py --ticker AAPL --model lstm --epochs 80 --window 90

The trained artifact is saved to:
    models/forecasting/{TICKER}_{MODEL}.joblib       (sidecar)
    models/forecasting/{TICKER}_{MODEL}.keras        (LSTM only — Keras model)

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


MODEL_DIR = PROJECT_ROOT / "models" / "forecasting"


# ─── Argument parsing helpers ─────────────────────────────────────────────


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


# ─── Model-specific trainers ──────────────────────────────────────────────


def _train_arima(ticker: str, lookback_days: int, order: tuple[int, int, int] | str) -> Path:
    """Fetch history, fit ARIMA, persist artifact. Returns the artifact path."""
    # Imported lazily so the LSTM path doesn't pay statsmodels' import cost.
    from ml_engine.forecasting.arima_model import ARIMAForecaster

    logger.info(f"Training ARIMA for {ticker} (lookback={lookback_days}d, order={order})")

    series = fetch_close_series(ticker, lookback_days=lookback_days)
    if series.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    logger.info(
        f"Loaded {len(series)} close prices "
        f"({series.index[0].date()} → {series.index[-1].date()})"
    )

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


def _train_lstm(
    ticker: str,
    lookback_days: int,
    window: int,
    epochs: int,
    batch_size: int,
    mc_samples: int,
) -> Path:
    """Fetch history, fit LSTM, persist artifact. Returns the sidecar path."""
    # Imported lazily — TF startup is ~3s and we don't want to pay it for ARIMA runs.
    from ml_engine.forecasting.lstm_model import LSTMForecaster

    logger.info(
        f"Training LSTM for {ticker} "
        f"(lookback={lookback_days}d, window={window}, epochs={epochs}, "
        f"batch={batch_size}, mc_samples={mc_samples})"
    )

    series = fetch_close_series(ticker, lookback_days=lookback_days)
    if series.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    logger.info(
        f"Loaded {len(series)} close prices "
        f"({series.index[0].date()} → {series.index[-1].date()})"
    )

    forecaster = LSTMForecaster(
        lookback=window,
        epochs=epochs,
        batch_size=batch_size,
        mc_samples=mc_samples,
    )
    forecaster.fit(series)

    artifact_path = MODEL_DIR / f"{ticker.upper()}_lstm.joblib"
    forecaster.save(artifact_path)

    metrics = forecaster.metrics
    logger.info(
        f"Done. "
        f"rmse={metrics.get('rmse', float('nan')):.4f} "
        f"mae={metrics.get('mae', float('nan')):.4f} "
        f"mape={metrics.get('mape', float('nan')):.2f}% "
        f"val_loss={metrics.get('val_loss', float('nan')):.6f}"
    )
    return artifact_path


# ─── CLI entrypoint ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a forecasting model (ARIMA or LSTM) for one ticker.",
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--model",
        choices=["arima", "lstm"],
        default="arima",
        help="Which model to train (default: arima)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=730,
        help="Days of history to fetch for training (default: 730 ≈ 2 years)",
    )

    # ARIMA-only
    parser.add_argument(
        "--order",
        type=_parse_order,
        default="auto",
        help='ARIMA only — order: "auto" for AIC grid search, or "p,d,q" e.g. "5,1,0"',
    )

    # LSTM-only
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="LSTM only — sliding-window length in trading days (default: 60)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="LSTM only — max training epochs; EarlyStopping may end sooner (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="LSTM only — mini-batch size (default: 32)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=50,
        help="LSTM only — MC-Dropout passes per horizon step at inference (default: 50)",
    )

    args = parser.parse_args()

    configure_logging()
    ticker = args.ticker.upper()

    if args.model == "arima":
        _train_arima(ticker, args.lookback_days, args.order)
    else:
        _train_lstm(
            ticker=ticker,
            lookback_days=args.lookback_days,
            window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            mc_samples=args.mc_samples,
        )


if __name__ == "__main__":
    main()
