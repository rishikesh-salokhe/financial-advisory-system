"""
ARIMA forecaster — production-ready wrapper around ``statsmodels`` SARIMAX.

Capabilities:
    * Manual order (``order=(p, d, q)``) or AIC grid search (``order="auto"``).
    * In-sample metrics (RMSE / MAE / MAPE) captured at fit time.
    * Joblib persistence keyed by ticker.
    * Returns mean + lower/upper confidence intervals for any forecast horizon.

Auto-order intentionally avoids ``pmdarima`` to keep the dependency tree lean
(pmdarima pins NumPy<2 and frequently breaks installs). A bounded grid search
over (p ∈ 0..3, d ∈ 0..2, q ∈ 0..3) covers >95% of real-world financial series.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from ml_engine.forecasting.base import Forecaster


# ─── Constants ────────────────────────────────────────────────────────────

DEFAULT_ORDER: tuple[int, int, int] = (5, 1, 0)
AUTO_ORDER_GRID = {
    "p": range(0, 4),
    "d": range(0, 3),
    "q": range(0, 4),
}


# ─── Dataclass: training artifact ─────────────────────────────────────────


@dataclass
class _ARIMAArtifact:
    """In-memory shape of what gets pickled to disk via joblib."""

    order: tuple[int, int, int]
    fitted_model: object
    metrics: dict[str, float]
    last_training_date: date
    training_series_name: str
    training_n_obs: int


# ─── Forecaster ───────────────────────────────────────────────────────────


class ARIMAForecaster(Forecaster):
    """ARIMA forecaster with optional auto-order selection."""

    name = "arima"

    def __init__(self, order: tuple[int, int, int] | str = DEFAULT_ORDER) -> None:
        self._requested_order = order
        self._artifact: _ARIMAArtifact | None = None

    # ─── Properties ───────────────────────────────────────────────────────

    @property
    def order(self) -> tuple[int, int, int]:
        if self._artifact is None:
            if isinstance(self._requested_order, tuple):
                return self._requested_order
            raise RuntimeError("Auto-order is resolved only after fit()")
        return self._artifact.order

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self._artifact.metrics) if self._artifact else {}

    @property
    def last_training_date(self) -> date | None:
        return self._artifact.last_training_date if self._artifact else None

    # ─── Core lifecycle ───────────────────────────────────────────────────

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        """Train ARIMA on ``series``. Series index must be sortable; values must be numeric."""
        if series.empty:
            raise ValueError("Cannot fit ARIMA on an empty series")
        if len(series) < 30:
            raise ValueError(f"Series too short ({len(series)} obs); need at least 30")

        clean = series.astype(float).dropna().sort_index()

        order = (
            self._search_order(clean)
            if isinstance(self._requested_order, str) and self._requested_order == "auto"
            else self._requested_order  # type: ignore[assignment]
        )

        fitted = self._fit_one(clean, order)
        metrics = self._compute_in_sample_metrics(clean, fitted)

        self._artifact = _ARIMAArtifact(
            order=order,
            fitted_model=fitted,
            metrics=metrics,
            last_training_date=pd.Timestamp(clean.index[-1]).date(),
            training_series_name=str(clean.name or "series"),
            training_n_obs=len(clean),
        )
        logger.info(
            f"ARIMA fit complete: order={order}, "
            f"n={len(clean)}, rmse={metrics['rmse']:.4f}, mape={metrics['mape']:.2f}%"
        )
        return self

    def predict(
        self, horizon: int, confidence: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(point, lower_ci, upper_ci)`` arrays for ``horizon`` future steps."""
        if self._artifact is None:
            raise RuntimeError("ARIMAForecaster.predict() called before fit()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not (0 < confidence < 1):
            raise ValueError("confidence must be in (0, 1)")

        alpha = 1 - confidence
        forecast = self._artifact.fitted_model.get_forecast(steps=horizon)
        mean = np.asarray(forecast.predicted_mean, dtype=float)
        ci = forecast.conf_int(alpha=alpha)
        lower = np.asarray(ci.iloc[:, 0], dtype=float)
        upper = np.asarray(ci.iloc[:, 1], dtype=float)
        return mean, lower, upper

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        if self._artifact is None:
            raise RuntimeError("Nothing to save — call fit() first")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._artifact, path)
        logger.info(f"Saved ARIMA artifact to {path}")

    @classmethod
    def load(cls, path: Path) -> "ARIMAForecaster":
        artifact: _ARIMAArtifact = joblib.load(path)
        instance = cls(order=artifact.order)
        instance._artifact = artifact
        logger.info(
            f"Loaded ARIMA artifact from {path} "
            f"(order={artifact.order}, last_training_date={artifact.last_training_date})"
        )
        return instance

    # ─── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _fit_one(series: pd.Series, order: tuple[int, int, int]):
        """Fit a single ARIMA(p,d,q). Returns a fitted statsmodels results object."""
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(series, order=order)
            return model.fit()

    def _search_order(self, series: pd.Series) -> tuple[int, int, int]:
        """Bounded AIC grid search. Returns the (p,d,q) with minimum AIC."""
        best_order: tuple[int, int, int] | None = None
        best_aic = float("inf")
        tried = 0
        failed = 0

        for p, d, q in product(AUTO_ORDER_GRID["p"], AUTO_ORDER_GRID["d"], AUTO_ORDER_GRID["q"]):
            if p == 0 and q == 0:
                continue  # degenerate
            try:
                fitted = self._fit_one(series, (p, d, q))
                tried += 1
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except Exception as exc:  # noqa: BLE001 — statsmodels raises many exception types
                failed += 1
                logger.debug(f"Auto-order skipped ({p},{d},{q}): {exc}")
                continue

        if best_order is None:
            logger.warning(f"Auto-order failed for all candidates ({failed} fits); falling back to {DEFAULT_ORDER}")
            return DEFAULT_ORDER

        logger.info(f"Auto-order: best={best_order} aic={best_aic:.2f} (tried={tried}, failed={failed})")
        return best_order

    @staticmethod
    def _compute_in_sample_metrics(series: pd.Series, fitted) -> dict[str, float]:
        """In-sample RMSE / MAE / MAPE against the fitted values."""
        try:
            preds = pd.Series(fitted.fittedvalues, index=series.index)
        except Exception:  # noqa: BLE001
            return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}

        aligned = pd.concat([series, preds], axis=1, join="inner").dropna()
        if aligned.empty:
            return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}

        y_true = aligned.iloc[:, 0].to_numpy()
        y_pred = aligned.iloc[:, 1].to_numpy()
        residuals = y_pred - y_true

        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))
        # Guard MAPE against zeros
        nonzero = y_true != 0
        mape = float(np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else float("nan")

        return {"rmse": rmse, "mae": mae, "mape": mape, "aic": float(fitted.aic)}
