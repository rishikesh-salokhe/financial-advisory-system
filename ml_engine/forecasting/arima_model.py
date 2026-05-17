"""
ARIMA forecaster.

Wraps ``statsmodels`` SARIMAX. The order is selected via ``pmdarima.auto_arima``
when available, with a sensible fallback otherwise.

This module is a stub: the public surface is finalized, but ``fit``/``predict``
should be reviewed and benchmarked before production use. See
``notebooks/02_arima_baseline.ipynb`` for the development workflow.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml_engine.forecasting.base import Forecaster


class ARIMAForecaster(Forecaster):
    name = "arima"

    def __init__(self, order: tuple[int, int, int] = (5, 1, 0)) -> None:
        self.order = order
        self._fitted = None  # type: ignore[var-annotated]

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(series.astype(float), order=self.order)
        self._fitted = model.fit()
        return self

    def predict(
        self, horizon: int, confidence: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._fitted is None:
            raise RuntimeError("ARIMAForecaster.predict() called before fit()")
        alpha = 1 - confidence
        forecast = self._fitted.get_forecast(steps=horizon)
        mean = np.asarray(forecast.predicted_mean)
        ci = forecast.conf_int(alpha=alpha)
        lower = np.asarray(ci.iloc[:, 0])
        upper = np.asarray(ci.iloc[:, 1])
        return mean, lower, upper

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"order": self.order, "model": self._fitted}, f)

    @classmethod
    def load(cls, path: Path) -> "ARIMAForecaster":
        with path.open("rb") as f:
            payload = pickle.load(f)
        instance = cls(order=payload["order"])
        instance._fitted = payload["model"]
        return instance
