"""
Forecaster interface.

Every concrete forecaster (ARIMA, LSTM, Prophet, etc.) implements ``Forecaster``.
The service layer interacts with the abstraction so models are interchangeable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    ticker: str
    model_name: str
    horizon_days: int
    predictions: list[float]
    dates: list[date]
    lower_ci: list[float] | None = None
    upper_ci: list[float] | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class Forecaster(ABC):
    """Abstract base class for all forecasters."""

    name: str = "abstract"

    @abstractmethod
    def fit(self, series: pd.Series) -> "Forecaster":
        """Train the model on a univariate time series indexed by date."""

    @abstractmethod
    def predict(self, horizon: int, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (point_forecast, lower_ci, upper_ci) arrays of length ``horizon``."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Forecaster":
        """Reconstruct a trained model from disk."""


class ForecasterFactory(Protocol):
    """Type alias used by the registry."""

    def __call__(self) -> Forecaster: ...
