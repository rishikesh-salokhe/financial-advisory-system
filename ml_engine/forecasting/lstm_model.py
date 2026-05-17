"""
LSTM forecaster (Keras / TensorFlow).

Stub: the public surface mirrors ARIMAForecaster so the service layer can use
them interchangeably via :class:`ml_engine.forecasting.base.Forecaster`. The
training loop, scaling pipeline, and confidence-interval estimation (MC dropout
or bootstrap) should be filled in during the modeling phase. See
``notebooks/03_lstm_modeling.ipynb``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_engine.forecasting.base import Forecaster


class LSTMForecaster(Forecaster):
    name = "lstm"

    def __init__(
        self,
        sequence_length: int = 60,
        units: int = 64,
        epochs: int = 25,
        batch_size: int = 32,
        dropout: float = 0.2,
    ) -> None:
        self.sequence_length = sequence_length
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self._model = None
        self._scaler = None
        self._history: pd.Series | None = None

    def fit(self, series: pd.Series) -> "LSTMForecaster":
        # TODO:
        # 1. Scale with MinMaxScaler in [0, 1]
        # 2. Build sliding windows of length ``sequence_length``
        # 3. Compile a Sequential([LSTM, Dropout, Dense]) and fit
        # 4. Persist scaler + model
        raise NotImplementedError("LSTMForecaster.fit is not yet implemented")

    def predict(
        self, horizon: int, confidence: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: recursive multi-step forecast with MC-dropout for CIs.
        raise NotImplementedError("LSTMForecaster.predict is not yet implemented")

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "LSTMForecaster":
        raise NotImplementedError
