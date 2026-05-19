"""
LSTM forecaster — production-ready wrapper around Keras/TensorFlow with
MC-Dropout confidence intervals.

Capabilities:
    * Stacked LSTM (2 layers) on a fixed-length sliding window of MinMax-scaled
      close prices. Recursive single-step rollout for any forecast horizon.
    * MC-Dropout at inference (K stochastic passes per horizon step) gives
      proper prediction-interval estimates — the band naturally fans out as the
      horizon grows.
    * Out-of-sample metrics (RMSE / MAE / MAPE) captured at fit time, evaluated
      on the chronological validation split so they reflect real generalization.
    * Two-file artifact:
        ``{TICKER}_lstm.keras``   — Keras model (native format)
        ``{TICKER}_lstm.joblib``  — sidecar with scaler + metadata + metrics

The service layer keys discovery off the ``.joblib`` sidecar; the loader
follows the convention to find the matching ``.keras`` file.

Design notes
------------
* TensorFlow is imported lazily inside methods so importing this module is
  cheap (the test suite, REPL, and FastAPI startup don't pay TF's ~3s import
  cost unless a forecast actually needs an LSTM).
* MC-Dropout is implemented by passing ``training=True`` to the model call —
  no special dropout layer subclass needed.
* The full last-known scaled window is persisted so inference can start
  immediately after ``load()`` without re-fetching the training data.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from ml_engine.forecasting.base import Forecaster


# ─── Constants ────────────────────────────────────────────────────────────

DEFAULT_LOOKBACK: int = 60
DEFAULT_UNITS: tuple[int, int] = (64, 32)
DEFAULT_DROPOUT: float = 0.2
DEFAULT_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_VAL_SPLIT: float = 0.2
DEFAULT_MC_SAMPLES: int = 50

# Quiet TF startup chatter unless the user has already configured it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ─── Dataclass: training artifact ─────────────────────────────────────────


@dataclass
class _LSTMArtifact:
    """In-memory shape of the joblib sidecar. Keras model lives in a separate file."""

    scaler: object                      # fitted sklearn MinMaxScaler
    lookback: int
    units: tuple[int, int]
    dropout: float
    last_window_scaled: np.ndarray      # shape (lookback,), float32
    metrics: dict[str, float]
    last_training_date: date
    training_series_name: str
    training_n_obs: int
    epochs_trained: int
    mc_samples: int = DEFAULT_MC_SAMPLES
    # Reserved for future feature engineering — placeholder for additional channels.
    extra: dict = field(default_factory=dict)


# ─── Forecaster ───────────────────────────────────────────────────────────


class LSTMForecaster(Forecaster):
    """Stacked LSTM forecaster with MC-Dropout confidence intervals."""

    name = "lstm"

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        units: tuple[int, int] = DEFAULT_UNITS,
        dropout: float = DEFAULT_DROPOUT,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split: float = DEFAULT_VAL_SPLIT,
        mc_samples: int = DEFAULT_MC_SAMPLES,
        seed: int | None = 42,
    ) -> None:
        if lookback < 10:
            raise ValueError("lookback must be >= 10")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if mc_samples < 2:
            raise ValueError("mc_samples must be >= 2 for a meaningful CI")

        self._lookback = lookback
        self._units = units
        self._dropout = dropout
        self._epochs = epochs
        self._batch_size = batch_size
        self._val_split = val_split
        self._mc_samples = mc_samples
        self._seed = seed

        self._model = None                            # set in fit() / load()
        self._artifact: _LSTMArtifact | None = None

    # ─── Properties (Forecaster contract) ─────────────────────────────────

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self._artifact.metrics) if self._artifact else {}

    @property
    def last_training_date(self) -> date | None:
        return self._artifact.last_training_date if self._artifact else None

    # ─── Core lifecycle ───────────────────────────────────────────────────

    def fit(self, series: pd.Series) -> "LSTMForecaster":
        """Train the LSTM on ``series`` (univariate, date-indexed close prices)."""
        if series.empty:
            raise ValueError("Cannot fit LSTM on an empty series")
        if len(series) < self._lookback + 30:
            raise ValueError(
                f"Series too short ({len(series)} obs); need at least "
                f"{self._lookback + 30} for a {self._lookback}-day window"
            )

        clean = series.astype(float).dropna().sort_index()
        self._seed_everything()

        # ── Scale ────────────────────────────────────────────────────────
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        values = clean.values.reshape(-1, 1).astype(np.float32)
        scaled = scaler.fit_transform(values).flatten()

        # ── Build sliding windows ───────────────────────────────────────
        X, y = self._make_windows(scaled, self._lookback)
        n_total = len(X)
        n_train = int(n_total * (1 - self._val_split))
        if n_train < self._lookback or n_total - n_train < 5:
            raise ValueError(
                f"Train/val split too small (train={n_train}, val={n_total - n_train})"
            )

        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        logger.info(
            f"LSTM data prepared: lookback={self._lookback}, "
            f"train_windows={len(X_train)}, val_windows={len(X_val)}"
        )

        # ── Build + train ───────────────────────────────────────────────
        model = self._build_model()
        history = self._train_model(model, X_train, y_train, X_val, y_val)

        # ── Validation metrics in original USD units ────────────────────
        metrics = self._compute_metrics(model, X_val, y_val, scaler, history)

        # ── Persist artifact (sidecar only — Keras model is saved separately) ─
        self._model = model
        self._artifact = _LSTMArtifact(
            scaler=scaler,
            lookback=self._lookback,
            units=self._units,
            dropout=self._dropout,
            last_window_scaled=scaled[-self._lookback:].astype(np.float32),
            metrics=metrics,
            last_training_date=pd.Timestamp(clean.index[-1]).date(),
            training_series_name=str(clean.name or "series"),
            training_n_obs=len(clean),
            epochs_trained=len(history.history.get("loss", [])),
            mc_samples=self._mc_samples,
        )
        logger.info(
            f"LSTM fit complete: epochs={self._artifact.epochs_trained}/{self._epochs}, "
            f"rmse={metrics['rmse']:.4f}, mape={metrics['mape']:.2f}%"
        )
        return self

    def predict(
        self, horizon: int, confidence: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recursive multi-step forecast with MC-Dropout CI bands.

        Returns ``(point_forecast, lower_ci, upper_ci)`` — each a 1-D
        ``numpy.ndarray`` of length ``horizon`` in original price units.
        """
        if self._model is None or self._artifact is None:
            raise RuntimeError("LSTMForecaster.predict() called before fit()/load()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not (0 < confidence < 1):
            raise ValueError("confidence must be in (0, 1)")

        K = self._artifact.mc_samples
        alpha = 1.0 - confidence
        seed_window = self._artifact.last_window_scaled.copy()

        # K independent stochastic rollouts → shape (K, horizon)
        paths = np.empty((K, horizon), dtype=np.float32)
        for k in range(K):
            window = seed_window.copy()
            for h in range(horizon):
                x_in = window.reshape(1, self._lookback, 1)
                # training=True keeps dropout active → stochastic forward pass
                y_scaled = float(self._model(x_in, training=True).numpy().squeeze())
                paths[k, h] = y_scaled
                window = np.append(window[1:], y_scaled)

        mean_scaled = paths.mean(axis=0)
        lower_scaled = np.quantile(paths, alpha / 2, axis=0)
        upper_scaled = np.quantile(paths, 1 - alpha / 2, axis=0)

        # Invert MinMax scaling back to USD
        scaler = self._artifact.scaler
        mean = scaler.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
        lower = scaler.inverse_transform(lower_scaled.reshape(-1, 1)).flatten()
        upper = scaler.inverse_transform(upper_scaled.reshape(-1, 1)).flatten()

        # MC quantiles aren't guaranteed monotone vs the mean — clamp for plotting.
        lower = np.minimum(lower, mean)
        upper = np.maximum(upper, mean)
        return mean.astype(float), lower.astype(float), upper.astype(float)

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save sidecar (``.joblib``) + Keras model (``.keras``)."""
        if self._artifact is None or self._model is None:
            raise RuntimeError("Nothing to save — call fit() first")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keras_path = path.with_suffix(".keras")

        # Keras 3 / TF 2.16+ uses ``.keras`` zip format — robust and version-safe.
        self._model.save(str(keras_path))
        joblib.dump(self._artifact, path)

        logger.info(f"Saved LSTM model to {keras_path}")
        logger.info(f"Saved LSTM sidecar to {path}")

    @classmethod
    def load(cls, path: Path) -> "LSTMForecaster":
        """Reconstruct from the ``.joblib`` sidecar + sibling ``.keras`` model."""
        path = Path(path)
        keras_path = path.with_suffix(".keras")
        if not keras_path.exists():
            raise FileNotFoundError(f"Expected Keras model file at {keras_path}")

        artifact: _LSTMArtifact = joblib.load(path)

        # Lazy TF import (matches fit())
        from tensorflow import keras

        model = keras.models.load_model(str(keras_path), compile=False)

        instance = cls(
            lookback=artifact.lookback,
            units=artifact.units,
            dropout=artifact.dropout,
            mc_samples=artifact.mc_samples,
        )
        instance._model = model
        instance._artifact = artifact

        logger.info(
            f"Loaded LSTM artifact from {path} "
            f"(lookback={artifact.lookback}, last_training_date={artifact.last_training_date})"
        )
        return instance

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _seed_everything(self) -> None:
        if self._seed is None:
            return
        import random

        import tensorflow as tf

        random.seed(self._seed)
        np.random.seed(self._seed)
        tf.random.set_seed(self._seed)

    @staticmethod
    def _make_windows(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        """Construct sliding (X, y) pairs from a 1-D scaled series."""
        n = len(series) - lookback
        X = np.empty((n, lookback, 1), dtype=np.float32)
        y = np.empty((n,), dtype=np.float32)
        for i in range(n):
            X[i, :, 0] = series[i : i + lookback]
            y[i] = series[i + lookback]
        return X, y

    def _build_model(self):
        """Construct the stacked-LSTM Keras model."""
        from tensorflow.keras import Input, Sequential, layers, optimizers

        u1, u2 = self._units
        model = Sequential(
            [
                Input(shape=(self._lookback, 1)),
                layers.LSTM(u1, return_sequences=True),
                layers.Dropout(self._dropout),
                layers.LSTM(u2),
                layers.Dropout(self._dropout),
                layers.Dense(1),
            ],
            name="lstm_forecaster",
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def _train_model(self, model, X_train, y_train, X_val, y_val):
        """Fit with EarlyStopping + ReduceLROnPlateau, return the Keras History."""
        from tensorflow.keras import callbacks

        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True, verbose=0
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=0
            ),
        ]
        return model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self._epochs,
            batch_size=self._batch_size,
            callbacks=cb,
            verbose=0,
            shuffle=False,  # time-series: don't shuffle within epoch
        )

    @staticmethod
    def _compute_metrics(model, X_val, y_val, scaler, history) -> dict[str, float]:
        """RMSE / MAE / MAPE on the held-out validation split, in original units."""
        if len(X_val) == 0:
            return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}

        # Deterministic forward pass for metrics (no dropout sampling).
        y_pred_scaled = model.predict(X_val, verbose=0).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

        residuals = y_pred - y_true
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))
        nonzero = y_true != 0
        mape = (
            float(np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100)
            if nonzero.any()
            else float("nan")
        )

        final_val_loss = float(history.history.get("val_loss", [float("nan")])[-1])

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "val_loss": final_val_loss,
        }
