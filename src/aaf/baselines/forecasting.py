"""Simple forecasting baselines used by the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aaf.eval.forecasting import MixtureForecast

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SeasonalNaiveForecaster:
    """Seasonal naive probabilistic baseline with diagonal Gaussian emissions."""

    season_length: int
    residual_std: FloatArray
    std_floor: float = 1e-3

    @classmethod
    def fit(
        cls,
        series: ArrayLike,
        *,
        season_length: int,
        std_floor: float = 1e-3,
    ) -> SeasonalNaiveForecaster:
        if season_length < 1:
            raise ValueError("season_length must be positive")
        if std_floor <= 0.0:
            raise ValueError("std_floor must be positive")

        values = _series_array(series)
        if values.shape[0] <= season_length:
            residual_std = np.ones(values.shape[1], dtype=np.float64)
        else:
            residuals = values[season_length:] - values[:-season_length]
            residual_std = np.std(residuals, axis=0)
        residual_std = np.maximum(residual_std, std_floor)
        return cls(
            season_length=season_length,
            residual_std=np.asarray(residual_std, dtype=np.float64),
            std_floor=std_floor,
        )

    def predict(self, history: ArrayLike, *, horizon: int) -> MixtureForecast:
        """Predict future values from batched history with shape (B, L, D)."""

        if horizon < 1:
            raise ValueError("horizon must be positive")

        history_array = _history_array(history)
        if history_array.shape[1] < self.season_length:
            raise ValueError("history length must be at least season_length")
        if history_array.shape[2] != self.residual_std.shape[0]:
            raise ValueError("history channel count must match fitted residual_std")

        batch_size, _, n_channels = history_array.shape
        means = np.empty((batch_size, horizon, 1, n_channels), dtype=np.float64)
        seasonal_source = history_array[:, -self.season_length :, :]
        for step in range(horizon):
            means[:, step, 0, :] = seasonal_source[:, step % self.season_length, :]

        weights = np.ones((batch_size, horizon, 1), dtype=np.float64)
        stds = np.broadcast_to(
            self.residual_std.reshape(1, 1, 1, n_channels),
            means.shape,
        ).copy()
        return MixtureForecast.from_arrays(weights=weights, means=means, stds=stds)


def persistence_forecast(history: ArrayLike, *, horizon: int, std: float = 1.0) -> MixtureForecast:
    """Convenience persistence baseline with a fixed predictive std."""

    if std <= 0.0:
        raise ValueError("std must be positive")
    history_array = _history_array(history)
    forecaster = SeasonalNaiveForecaster(
        season_length=1,
        residual_std=np.full(history_array.shape[2], std, dtype=np.float64),
    )
    return forecaster.predict(history_array, horizon=horizon)


def _series_array(series: ArrayLike) -> FloatArray:
    values = np.asarray(series, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError("series must have shape (T,) or (T, D)")
    if values.shape[0] == 0:
        raise ValueError("series must contain at least one timestep")
    if np.any(~np.isfinite(values)):
        raise ValueError("series must be finite")
    return values


def _history_array(history: ArrayLike) -> FloatArray:
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("history must have shape (B, L, D)")
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("history dimensions must be non-empty")
    if np.any(~np.isfinite(values)):
        raise ValueError("history must be finite")
    return values
