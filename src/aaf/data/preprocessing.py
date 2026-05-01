"""Deterministic preprocessing and window construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from aaf.data.synthetic import FloatArray, IntArray, SyntheticSeries


@dataclass(frozen=True)
class Standardizer:
    """Per-channel z-score standardizer fitted on training data only."""

    mean: FloatArray
    std: FloatArray

    @classmethod
    def fit(cls, train_observations: ArrayLike, *, std_floor: float = 1e-6) -> Standardizer:
        if std_floor <= 0.0:
            raise ValueError("std_floor must be positive")
        values = _observation_array(train_observations)
        std = np.maximum(np.std(values, axis=0), std_floor)
        return cls(mean=np.mean(values, axis=0), std=std)

    def transform(self, observations: ArrayLike) -> FloatArray:
        values = _observation_array(observations)
        if values.shape[1] != self.mean.shape[0]:
            raise ValueError("observation channel count must match standardizer")
        return np.asarray((values - self.mean) / self.std, dtype=np.float64)

    def inverse_transform(self, observations: ArrayLike) -> FloatArray:
        values = _observation_array(observations)
        if values.shape[1] != self.mean.shape[0]:
            raise ValueError("observation channel count must match standardizer")
        return np.asarray(values * self.std + self.mean, dtype=np.float64)


@dataclass(frozen=True)
class WindowedDataset:
    """Array-backed dataset exposing window/target/label tuples."""

    windows: FloatArray
    targets: FloatArray
    regime_labels: IntArray
    anomaly_labels: IntArray

    def __post_init__(self) -> None:
        if self.windows.ndim != 3:
            raise ValueError("windows must have shape (N, L, D)")
        if self.targets.ndim != 3:
            raise ValueError("targets must have shape (N, H, D)")
        if self.windows.shape[0] != self.targets.shape[0]:
            raise ValueError("windows and targets must contain the same number of examples")
        if self.windows.shape[2] != self.targets.shape[2]:
            raise ValueError("windows and targets must have the same channel count")
        if self.regime_labels.shape != (self.windows.shape[0],):
            raise ValueError("regime_labels must have shape (N,)")
        if self.anomaly_labels.shape != (self.windows.shape[0],):
            raise ValueError("anomaly_labels must have shape (N,)")

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> tuple[FloatArray, FloatArray, int, int]:
        return (
            self.windows[index],
            self.targets[index],
            int(self.regime_labels[index]),
            int(self.anomaly_labels[index]),
        )


def standardize_series(series: SyntheticSeries, standardizer: Standardizer) -> SyntheticSeries:
    """Apply a fitted standardizer without changing labels."""

    standardized = SyntheticSeries(
        observations=standardizer.transform(series.observations),
        regime_labels=series.regime_labels.copy(),
        anomaly_labels=series.anomaly_labels.copy(),
        config_id=series.config_id,
    )
    standardized.validate()
    return standardized


def make_windowed_dataset(
    series: SyntheticSeries,
    *,
    lookback: int,
    horizon: int,
    stride: int = 1,
) -> WindowedDataset:
    """Create deterministic sliding windows from a synthetic series."""

    series.validate()
    if lookback < 1:
        raise ValueError("lookback must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")

    max_start = series.observations.shape[0] - lookback - horizon + 1
    if max_start < 1:
        raise ValueError("series is too short for requested lookback and horizon")

    starts = np.arange(0, max_start, stride, dtype=np.int64)
    windows = np.stack(
        [series.observations[start : start + lookback] for start in starts],
        axis=0,
    )
    targets = np.stack(
        [series.observations[start + lookback : start + lookback + horizon] for start in starts],
        axis=0,
    )
    label_index = starts + lookback - 1
    target_slices = (
        series.anomaly_labels[start + lookback : start + lookback + horizon] for start in starts
    )
    anomaly_labels = np.fromiter((int(np.any(labels)) for labels in target_slices), dtype=np.int64)
    return WindowedDataset(
        windows=np.asarray(windows, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        regime_labels=np.asarray(series.regime_labels[label_index], dtype=np.int64),
        anomaly_labels=anomaly_labels,
    )


def _observation_array(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2:
        raise ValueError("observations must have shape (T,) or (T, D)")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("observations must be non-empty")
    if np.any(~np.isfinite(array)):
        raise ValueError("observations must be finite")
    return array
