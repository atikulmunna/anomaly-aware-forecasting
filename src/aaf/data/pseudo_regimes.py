"""Pseudo-regime labels for datasets without explicit state annotations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from aaf.data.preprocessing import WindowedDataset
from aaf.data.synthetic import FloatArray, IntArray


@dataclass(frozen=True)
class PseudoRegimeModel:
    """Train-fitted feature scaler and centroids for pseudo-regime assignment."""

    feature_mean: FloatArray
    feature_std: FloatArray
    centroids: FloatArray

    def predict(self, windows: ArrayLike) -> IntArray:
        """Assign each window to the nearest fitted pseudo-regime centroid."""

        features = transform_pseudo_regime_features(windows, self.feature_mean, self.feature_std)
        return nearest_centroid_labels(features, self.centroids)


def assign_pseudo_regime_labels(
    train: WindowedDataset,
    validation: WindowedDataset,
    test: WindowedDataset,
    *,
    n_regimes: int,
    seed: int = 0,
    max_iter: int = 50,
) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset, PseudoRegimeModel]:
    """Fit pseudo-regimes on train windows and apply them to all splits."""

    model = fit_pseudo_regime_model(
        train.windows,
        n_regimes=n_regimes,
        seed=seed,
        max_iter=max_iter,
    )
    return (
        replace_regime_labels(train, model.predict(train.windows)),
        replace_regime_labels(validation, model.predict(validation.windows)),
        replace_regime_labels(test, model.predict(test.windows)),
        model,
    )


def fit_pseudo_regime_model(
    windows: ArrayLike,
    *,
    n_regimes: int,
    seed: int = 0,
    max_iter: int = 50,
) -> PseudoRegimeModel:
    """Fit deterministic k-means pseudo-regime centroids from training windows."""

    if n_regimes < 2:
        raise ValueError("n_regimes must be at least 2")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    raw_features = pseudo_regime_features(windows)
    if raw_features.shape[0] < n_regimes:
        raise ValueError("at least n_regimes windows are required")

    feature_mean = np.mean(raw_features, axis=0)
    feature_std = np.maximum(np.std(raw_features, axis=0), 1e-6)
    features = np.asarray((raw_features - feature_mean) / feature_std, dtype=np.float64)
    centroids = _kmeans(features, n_clusters=n_regimes, seed=seed, max_iter=max_iter)
    return PseudoRegimeModel(
        feature_mean=np.asarray(feature_mean, dtype=np.float64),
        feature_std=np.asarray(feature_std, dtype=np.float64),
        centroids=centroids,
    )


def pseudo_regime_features(windows: ArrayLike) -> FloatArray:
    """Summarize each time window into simple level, variability, and trend features."""

    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("windows must have shape (N, L, D)")
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("windows must be non-empty")
    if np.any(~np.isfinite(values)):
        raise ValueError("windows must be finite")

    mean = np.mean(values, axis=1)
    std = np.std(values, axis=1)
    last = values[:, -1, :]
    delta = values[:, -1, :] - values[:, 0, :]
    return np.asarray(np.concatenate([mean, std, last, delta], axis=-1), dtype=np.float64)


def transform_pseudo_regime_features(
    windows: ArrayLike,
    feature_mean: ArrayLike,
    feature_std: ArrayLike,
) -> FloatArray:
    """Transform windows with train-fitted pseudo-regime feature statistics."""

    features = pseudo_regime_features(windows)
    mean = np.asarray(feature_mean, dtype=np.float64)
    std = np.asarray(feature_std, dtype=np.float64)
    if mean.shape != (features.shape[1],) or std.shape != (features.shape[1],):
        raise ValueError("feature scaler shape must match pseudo-regime features")
    if np.any(std <= 0.0):
        raise ValueError("feature std values must be positive")
    return np.asarray((features - mean) / std, dtype=np.float64)


def nearest_centroid_labels(features: ArrayLike, centroids: ArrayLike) -> IntArray:
    """Return nearest-centroid labels for already transformed features."""

    values = np.asarray(features, dtype=np.float64)
    centers = np.asarray(centroids, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("features must have shape (N, F)")
    if centers.ndim != 2:
        raise ValueError("centroids must have shape (K, F)")
    if values.shape[1] != centers.shape[1]:
        raise ValueError("features and centroids must have matching dimensions")
    distances = np.sum((values[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=-1)
    return np.asarray(np.argmin(distances, axis=-1), dtype=np.int64)


def replace_regime_labels(dataset: WindowedDataset, regime_labels: ArrayLike) -> WindowedDataset:
    """Return a dataset copy with replaced regime labels."""

    labels = np.asarray(regime_labels, dtype=np.int64)
    if labels.shape != dataset.regime_labels.shape:
        raise ValueError("regime_labels must match dataset length")
    return WindowedDataset(
        windows=dataset.windows,
        targets=dataset.targets,
        regime_labels=labels,
        anomaly_labels=dataset.anomaly_labels,
    )


def _kmeans(
    features: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
    max_iter: int,
) -> FloatArray:
    rng = np.random.default_rng(seed)
    centroids = _initial_centroids(features, n_clusters=n_clusters, rng=rng)
    labels = np.full(features.shape[0], -1, dtype=np.int64)
    for _ in range(max_iter):
        next_labels = nearest_centroid_labels(features, centroids)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        centroids = _updated_centroids(features, labels, centroids)
    return np.asarray(centroids, dtype=np.float64)


def _initial_centroids(
    features: np.ndarray,
    *,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = [int(rng.integers(features.shape[0]))]
    while len(selected) < n_clusters:
        current = features[np.asarray(selected, dtype=np.int64)]
        distances = np.min(
            np.sum((features[:, np.newaxis, :] - current[np.newaxis, :, :]) ** 2, axis=-1),
            axis=1,
        )
        distances[np.asarray(selected, dtype=np.int64)] = -1.0
        selected.append(int(np.argmax(distances)))
    return np.asarray(features[np.asarray(selected, dtype=np.int64)].copy(), dtype=np.float64)


def _updated_centroids(
    features: np.ndarray,
    labels: np.ndarray,
    previous_centroids: np.ndarray,
) -> np.ndarray:
    centroids = np.asarray(previous_centroids.copy(), dtype=np.float64)
    for cluster_id in range(previous_centroids.shape[0]):
        members = features[labels == cluster_id]
        if members.shape[0] > 0:
            centroids[cluster_id] = np.mean(members, axis=0)
    return np.asarray(centroids, dtype=np.float64)
