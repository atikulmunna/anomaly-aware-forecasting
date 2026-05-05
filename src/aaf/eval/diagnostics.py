"""Diagnostics for Gaussian mixture forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from aaf.data.synthetic import FloatArray
from aaf.eval.forecasting import MixtureForecast

_EPS = 1e-12


@dataclass(frozen=True)
class MixtureDiagnostics:
    entropy_mean: float
    entropy_min: float
    entropy_max: float
    normalized_entropy_mean: float
    component_mean_weights: list[float]
    active_components_1pct: int
    effective_components: float
    std: dict[str, float]
    mean_pairwise_distance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "entropy_mean": self.entropy_mean,
            "entropy_min": self.entropy_min,
            "entropy_max": self.entropy_max,
            "normalized_entropy_mean": self.normalized_entropy_mean,
            "component_mean_weights": self.component_mean_weights,
            "active_components_1pct": self.active_components_1pct,
            "effective_components": self.effective_components,
            "std": self.std,
            "mean_pairwise_distance": self.mean_pairwise_distance,
        }


def mixture_diagnostics(forecast: MixtureForecast) -> MixtureDiagnostics:
    """Return aggregate diagnostics for a Gaussian mixture forecast."""

    forecast.validate()
    entropy = mixture_entropy_values(forecast.weights)
    normalized_entropy = normalized_mixture_entropy_values(forecast.weights)
    return MixtureDiagnostics(
        entropy_mean=float(np.mean(entropy)),
        entropy_min=float(np.min(entropy)),
        entropy_max=float(np.max(entropy)),
        normalized_entropy_mean=float(np.mean(normalized_entropy)),
        component_mean_weights=component_mean_weights(forecast.weights).tolist(),
        active_components_1pct=active_component_count(forecast.weights, threshold=0.01),
        effective_components=effective_component_count(forecast.weights),
        std=std_summary(forecast.stds),
        mean_pairwise_distance=mean_pairwise_distance(forecast.means),
    )


def normalized_weights(weights: ArrayLike) -> FloatArray:
    """Return mixture weights normalized over the component axis."""

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("weights must have at least one dimension")
    if np.any(~np.isfinite(values)):
        raise ValueError("weights must be finite")
    if np.any(values < 0.0):
        raise ValueError("weights must be non-negative")
    totals = values.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("at least one mixture component must have positive weight")
    return np.asarray(values / totals, dtype=np.float64)


def mixture_entropy_values(weights: ArrayLike) -> FloatArray:
    """Return per-forecast entropy over mixture components."""

    probs = normalized_weights(weights)
    return np.asarray(-np.sum(probs * np.log(np.maximum(probs, _EPS)), axis=-1))


def normalized_mixture_entropy_values(weights: ArrayLike) -> FloatArray:
    """Return per-forecast entropy scaled to [0, 1] when M > 1."""

    probs = normalized_weights(weights)
    n_components = probs.shape[-1]
    if n_components == 1:
        return np.zeros(probs.shape[:-1], dtype=np.float64)
    return np.asarray(mixture_entropy_values(probs) / np.log(n_components), dtype=np.float64)


def component_mean_weights(weights: ArrayLike) -> FloatArray:
    """Return mean assignment probability per component."""

    probs = normalized_weights(weights)
    flat = probs.reshape(-1, probs.shape[-1])
    return np.asarray(np.mean(flat, axis=0), dtype=np.float64)


def active_component_count(weights: ArrayLike, *, threshold: float = 0.01) -> int:
    """Count components whose average assignment probability exceeds a threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return int(np.sum(component_mean_weights(weights) >= threshold))


def effective_component_count(weights: ArrayLike) -> float:
    """Return exp(entropy(mean component weights))."""

    means = component_mean_weights(weights)
    entropy = float(mixture_entropy_values(means))
    return float(np.exp(entropy))


def std_summary(stds: ArrayLike) -> dict[str, float]:
    """Summarize predicted component standard deviations."""

    values = np.asarray(stds, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("stds must have at least one dimension")
    if np.any(~np.isfinite(values)):
        raise ValueError("stds must be finite")
    if np.any(values <= 0.0):
        raise ValueError("stds must be strictly positive")
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def mean_pairwise_distance(means: ArrayLike) -> float:
    """Return average pairwise Euclidean distance between component means."""

    values = np.asarray(means, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("means must include component and channel dimensions")
    if np.any(~np.isfinite(values)):
        raise ValueError("means must be finite")
    n_components = values.shape[-2]
    if n_components < 2:
        return 0.0

    flat = values.reshape(-1, n_components, values.shape[-1])
    distances = []
    for first in range(n_components):
        for second in range(first + 1, n_components):
            distances.append(np.linalg.norm(flat[:, first, :] - flat[:, second, :], axis=-1))
    return float(np.mean(np.concatenate(distances)))
