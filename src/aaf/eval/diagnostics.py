"""Diagnostics for Gaussian mixture forecasts."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from aaf.data.synthetic import FloatArray

_EPS = 1e-12


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
