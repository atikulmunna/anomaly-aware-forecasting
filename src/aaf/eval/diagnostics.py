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
