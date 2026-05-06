"""Diagnostics for regime posterior sequences."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from aaf.data.synthetic import FloatArray

_EPS = 1e-12


def regime_posterior_entropy_values(probs: ArrayLike) -> FloatArray:
    """Return entropy of regime posterior probabilities at each timestep."""

    values = _posterior_array(probs)
    return np.asarray(-np.sum(values * np.log(np.maximum(values, _EPS)), axis=-1))


def normalized_regime_entropy_values(probs: ArrayLike) -> FloatArray:
    """Return regime entropy normalized to [0, 1]."""

    values = _posterior_array(probs)
    if values.shape[-1] == 1:
        return np.zeros(values.shape[:-1], dtype=np.float64)
    return np.asarray(regime_posterior_entropy_values(values) / np.log(values.shape[-1]))


def _posterior_array(probs: ArrayLike) -> FloatArray:
    values = np.asarray(probs, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("posterior probabilities must include time and regime dimensions")
    if np.any(~np.isfinite(values)):
        raise ValueError("posterior probabilities must be finite")
    if np.any(values < 0.0):
        raise ValueError("posterior probabilities must be non-negative")
    if not np.allclose(values.sum(axis=-1), 1.0):
        raise ValueError("posterior probabilities must sum to 1 over regimes")
    return values
