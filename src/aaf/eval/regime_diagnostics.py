"""Diagnostics for regime posterior sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from aaf.data.synthetic import FloatArray

_EPS = 1e-12


@dataclass(frozen=True)
class RegimePosteriorDiagnostics:
    entropy_mean: float
    normalized_entropy_mean: float
    confidence_mean: float
    switch_count: int
    n_regimes: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "entropy_mean": self.entropy_mean,
            "normalized_entropy_mean": self.normalized_entropy_mean,
            "confidence_mean": self.confidence_mean,
            "switch_count": self.switch_count,
            "n_regimes": self.n_regimes,
        }


def regime_posterior_diagnostics(probs: ArrayLike) -> RegimePosteriorDiagnostics:
    """Return aggregate diagnostics for regime posterior probabilities."""

    values = _posterior_array(probs)
    return RegimePosteriorDiagnostics(
        entropy_mean=float(np.mean(regime_posterior_entropy_values(values))),
        normalized_entropy_mean=float(np.mean(normalized_regime_entropy_values(values))),
        confidence_mean=mean_regime_confidence(values),
        switch_count=posterior_switch_count(values),
        n_regimes=int(values.shape[-1]),
    )


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


def regime_confidence_values(probs: ArrayLike) -> FloatArray:
    """Return maximum posterior probability at each timestep."""

    values = _posterior_array(probs)
    return np.asarray(np.max(values, axis=-1), dtype=np.float64)


def mean_regime_confidence(probs: ArrayLike) -> float:
    """Return mean maximum posterior probability."""

    return float(np.mean(regime_confidence_values(probs)))


def posterior_argmax_labels(probs: ArrayLike) -> np.ndarray:
    """Return argmax regime labels from posterior probabilities."""

    values = _posterior_array(probs)
    return np.asarray(np.argmax(values, axis=-1), dtype=np.int64)


def posterior_switch_count(probs: ArrayLike) -> int:
    """Count argmax regime switches in posterior sequence."""

    labels = posterior_argmax_labels(probs).reshape(-1)
    if labels.size < 2:
        return 0
    return int(np.sum(labels[1:] != labels[:-1]))


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
