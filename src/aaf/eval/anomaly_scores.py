"""Forecast-derived anomaly score aggregation."""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike

from aaf.eval.forecasting import (
    MixtureForecast,
    channelwise_negative_log_likelihood_values,
    negative_log_likelihood_values,
)

AnomalyScoreMethod = Literal[
    "mean_nll",
    "channel_mean_nll",
    "channel_max_nll",
    "channel_top3_mean_nll",
]

ANOMALY_SCORE_METHODS = get_args(AnomalyScoreMethod)
JOINT_ANOMALY_SCORE_METHODS = (
    *ANOMALY_SCORE_METHODS,
    "regime_entropy",
    "regime_confidence_gap",
    "regime_switch",
)


def validate_anomaly_score_method(method: str) -> None:
    """Validate a forecast-derived anomaly score aggregation method."""

    if method not in ANOMALY_SCORE_METHODS:
        raise ValueError(
            "anomaly_score_method must be one of: " + ", ".join(ANOMALY_SCORE_METHODS)
        )


def validate_joint_anomaly_score_method(method: str) -> None:
    """Validate anomaly score methods available to joint forecast/regime models."""

    if method not in JOINT_ANOMALY_SCORE_METHODS:
        raise ValueError(
            "anomaly_score_method must be one of: "
            + ", ".join(JOINT_ANOMALY_SCORE_METHODS)
        )


def forecast_anomaly_scores(
    observed: ArrayLike,
    forecast: MixtureForecast,
    *,
    method: str = "mean_nll",
) -> np.ndarray:
    """Aggregate forecast likelihoods into one anomaly score per example."""

    validate_anomaly_score_method(method)
    if method == "mean_nll":
        values = negative_log_likelihood_values(observed, forecast)
        return np.asarray(np.mean(values, axis=tuple(range(1, values.ndim))), dtype=np.float64)

    channel_values = channelwise_negative_log_likelihood_values(observed, forecast)
    if channel_values.ndim < 2:
        raise ValueError("channelwise anomaly scores require at least one channel dimension")
    flat_channels = channel_values.reshape(channel_values.shape[0], -1)

    if method == "channel_mean_nll":
        scores = np.mean(flat_channels, axis=-1)
    elif method == "channel_max_nll":
        scores = np.max(flat_channels, axis=-1)
    elif method == "channel_top3_mean_nll":
        k = min(3, flat_channels.shape[-1])
        scores = np.mean(np.sort(flat_channels, axis=-1)[:, -k:], axis=-1)
    else:  # pragma: no cover - kept for type checkers if the Literal expands.
        raise ValueError(f"unsupported anomaly score method: {method}")

    return np.asarray(scores, dtype=np.float64)


def joint_anomaly_scores(
    observed: ArrayLike,
    forecast: MixtureForecast,
    regime_probs: ArrayLike,
    *,
    method: str = "mean_nll",
) -> np.ndarray:
    """Aggregate joint forecast/regime outputs into one anomaly score per example."""

    validate_joint_anomaly_score_method(method)
    if method in ANOMALY_SCORE_METHODS:
        return forecast_anomaly_scores(observed, forecast, method=method)
    return regime_anomaly_scores(regime_probs, method=method)


def regime_anomaly_scores(regime_probs: ArrayLike, *, method: str) -> np.ndarray:
    """Return anomaly scores derived from regime posterior probabilities."""

    probs = _regime_probability_array(regime_probs)
    if method == "regime_entropy":
        scores = _normalized_entropy(probs)
    elif method == "regime_confidence_gap":
        scores = 1.0 - np.max(probs, axis=-1)
    elif method == "regime_switch":
        scores = _posterior_switch_scores(probs)
    else:
        raise ValueError(f"unsupported regime anomaly score method: {method}")
    return np.asarray(scores, dtype=np.float64)


def _regime_probability_array(regime_probs: ArrayLike) -> np.ndarray:
    probs = np.asarray(regime_probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("regime_probs must have shape (N, K)")
    if probs.shape[0] == 0 or probs.shape[1] == 0:
        raise ValueError("regime_probs must be non-empty")
    if np.any(~np.isfinite(probs)):
        raise ValueError("regime_probs must be finite")
    if np.any(probs < 0.0):
        raise ValueError("regime_probs must be non-negative")
    if not np.allclose(probs.sum(axis=-1), 1.0):
        raise ValueError("regime_probs must sum to 1 over regimes")
    return probs


def _normalized_entropy(probs: np.ndarray) -> np.ndarray:
    if probs.shape[-1] == 1:
        return np.zeros(probs.shape[0], dtype=np.float64)
    clipped = np.maximum(probs, 1e-12)
    entropy = -np.sum(probs * np.log(clipped), axis=-1)
    return np.asarray(entropy / np.log(probs.shape[-1]), dtype=np.float64)


def _posterior_switch_scores(probs: np.ndarray) -> np.ndarray:
    scores = np.zeros(probs.shape[0], dtype=np.float64)
    if probs.shape[0] < 2:
        return scores
    scores[1:] = 1.0 - np.sum(probs[1:] * probs[:-1], axis=-1)
    return scores
