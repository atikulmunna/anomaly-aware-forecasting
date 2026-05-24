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


def validate_anomaly_score_method(method: str) -> None:
    """Validate a forecast-derived anomaly score aggregation method."""

    if method not in ANOMALY_SCORE_METHODS:
        raise ValueError(
            "anomaly_score_method must be one of: " + ", ".join(ANOMALY_SCORE_METHODS)
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
