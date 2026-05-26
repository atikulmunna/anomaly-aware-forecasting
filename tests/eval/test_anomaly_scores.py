import numpy as np
import pytest

from aaf.eval.anomaly_scores import (
    forecast_anomaly_scores,
    joint_anomaly_scores,
    regime_anomaly_scores,
    validate_anomaly_score_method,
    validate_joint_anomaly_score_method,
)
from aaf.eval.forecasting import MixtureForecast, channelwise_negative_log_likelihood_values


def make_forecast() -> MixtureForecast:
    return MixtureForecast.from_arrays(
        weights=np.ones((2, 1)),
        means=np.zeros((2, 1, 4)),
        stds=np.ones((2, 1, 4)),
    )


def test_forecast_anomaly_scores_mean_nll_returns_one_score_per_example() -> None:
    forecast = make_forecast()
    observed = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]])

    scores = forecast_anomaly_scores(observed, forecast, method="mean_nll")

    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_forecast_anomaly_scores_channel_aggregations() -> None:
    forecast = make_forecast()
    observed = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]])
    channel_values = channelwise_negative_log_likelihood_values(observed, forecast)

    assert forecast_anomaly_scores(observed, forecast, method="channel_mean_nll") == pytest.approx(
        channel_values.mean(axis=-1)
    )
    assert forecast_anomaly_scores(observed, forecast, method="channel_max_nll") == pytest.approx(
        channel_values.max(axis=-1)
    )
    assert forecast_anomaly_scores(
        observed,
        forecast,
        method="channel_top3_mean_nll",
    ) == pytest.approx(np.sort(channel_values, axis=-1)[:, -3:].mean(axis=-1))


def test_validate_anomaly_score_method_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="anomaly_score_method"):
        validate_anomaly_score_method("unknown")


def test_validate_joint_anomaly_score_method_accepts_regime_methods() -> None:
    validate_joint_anomaly_score_method("regime_entropy")
    validate_joint_anomaly_score_method("regime_confidence_gap")
    validate_joint_anomaly_score_method("regime_switch")


def test_forecast_anomaly_score_methods_reject_joint_only_methods() -> None:
    with pytest.raises(ValueError, match="anomaly_score_method"):
        validate_anomaly_score_method("regime_entropy")


def test_regime_anomaly_scores_entropy_and_confidence_gap() -> None:
    probs = np.array([[1.0, 0.0], [0.5, 0.5], [0.8, 0.2]])

    entropy = regime_anomaly_scores(probs, method="regime_entropy")
    confidence_gap = regime_anomaly_scores(probs, method="regime_confidence_gap")

    assert entropy[0] == pytest.approx(0.0)
    assert entropy[1] == pytest.approx(1.0)
    assert confidence_gap == pytest.approx([0.0, 0.5, 0.2])


def test_regime_anomaly_scores_switch_uses_posterior_distance() -> None:
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.25, 0.75]])

    scores = regime_anomaly_scores(probs, method="regime_switch")

    assert scores == pytest.approx([0.0, 0.0, 1.0, 0.25])


def test_joint_anomaly_scores_dispatches_to_forecast_or_regime_methods() -> None:
    forecast = make_forecast()
    observed = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]])
    probs = np.array([[1.0, 0.0], [0.5, 0.5]])

    forecast_scores = joint_anomaly_scores(
        observed,
        forecast,
        probs,
        method="channel_max_nll",
    )
    regime_scores = joint_anomaly_scores(
        observed,
        forecast,
        probs,
        method="regime_entropy",
    )

    assert forecast_scores == pytest.approx(
        forecast_anomaly_scores(observed, forecast, method="channel_max_nll")
    )
    assert regime_scores == pytest.approx([0.0, 1.0])
