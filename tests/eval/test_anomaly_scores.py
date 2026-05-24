import numpy as np
import pytest

from aaf.eval.anomaly_scores import (
    forecast_anomaly_scores,
    validate_anomaly_score_method,
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
