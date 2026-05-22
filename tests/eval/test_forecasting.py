import math

import numpy as np
import pytest

from aaf.eval.forecasting import (
    MixtureForecast,
    central_interval_coverage,
    channelwise_crps,
    channelwise_crps_values,
    energy_score,
    mean_absolute_error,
    negative_log_likelihood,
    negative_log_likelihood_values,
    pit_values,
    predictive_mean,
    root_mean_squared_error,
    sample_mixture,
)


def test_standard_normal_nll_known_value() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.array([[1.0]]),
        means=np.array([[[0.0]]]),
        stds=np.array([[[1.0]]]),
    )

    assert negative_log_likelihood(np.array([[0.0]]), forecast) == pytest.approx(
        0.5 * math.log(2.0 * math.pi)
    )


def test_nll_values_preserve_batch_dimensions() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.ones((2, 3, 1)),
        means=np.zeros((2, 3, 1, 1)),
        stds=np.ones((2, 3, 1, 1)),
    )

    values = negative_log_likelihood_values(np.zeros((2, 3, 1)), forecast)

    assert values.shape == (2, 3)
    assert negative_log_likelihood(np.zeros((2, 3, 1)), forecast) == pytest.approx(
        float(values.mean())
    )


def test_standard_normal_crps_known_value() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.array([[1.0]]),
        means=np.array([[[0.0]]]),
        stds=np.array([[[1.0]]]),
    )

    expected = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)

    assert channelwise_crps(np.array([[0.0]]), forecast) == pytest.approx(expected)


def test_channelwise_crps_averages_over_channels() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.array([[1.0]]),
        means=np.array([[[0.0, 2.0]]]),
        stds=np.array([[[1.0, 1.0]]]),
    )
    observed = np.array([[0.0, 2.0]])

    values = channelwise_crps_values(observed, forecast)

    assert values.shape == (1, 2)
    assert channelwise_crps(observed, forecast) == pytest.approx(float(values.mean()))


def test_mixture_weights_are_normalized_in_metrics() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.array([[2.0, 2.0]]),
        means=np.array([[[0.0], [2.0]]]),
        stds=np.array([[[1.0], [1.0]]]),
    )

    assert predictive_mean(forecast) == pytest.approx(np.array([[1.0]]))


def test_point_metrics_use_predictive_mean() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.array([[0.25, 0.75]]),
        means=np.array([[[0.0], [4.0]]]),
        stds=np.array([[[1.0], [1.0]]]),
    )
    observed = np.array([[1.0]])

    assert predictive_mean(forecast) == pytest.approx(np.array([[3.0]]))
    assert mean_absolute_error(observed, forecast) == pytest.approx(2.0)
    assert root_mean_squared_error(observed, forecast) == pytest.approx(2.0)


def test_pit_values_match_standard_normal_cdf_at_mean() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.array([[1.0]]),
        means=np.array([[[0.0]]]),
        stds=np.array([[[1.0]]]),
    )

    assert pit_values(np.array([[0.0]]), forecast) == pytest.approx(np.array([[0.5]]))


def test_central_interval_coverage_for_obvious_inside_and_outside_points() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.ones((2, 1)),
        means=np.zeros((2, 1, 1)),
        stds=np.ones((2, 1, 1)),
    )
    observed = np.array([[0.0], [10.0]])

    assert central_interval_coverage(observed, forecast, level=0.9) == pytest.approx(0.5)


def test_energy_score_is_deterministic_for_seed() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.array([[0.4, 0.6]]),
        means=np.array([[[0.0, 0.0], [2.0, 2.0]]]),
        stds=np.array([[[1.0, 1.0], [1.0, 1.0]]]),
    )
    observed = np.array([[1.0, 1.0]])

    first = energy_score(observed, forecast, n_samples=64, seed=123)
    second = energy_score(observed, forecast, n_samples=64, seed=123)

    assert first == pytest.approx(second)
    assert first >= 0.0


def test_energy_score_matches_direct_pairwise_estimate() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.ones((2, 1)),
        means=np.array([[[0.0, 0.0]], [[2.0, 2.0]]]),
        stds=np.ones((2, 1, 2)),
    )
    observed = np.array([[0.5, 0.5], [1.5, 1.5]])
    samples = sample_mixture(forecast, n_samples=8, seed=7)
    diff_obs = np.linalg.norm(samples - observed[:, np.newaxis, :], axis=-1).mean(axis=-1)
    sample_diff = np.linalg.norm(
        samples[..., :, np.newaxis, :] - samples[..., np.newaxis, :, :],
        axis=-1,
    ).mean(axis=(-2, -1))
    expected = float(np.mean(diff_obs - 0.5 * sample_diff))

    assert energy_score(observed, forecast, n_samples=8, seed=7) == pytest.approx(expected)


def test_sample_mixture_shape_matches_forecast_batch_dimensions() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.ones((2, 3, 1)),
        means=np.zeros((2, 3, 1, 4)),
        stds=np.ones((2, 3, 1, 4)),
    )

    samples = sample_mixture(forecast, n_samples=5, seed=0)

    assert samples.shape == (2, 3, 5, 4)


def test_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="weights shape"):
        MixtureForecast.from_arrays(
            weights=np.ones((2, 1)),
            means=np.zeros((2, 3, 1, 1)),
            stds=np.ones((2, 3, 1, 1)),
        )


def test_rejects_non_positive_stds() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        MixtureForecast.from_arrays(
            weights=np.array([[1.0]]),
            means=np.array([[[0.0]]]),
            stds=np.array([[[0.0]]]),
        )
