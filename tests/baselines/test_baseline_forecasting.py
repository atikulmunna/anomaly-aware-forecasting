import numpy as np
import pytest

from aaf.baselines.forecasting import SeasonalNaiveForecaster, persistence_forecast
from aaf.eval.forecasting import predictive_mean


def test_seasonal_naive_fit_estimates_per_channel_residual_std() -> None:
    series = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    forecaster = SeasonalNaiveForecaster.fit(series, season_length=2)

    assert forecaster.residual_std.shape == (2,)
    assert forecaster.residual_std[0] >= forecaster.std_floor
    assert forecaster.residual_std[1] >= forecaster.std_floor


def test_seasonal_naive_predict_repeats_last_season_across_horizon() -> None:
    forecaster = SeasonalNaiveForecaster.fit(np.arange(10.0), season_length=3)
    history = np.array([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]])

    forecast = forecaster.predict(history, horizon=5)

    assert forecast.weights.shape == (1, 5, 1)
    assert forecast.means.shape == (1, 5, 1, 1)
    assert predictive_mean(forecast).reshape(-1).tolist() == [4.0, 5.0, 6.0, 4.0, 5.0]


def test_persistence_forecast_repeats_last_value() -> None:
    history = np.array([[[1.0, 2.0], [3.0, 4.0]]])

    forecast = persistence_forecast(history, horizon=3, std=0.5)

    assert predictive_mean(forecast).tolist() == [[[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]]
    assert np.all(forecast.stds == pytest.approx(0.5))


def test_rejects_history_shorter_than_season() -> None:
    forecaster = SeasonalNaiveForecaster.fit(np.arange(10.0), season_length=4)

    with pytest.raises(ValueError, match="at least season_length"):
        forecaster.predict(np.ones((1, 3, 1)), horizon=1)


def test_rejects_invalid_season_length() -> None:
    with pytest.raises(ValueError, match="positive"):
        SeasonalNaiveForecaster.fit(np.arange(10.0), season_length=0)
