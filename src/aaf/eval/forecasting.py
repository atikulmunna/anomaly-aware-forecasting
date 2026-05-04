"""Forecasting metrics for diagonal-covariance Gaussian mixture predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp as scipy_logsumexp
from scipy.special import ndtr as scipy_ndtr

FloatArray = NDArray[np.float64]

_LOG_2PI = float(np.log(2.0 * np.pi))
_MIN_STD = 1e-12


@dataclass(frozen=True)
class MixtureForecast:
    """Diagonal-covariance Gaussian mixture forecast.

    Shapes follow the project SRS:
    - weights: (..., M)
    - means: (..., M, D)
    - stds: (..., M, D)
    """

    weights: FloatArray
    means: FloatArray
    stds: FloatArray

    @classmethod
    def from_arrays(
        cls,
        weights: ArrayLike,
        means: ArrayLike,
        stds: ArrayLike,
    ) -> MixtureForecast:
        forecast = cls(
            weights=np.asarray(weights, dtype=np.float64),
            means=np.asarray(means, dtype=np.float64),
            stds=np.asarray(stds, dtype=np.float64),
        )
        forecast.validate()
        return forecast

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[-1])

    @property
    def n_channels(self) -> int:
        return int(self.means.shape[-1])

    def validate(self) -> None:
        if self.weights.ndim < 1:
            raise ValueError("weights must have at least one dimension")
        if self.means.ndim < 2 or self.stds.ndim < 2:
            raise ValueError("means and stds must have at least two dimensions")
        if self.means.shape != self.stds.shape:
            raise ValueError("means and stds must have identical shapes")
        if self.weights.shape != self.means.shape[:-1]:
            raise ValueError(
                "weights shape must match means shape without the output-channel dimension"
            )
        if np.any(~np.isfinite(self.weights)):
            raise ValueError("weights must be finite")
        if np.any(~np.isfinite(self.means)):
            raise ValueError("means must be finite")
        if np.any(~np.isfinite(self.stds)):
            raise ValueError("stds must be finite")
        if np.any(self.weights < 0.0):
            raise ValueError("weights must be non-negative")
        if np.any(self.stds <= 0.0):
            raise ValueError("stds must be strictly positive")
        sums = self.weights.sum(axis=-1)
        if np.any(sums <= 0.0):
            raise ValueError("at least one mixture weight must be positive per forecast")

    def normalized_weights(self) -> FloatArray:
        return np.asarray(self.weights / self.weights.sum(axis=-1, keepdims=True), dtype=np.float64)


def negative_log_likelihood(observed: ArrayLike, forecast: MixtureForecast) -> float:
    """Return mean NLL under a diagonal multivariate Gaussian mixture."""

    return float(np.mean(negative_log_likelihood_values(observed, forecast)))


def negative_log_likelihood_values(observed: ArrayLike, forecast: MixtureForecast) -> FloatArray:
    """Return per-forecast NLL values with shape matching the forecast batch dimensions."""

    y = _observed_array(observed, forecast)
    weights = forecast.normalized_weights()
    diff = y[..., np.newaxis, :] - forecast.means
    stds = np.maximum(forecast.stds, _MIN_STD)
    log_component = -0.5 * (((diff / stds) ** 2) + (2.0 * np.log(stds)) + _LOG_2PI)
    log_component = log_component.sum(axis=-1)
    log_mix = _logsumexp(np.log(weights) + log_component, axis=-1)
    return np.asarray(-log_mix, dtype=np.float64)


def predictive_mean(forecast: MixtureForecast) -> FloatArray:
    """Return the mixture predictive mean with shape (..., D)."""

    weights = forecast.normalized_weights()
    return np.asarray(np.sum(weights[..., np.newaxis] * forecast.means, axis=-2), dtype=np.float64)


def mean_absolute_error(observed: ArrayLike, forecast: MixtureForecast) -> float:
    """Return MAE of the mixture predictive mean."""

    y = _observed_array(observed, forecast)
    return float(np.mean(np.abs(y - predictive_mean(forecast))))


def root_mean_squared_error(observed: ArrayLike, forecast: MixtureForecast) -> float:
    """Return RMSE of the mixture predictive mean."""

    y = _observed_array(observed, forecast)
    return float(np.sqrt(np.mean((y - predictive_mean(forecast)) ** 2)))


def channelwise_crps(observed: ArrayLike, forecast: MixtureForecast) -> float:
    """Return CRPS per output channel, averaged over observations and channels."""

    return float(np.mean(channelwise_crps_values(observed, forecast)))


def channelwise_crps_values(observed: ArrayLike, forecast: MixtureForecast) -> FloatArray:
    """Return per-observation, per-channel CRPS values with shape (..., D)."""

    y = _observed_array(observed, forecast)
    weights = forecast.normalized_weights()
    first = np.asarray(
        np.sum(
            weights[..., np.newaxis]
            * _normal_abs_expectation(y[..., np.newaxis, :] - forecast.means, forecast.stds),
            axis=-2,
        ),
        dtype=np.float64,
    )

    mean_diff = forecast.means[..., :, np.newaxis, :] - forecast.means[..., np.newaxis, :, :]
    std_pair = np.sqrt(
        forecast.stds[..., :, np.newaxis, :] ** 2 + forecast.stds[..., np.newaxis, :, :] ** 2
    )
    weight_pair = weights[..., :, np.newaxis] * weights[..., np.newaxis, :]
    second = np.asarray(
        0.5
        * np.sum(
            weight_pair[..., np.newaxis] * _normal_abs_expectation(mean_diff, std_pair),
            axis=(-3, -2),
        ),
        dtype=np.float64,
    )
    return np.asarray(first - second, dtype=np.float64)


def pit_values(observed: ArrayLike, forecast: MixtureForecast) -> FloatArray:
    """Return probability integral transform values per channel."""

    y = _observed_array(observed, forecast)
    weights = forecast.normalized_weights()
    z = (y[..., np.newaxis, :] - forecast.means) / np.maximum(forecast.stds, _MIN_STD)
    return np.asarray(
        np.sum(weights[..., np.newaxis] * _normal_cdf(z), axis=-2),
        dtype=np.float64,
    )


def central_interval_coverage(
    observed: ArrayLike,
    forecast: MixtureForecast,
    *,
    level: float = 0.9,
    n_grid: int = 2048,
) -> float:
    """Estimate central predictive interval coverage from mixture CDF grids.

    This is intended for diagnostics, not as a training objective. The grid approach keeps the
    implementation dependency-light while supporting non-Gaussian mixture shapes.
    """

    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    if n_grid < 128:
        raise ValueError("n_grid must be at least 128")

    y = _observed_array(observed, forecast)
    lower_q = (1.0 - level) / 2.0
    upper_q = 1.0 - lower_q
    lower, upper = _mixture_quantiles(forecast, (lower_q, upper_q), n_grid=n_grid)
    return float(np.mean((y >= lower) & (y <= upper)))


def energy_score(
    observed: ArrayLike,
    forecast: MixtureForecast,
    *,
    n_samples: int = 256,
    seed: int = 0,
) -> float:
    """Estimate the multivariate Energy Score via Monte Carlo samples."""

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    y = _observed_array(observed, forecast)
    samples = sample_mixture(forecast, n_samples=n_samples, seed=seed)
    diff_obs = np.linalg.norm(samples - y[..., np.newaxis, :], axis=-1).mean(axis=-1)
    sample_diff = np.linalg.norm(
        samples[..., :, np.newaxis, :] - samples[..., np.newaxis, :, :],
        axis=-1,
    ).mean(axis=(-2, -1))
    return float(np.mean(diff_obs - 0.5 * sample_diff))


def sample_mixture(
    forecast: MixtureForecast,
    *,
    n_samples: int,
    seed: int | None = None,
) -> FloatArray:
    """Draw samples with shape (..., n_samples, D) from the forecast mixture."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    weights = forecast.normalized_weights()
    flat_weights = weights.reshape(-1, forecast.n_components)
    flat_means = forecast.means.reshape(-1, forecast.n_components, forecast.n_channels)
    flat_stds = forecast.stds.reshape(-1, forecast.n_components, forecast.n_channels)

    flat_samples = np.empty((flat_weights.shape[0], n_samples, forecast.n_channels), dtype=np.float64)
    for idx, probs in enumerate(flat_weights):
        components = rng.choice(forecast.n_components, size=n_samples, p=probs)
        loc = flat_means[idx, components]
        scale = flat_stds[idx, components]
        flat_samples[idx] = rng.normal(loc=loc, scale=scale)

    return flat_samples.reshape(*weights.shape[:-1], n_samples, forecast.n_channels)


def _observed_array(observed: ArrayLike, forecast: MixtureForecast) -> FloatArray:
    y = np.asarray(observed, dtype=np.float64)
    if y.shape != forecast.means.shape[:-2] + (forecast.n_channels,):
        raise ValueError(
            "observed shape must match forecast batch dimensions plus output channels"
        )
    if np.any(~np.isfinite(y)):
        raise ValueError("observed values must be finite")
    return y


def _normal_abs_expectation(mean: FloatArray, std: FloatArray) -> FloatArray:
    std = np.maximum(std, _MIN_STD)
    z = mean / std
    return np.asarray(
        std * (2.0 * _standard_normal_pdf(z) + z * (2.0 * _normal_cdf(z) - 1.0)),
        dtype=np.float64,
    )


def _standard_normal_pdf(x: FloatArray) -> FloatArray:
    return np.asarray(np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi), dtype=np.float64)


def _normal_cdf(x: FloatArray) -> FloatArray:
    return np.asarray(scipy_ndtr(x), dtype=np.float64)


def _logsumexp(x: FloatArray, *, axis: int) -> FloatArray:
    return np.asarray(scipy_logsumexp(x, axis=axis), dtype=np.float64)


def _mixture_quantiles(
    forecast: MixtureForecast,
    quantiles: tuple[float, float],
    *,
    n_grid: int,
) -> tuple[FloatArray, FloatArray]:
    weights = forecast.normalized_weights()
    low = np.min(forecast.means - 8.0 * forecast.stds, axis=-2)
    high = np.max(forecast.means + 8.0 * forecast.stds, axis=-2)
    grid_unit = np.linspace(0.0, 1.0, n_grid, dtype=np.float64)

    flat_weights = weights.reshape(-1, forecast.n_components)
    flat_means = forecast.means.reshape(-1, forecast.n_components, forecast.n_channels)
    flat_stds = forecast.stds.reshape(-1, forecast.n_components, forecast.n_channels)
    flat_low = low.reshape(-1, forecast.n_channels)
    flat_high = high.reshape(-1, forecast.n_channels)

    out = np.empty((2, flat_weights.shape[0], forecast.n_channels), dtype=np.float64)
    for row in range(flat_weights.shape[0]):
        for channel in range(forecast.n_channels):
            grid = flat_low[row, channel] + grid_unit * (
                flat_high[row, channel] - flat_low[row, channel]
            )
            z = (grid[:, np.newaxis] - flat_means[row, :, channel]) / flat_stds[row, :, channel]
            cdf = _normal_cdf(z) @ flat_weights[row]
            out[:, row, channel] = np.interp(quantiles, cdf, grid)

    shape = forecast.means.shape[:-2] + (forecast.n_channels,)
    return out[0].reshape(shape), out[1].reshape(shape)
