"""SMD seasonal-naive baseline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aaf.baselines.forecasting import SeasonalNaiveForecaster
from aaf.data.preprocessing import Standardizer, WindowedDataset
from aaf.data.smd import prepare_smd_windowed_datasets
from aaf.data.synthetic import FloatArray
from aaf.eval.forecasting import MixtureForecast, negative_log_likelihood_values


@dataclass(frozen=True)
class SMDBaselineConfig:
    root: Path
    machine_ids: tuple[str, ...] | None = None
    validation_fraction: float = 0.2
    lookback: int = 100
    horizon: int = 1
    stride: int = 1
    season_length: int = 1
    energy_samples: int = 128
    seed: int = 0

    def validate(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.lookback < 1:
            raise ValueError("lookback must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.season_length < 1:
            raise ValueError("season_length must be positive")
        if self.energy_samples < 2:
            raise ValueError("energy_samples must be at least 2")


def build_smd_baseline_datasets(
    config: SMDBaselineConfig,
) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset, tuple[Standardizer, ...]]:
    """Build SMD windowed datasets for the baseline pipeline."""

    config.validate()
    return prepare_smd_windowed_datasets(
        config.root,
        machine_ids=config.machine_ids,
        validation_fraction=config.validation_fraction,
        lookback=config.lookback,
        horizon=config.horizon,
        stride=config.stride,
    )


def fit_smd_baseline(
    train_dataset: WindowedDataset,
    config: SMDBaselineConfig,
) -> SeasonalNaiveForecaster:
    """Fit a seasonal-naive baseline from SMD training windows."""

    return SeasonalNaiveForecaster.fit(
        _flatten_windows(train_dataset),
        season_length=config.season_length,
    )


def write_smd_forecast_artifact(
    path: Path,
    observed: FloatArray,
    forecast: MixtureForecast,
) -> None:
    """Write a forecast artifact compatible with aaf-evaluate."""

    np.savez(
        path,
        observed=observed,
        weights=forecast.weights,
        means=forecast.means,
        stds=forecast.stds,
    )


def write_smd_anomaly_artifact(
    path: Path,
    dataset: WindowedDataset,
    forecast: MixtureForecast,
) -> None:
    """Write SMD anomaly scores from forecast likelihoods."""

    scores = np.mean(negative_log_likelihood_values(dataset.targets, forecast), axis=-1)
    np.savez(path, scores=scores, labels=dataset.anomaly_labels)


def _flatten_windows(dataset: WindowedDataset) -> FloatArray:
    return np.asarray(dataset.windows.reshape(-1, dataset.windows.shape[-1]), dtype=np.float64)
