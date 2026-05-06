"""End-to-end synthetic joint MDN-LSTM training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aaf.data.failures import FailureEvent, apply_failure_events
from aaf.data.preprocessing import (
    Standardizer,
    WindowedDataset,
    make_windowed_dataset,
    standardize_series,
)
from aaf.data.synthetic import (
    FloatArray,
    GeneratorConfigSpace,
    SwitchingARConfig,
    SyntheticSeries,
    generate_switching_ar,
    sample_config_split,
)
from aaf.eval.forecasting import MixtureForecast, negative_log_likelihood_values


@dataclass(frozen=True)
class JointSyntheticConfig:
    seed: int = 0
    n_train_configs: int = 4
    n_validation_configs: int = 1
    n_test_configs: int = 1
    series_length: int = 500
    burn_in: int = 50
    lookback: int = 32
    horizon: int = 1
    stride: int = 1
    n_regimes: int = 3
    n_channels: int = 1
    ar_order: int = 2
    hidden_size: int = 32
    num_layers: int = 1
    n_components: int = 3
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    smoothness_weight: float = 0.1
    supervised_regime_weight: float = 0.0
    energy_samples: int = 64

    def validate(self) -> None:
        if self.series_length < self.lookback + self.horizon + 1:
            raise ValueError("series_length must exceed lookback + horizon")
        if self.n_train_configs < 1:
            raise ValueError("n_train_configs must be positive")
        if self.n_validation_configs < 1:
            raise ValueError("n_validation_configs must be positive")
        if self.n_test_configs < 1:
            raise ValueError("n_test_configs must be positive")
        if self.smoothness_weight < 0.0:
            raise ValueError("smoothness_weight must be non-negative")
        if self.supervised_regime_weight < 0.0:
            raise ValueError("supervised_regime_weight must be non-negative")


def build_joint_synthetic_datasets(
    config: JointSyntheticConfig,
) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset, Standardizer]:
    """Generate standardized train/validation/test datasets."""

    config.validate()
    split = sample_config_split(
        space=GeneratorConfigSpace(
            n_regimes=config.n_regimes,
            n_channels=config.n_channels,
            ar_order=config.ar_order,
        ),
        n_train=config.n_train_configs,
        n_validation=config.n_validation_configs,
        n_test=config.n_test_configs,
        seed=config.seed,
    )
    train_series = _generate_series(split.train, config=config, seed_offset=1_000, failures=False)
    validation_series = _generate_series(
        split.validation,
        config=config,
        seed_offset=2_000,
        failures=True,
    )
    test_series = _generate_series(split.test, config=config, seed_offset=3_000, failures=True)
    standardizer = Standardizer.fit(_concat_observations(train_series))
    return (
        _window_collection(
            tuple(standardize_series(series, standardizer) for series in train_series),
            config,
        ),
        _window_collection(
            tuple(standardize_series(series, standardizer) for series in validation_series),
            config,
        ),
        _window_collection(
            tuple(standardize_series(series, standardizer) for series in test_series),
            config,
        ),
        standardizer,
    )


def _generate_series(
    configs: tuple[SwitchingARConfig, ...],
    *,
    config: JointSyntheticConfig,
    seed_offset: int,
    failures: bool,
) -> tuple[SyntheticSeries, ...]:
    series_collection = []
    for idx, generator_config in enumerate(configs):
        series = generate_switching_ar(
            generator_config,
            length=config.series_length,
            seed=config.seed + seed_offset + idx,
            burn_in=config.burn_in,
        )
        if failures:
            series = apply_failure_events(
                series,
                _failure_events(config.series_length),
                seed=config.seed + seed_offset + 10_000 + idx,
            )
        series_collection.append(series)
    return tuple(series_collection)


def _failure_events(length: int) -> tuple[FailureEvent, ...]:
    start = max(1, int(length * 0.60))
    end = min(length, start + max(3, length // 20))
    return (FailureEvent(mode="spike_train", start=start, end=end, channels=(0,), magnitude=4.0),)


def _window_collection(
    series_collection: tuple[SyntheticSeries, ...],
    config: JointSyntheticConfig,
) -> WindowedDataset:
    datasets = [
        make_windowed_dataset(
            series,
            lookback=config.lookback,
            horizon=config.horizon,
            stride=config.stride,
        )
        for series in series_collection
    ]
    return WindowedDataset(
        windows=np.concatenate([dataset.windows for dataset in datasets], axis=0),
        targets=np.concatenate([dataset.targets for dataset in datasets], axis=0),
        regime_labels=np.concatenate([dataset.regime_labels for dataset in datasets], axis=0),
        anomaly_labels=np.concatenate([dataset.anomaly_labels for dataset in datasets], axis=0),
    )


def _concat_observations(series_collection: tuple[SyntheticSeries, ...]) -> FloatArray:
    return np.concatenate([series.observations for series in series_collection], axis=0)


def write_joint_forecast_artifact(
    path: Path,
    observed: FloatArray,
    forecast: MixtureForecast,
) -> None:
    """Write forecast artifact compatible with aaf-evaluate."""

    np.savez(
        path,
        observed=observed,
        weights=forecast.weights,
        means=forecast.means,
        stds=forecast.stds,
    )


def write_joint_anomaly_artifact(
    path: Path,
    dataset: WindowedDataset,
    forecast: MixtureForecast,
) -> None:
    """Write anomaly-score artifact from forecast likelihoods."""

    scores = np.mean(negative_log_likelihood_values(dataset.targets, forecast), axis=-1)
    np.savez(path, scores=scores, labels=dataset.anomaly_labels)
