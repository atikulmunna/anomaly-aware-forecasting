"""End-to-end synthetic seasonal-naive baseline pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aaf.baselines.forecasting import SeasonalNaiveForecaster
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
    SyntheticSeries,
    generate_switching_ar,
    sample_config_split,
)
from aaf.eval.artifacts import write_mixture_diagnostics_json
from aaf.eval.forecasting import MixtureForecast, negative_log_likelihood_values
from aaf.eval.report import EvaluationReport, evaluate_run_directory


@dataclass(frozen=True)
class SyntheticBaselineConfig:
    seed: int = 0
    n_train_configs: int = 20
    n_validation_configs: int = 5
    n_test_configs: int = 10
    series_length: int = 3_000
    burn_in: int = 50
    lookback: int = 100
    horizon: int = 1
    stride: int = 1
    season_length: int = 1
    n_regimes: int = 3
    n_channels: int = 1
    ar_order: int = 2
    energy_samples: int = 128

    def validate(self) -> None:
        if self.n_train_configs < 1:
            raise ValueError("n_train_configs must be positive")
        if self.n_validation_configs < 1:
            raise ValueError("n_validation_configs must be positive")
        if self.n_test_configs < 1:
            raise ValueError("n_test_configs must be positive")
        if self.series_length < self.lookback + self.horizon + 1:
            raise ValueError("series_length must exceed lookback + horizon")
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


def run_synthetic_baseline(
    output_dir: Path,
    config: SyntheticBaselineConfig,
    *,
    overwrite: bool = False,
) -> EvaluationReport:
    """Generate synthetic data, run a seasonal-naive baseline, and write run artifacts."""

    config.validate()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    train_series = _generate_series_collection(
        split.train,
        length=config.series_length,
        burn_in=config.burn_in,
        seed=config.seed + 1_000,
        inject_failures=False,
    )
    validation_series = _generate_series_collection(
        split.validation,
        length=config.series_length,
        burn_in=config.burn_in,
        seed=config.seed + 2_000,
        inject_failures=True,
    )
    test_series = _generate_series_collection(
        split.test,
        length=config.series_length,
        burn_in=config.burn_in,
        seed=config.seed + 3_000,
        inject_failures=True,
    )

    standardizer = Standardizer.fit(_concat_observations(train_series))
    train_standardized = tuple(standardize_series(series, standardizer) for series in train_series)
    validation_standardized = tuple(
        standardize_series(series, standardizer) for series in validation_series
    )
    test_standardized = tuple(standardize_series(series, standardizer) for series in test_series)

    train_observations = _concat_observations(train_standardized)
    forecaster = SeasonalNaiveForecaster.fit(
        train_observations,
        season_length=config.season_length,
    )

    validation_dataset = _concat_datasets(
        tuple(
            make_windowed_dataset(
                series,
                lookback=config.lookback,
                horizon=config.horizon,
                stride=config.stride,
            )
            for series in validation_standardized
        )
    )
    test_dataset = _concat_datasets(
        tuple(
            make_windowed_dataset(
                series,
                lookback=config.lookback,
                horizon=config.horizon,
                stride=config.stride,
            )
            for series in test_standardized
        )
    )

    validation_forecast = forecaster.predict(validation_dataset.windows, horizon=config.horizon)
    test_forecast = forecaster.predict(test_dataset.windows, horizon=config.horizon)

    _write_config(output_dir / "config.json", config)
    np.savez(output_dir / "standardizer.npz", mean=standardizer.mean, std=standardizer.std)
    _write_forecast_artifact(output_dir / "forecast.npz", test_dataset.targets, test_forecast)
    write_mixture_diagnostics_json(
        output_dir / "mixture_diagnostics.json",
        validation=validation_forecast,
        test=test_forecast,
    )
    _write_anomaly_artifact(
        output_dir / "anomaly_validation.npz",
        validation_dataset,
        validation_forecast,
    )
    _write_anomaly_artifact(output_dir / "anomaly_test.npz", test_dataset, test_forecast)
    np.savez(
        output_dir / "regime.npz",
        true_labels=test_dataset.regime_labels,
        pred_labels=np.zeros_like(test_dataset.regime_labels),
    )

    return evaluate_run_directory(
        output_dir,
        output_path=output_dir / "metrics.json",
        energy_samples=config.energy_samples,
        seed=config.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the synthetic seasonal-naive baseline.")
    parser.add_argument("output_dir", type=Path, help="Directory where run artifacts are written.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--series-length", type=int, default=3_000)
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--season-length", type=int, default=1)
    parser.add_argument("--n-train-configs", type=int, default=20)
    parser.add_argument("--n-validation-configs", type=int, default=5)
    parser.add_argument("--n-test-configs", type=int, default=10)
    parser.add_argument("--n-regimes", type=int, default=3)
    parser.add_argument("--n-channels", type=int, default=1)
    parser.add_argument("--ar-order", type=int, default=2)
    parser.add_argument("--energy-samples", type=int, default=128)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_synthetic_baseline(
        args.output_dir,
        SyntheticBaselineConfig(
            seed=args.seed,
            n_train_configs=args.n_train_configs,
            n_validation_configs=args.n_validation_configs,
            n_test_configs=args.n_test_configs,
            series_length=args.series_length,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=args.stride,
            season_length=args.season_length,
            n_regimes=args.n_regimes,
            n_channels=args.n_channels,
            ar_order=args.ar_order,
            energy_samples=args.energy_samples,
        ),
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _generate_series_collection(
    configs: Sequence[Any],
    *,
    length: int,
    burn_in: int,
    seed: int,
    inject_failures: bool,
) -> tuple[SyntheticSeries, ...]:
    series = []
    for idx, generator_config in enumerate(configs):
        generated = generate_switching_ar(
            generator_config,
            length=length,
            seed=seed + idx,
            burn_in=burn_in,
        )
        if inject_failures:
            generated = apply_failure_events(
                generated,
                _default_failure_events(length, generator_config.n_channels),
                seed=seed + 10_000 + idx,
            )
        series.append(generated)
    return tuple(series)


def _default_failure_events(length: int, n_channels: int) -> tuple[FailureEvent, ...]:
    first_start = max(1, int(length * 0.35))
    first_end = min(length, first_start + max(3, length // 20))
    second_start = max(first_end + 1, int(length * 0.70))
    second_end = min(length, second_start + max(3, length // 25))
    channel = (0,) if n_channels > 0 else None
    return (
        FailureEvent(
            mode="additive_drift",
            start=first_start,
            end=first_end,
            channels=channel,
            magnitude=3.0,
        ),
        FailureEvent(
            mode="spike_train",
            start=second_start,
            end=second_end,
            channels=channel,
            magnitude=5.0,
        ),
    )


def _concat_observations(series_collection: Sequence[SyntheticSeries]) -> FloatArray:
    return np.concatenate([series.observations for series in series_collection], axis=0)


def _concat_datasets(datasets: Sequence[WindowedDataset]) -> WindowedDataset:
    return WindowedDataset(
        windows=np.concatenate([dataset.windows for dataset in datasets], axis=0),
        targets=np.concatenate([dataset.targets for dataset in datasets], axis=0),
        regime_labels=np.concatenate([dataset.regime_labels for dataset in datasets], axis=0),
        anomaly_labels=np.concatenate([dataset.anomaly_labels for dataset in datasets], axis=0),
    )


def _write_forecast_artifact(
    path: Path,
    observed: FloatArray,
    forecast: MixtureForecast,
) -> None:
    np.savez(
        path,
        observed=observed,
        weights=forecast.weights,
        means=forecast.means,
        stds=forecast.stds,
    )


def _write_anomaly_artifact(
    path: Path,
    dataset: WindowedDataset,
    forecast: MixtureForecast,
) -> None:
    scores = np.mean(negative_log_likelihood_values(dataset.targets, forecast), axis=-1)
    np.savez(path, scores=scores, labels=dataset.anomaly_labels)


def _write_config(path: Path, config: SyntheticBaselineConfig) -> None:
    path.write_text(
        json.dumps(_json_ready(asdict(config)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value
