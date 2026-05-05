"""End-to-end synthetic MDN-LSTM training pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from aaf.models.mdn_lstm import MDNLSTMConfig
from aaf.train.loop import TrainingConfig, predict_mdn_lstm, train_mdn_lstm


@dataclass(frozen=True)
class MDNSyntheticConfig:
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


def run_mdn_synthetic(
    output_dir: Path,
    config: MDNSyntheticConfig,
    *,
    overwrite: bool = False,
) -> EvaluationReport:
    """Train MDN-LSTM on synthetic data and write evaluation artifacts."""

    config.validate()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, validation_dataset, test_dataset, standardizer = _build_datasets(config)
    model_config = MDNLSTMConfig(
        input_size=config.n_channels,
        output_size=config.n_channels,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        horizon=config.horizon,
        n_components=config.n_components,
    )
    result = train_mdn_lstm(
        train_dataset,
        model_config,
        TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=config.seed,
        ),
        validation_dataset=validation_dataset,
    )

    validation_forecast = predict_mdn_lstm(result.model, validation_dataset)
    test_forecast = predict_mdn_lstm(result.model, test_dataset)
    _write_config(output_dir / "config.json", config)
    (output_dir / "training_history.json").write_text(
        json.dumps(asdict(result.history), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    torch.save(
        {
            "model_config": asdict(model_config),
            "state_dict": result.model.state_dict(),
        },
        output_dir / "model.pt",
    )
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
    parser = argparse.ArgumentParser(description="Train an MDN-LSTM on synthetic data.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--series-length", type=int, default=500)
    parser.add_argument("--lookback", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--n-train-configs", type=int, default=4)
    parser.add_argument("--n-validation-configs", type=int, default=1)
    parser.add_argument("--n-test-configs", type=int, default=1)
    parser.add_argument("--energy-samples", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_mdn_synthetic(
        args.output_dir,
        MDNSyntheticConfig(
            seed=args.seed,
            n_train_configs=args.n_train_configs,
            n_validation_configs=args.n_validation_configs,
            n_test_configs=args.n_test_configs,
            series_length=args.series_length,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=args.stride,
            hidden_size=args.hidden_size,
            n_components=args.n_components,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            energy_samples=args.energy_samples,
        ),
        overwrite=args.overwrite,
    )
    return 0


def _build_datasets(
    config: MDNSyntheticConfig,
) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset, Standardizer]:
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
    configs: Sequence[Any],
    *,
    config: MDNSyntheticConfig,
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
    return (
        FailureEvent(
            mode="spike_train",
            start=start,
            end=end,
            channels=(0,),
            magnitude=4.0,
            period=3,
        ),
    )


def _window_collection(
    series_collection: Sequence[SyntheticSeries],
    config: MDNSyntheticConfig,
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


def _concat_observations(series_collection: Sequence[SyntheticSeries]) -> FloatArray:
    return np.concatenate([series.observations for series in series_collection], axis=0)


def _write_forecast_artifact(path: Path, observed: FloatArray, forecast: MixtureForecast) -> None:
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


def _write_config(path: Path, config: MDNSyntheticConfig) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
