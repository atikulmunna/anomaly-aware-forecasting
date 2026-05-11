"""SMD seasonal-naive baseline pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aaf.baselines.forecasting import SeasonalNaiveForecaster
from aaf.data.preprocessing import Standardizer, WindowedDataset
from aaf.data.smd import prepare_smd_windowed_datasets
from aaf.data.synthetic import FloatArray
from aaf.eval.artifacts import write_mixture_diagnostics_json
from aaf.eval.forecasting import MixtureForecast, negative_log_likelihood_values
from aaf.eval.report import EvaluationReport, evaluate_run_directory


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


def run_smd_baseline(
    output_dir: Path,
    config: SMDBaselineConfig,
    *,
    overwrite: bool = False,
) -> EvaluationReport:
    """Run the SMD seasonal-naive baseline and write evaluation artifacts."""

    config.validate()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test, standardizers = build_smd_baseline_datasets(config)
    forecaster = fit_smd_baseline(train, config)
    validation_forecast = forecaster.predict(validation.windows, horizon=config.horizon)
    test_forecast = forecaster.predict(test.windows, horizon=config.horizon)

    _write_config(output_dir / "config.json", config)
    _write_standardizers(output_dir / "standardizers.npz", standardizers)
    write_smd_forecast_artifact(output_dir / "forecast.npz", test.targets, test_forecast)
    write_smd_anomaly_artifact(
        output_dir / "anomaly_validation.npz",
        validation,
        validation_forecast,
    )
    write_smd_anomaly_artifact(output_dir / "anomaly_test.npz", test, test_forecast)
    np.savez(
        output_dir / "regime.npz",
        true_labels=test.regime_labels,
        pred_labels=np.zeros_like(test.regime_labels),
    )
    write_mixture_diagnostics_json(
        output_dir / "mixture_diagnostics.json",
        validation=validation_forecast,
        test=test_forecast,
    )
    return evaluate_run_directory(
        output_dir,
        output_path=output_dir / "metrics.json",
        energy_samples=config.energy_samples,
        seed=config.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SMD seasonal-naive baseline.")
    parser.add_argument("root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--machine-id", action="append", dest="machine_ids")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--season-length", type=int, default=1)
    parser.add_argument("--energy-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_smd_baseline(
        args.output_dir,
        SMDBaselineConfig(
            root=args.root,
            machine_ids=None if args.machine_ids is None else tuple(args.machine_ids),
            validation_fraction=args.validation_fraction,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=args.stride,
            season_length=args.season_length,
            energy_samples=args.energy_samples,
            seed=args.seed,
        ),
        overwrite=args.overwrite,
    )
    return 0


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


def _write_standardizers(path: Path, standardizers: tuple[Standardizer, ...]) -> None:
    np.savez(
        path,
        mean=np.stack([standardizer.mean for standardizer in standardizers], axis=0),
        std=np.stack([standardizer.std for standardizer in standardizers], axis=0),
    )


def _write_config(path: Path, config: SMDBaselineConfig) -> None:
    path.write_text(
        json.dumps(_json_ready(asdict(config)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
