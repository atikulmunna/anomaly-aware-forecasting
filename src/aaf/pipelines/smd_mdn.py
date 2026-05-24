"""SMD non-regime MDN-LSTM pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aaf.data.preprocessing import Standardizer, WindowedDataset
from aaf.data.smd import prepare_smd_windowed_datasets
from aaf.data.synthetic import FloatArray
from aaf.eval.anomaly import validate_threshold_strategy
from aaf.eval.anomaly_scores import forecast_anomaly_scores, validate_anomaly_score_method
from aaf.eval.artifacts import write_mixture_diagnostics_json
from aaf.eval.forecasting import MixtureForecast
from aaf.eval.report import EvaluationReport, evaluate_run_directory
from aaf.models.mdn_lstm import MDNLSTMConfig
from aaf.train.loop import TrainingConfig, TrainingResult, predict_mdn_lstm, train_mdn_lstm


@dataclass(frozen=True)
class SMDMDNConfig:
    root: Path
    machine_ids: tuple[str, ...] | None = None
    validation_fraction: float = 0.2
    lookback: int = 100
    horizon: int = 1
    stride: int = 1
    hidden_size: int = 32
    num_layers: int = 1
    n_components: int = 3
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    energy_samples: int = 128
    seed: int = 0
    device: str = "cpu"
    anomaly_score_method: str = "mean_nll"
    threshold_strategy: str = "max_validation_f1"

    def validate(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.lookback < 1:
            raise ValueError("lookback must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.energy_samples < 2:
            raise ValueError("energy_samples must be at least 2")
        if not self.device:
            raise ValueError("device must be non-empty")
        validate_anomaly_score_method(self.anomaly_score_method)
        validate_threshold_strategy(self.threshold_strategy)


def build_smd_mdn_datasets(
    config: SMDMDNConfig,
) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset, tuple[Standardizer, ...]]:
    """Build SMD windowed datasets for non-regime MDN-LSTM training."""

    config.validate()
    return prepare_smd_windowed_datasets(
        config.root,
        machine_ids=config.machine_ids,
        validation_fraction=config.validation_fraction,
        lookback=config.lookback,
        horizon=config.horizon,
        stride=config.stride,
    )


def run_smd_mdn(
    output_dir: Path,
    config: SMDMDNConfig,
    *,
    overwrite: bool = False,
) -> EvaluationReport:
    """Train the SMD non-regime MDN-LSTM and write evaluation artifacts."""

    config.validate()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train, validation, test, standardizers = build_smd_mdn_datasets(config)
    model_config = smd_mdn_model_config(train, config)
    result = train_smd_mdn_model(train, validation, config)
    validation_forecast, test_forecast = predict_smd_mdn_splits(
        result,
        validation,
        test,
        device=config.device,
    )

    _write_config(output_dir / "config.json", config)
    _write_training_artifacts(output_dir, result, model_config, standardizers)
    write_smd_mdn_forecast_artifact(output_dir / "forecast.npz", test.targets, test_forecast)
    write_smd_mdn_anomaly_artifact(
        output_dir / "anomaly_validation.npz",
        validation,
        validation_forecast,
        method=config.anomaly_score_method,
    )
    write_smd_mdn_anomaly_artifact(
        output_dir / "anomaly_test.npz",
        test,
        test_forecast,
        method=config.anomaly_score_method,
    )
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
        threshold_strategy=config.threshold_strategy,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a non-regime MDN-LSTM on SMD data.")
    parser.add_argument("root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--machine-id", action="append", dest="machine_ids")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--energy-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--anomaly-score-method", default="mean_nll")
    parser.add_argument("--threshold-strategy", default="max_validation_f1")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_smd_mdn(
        args.output_dir,
        SMDMDNConfig(
            root=args.root,
            machine_ids=None if args.machine_ids is None else tuple(args.machine_ids),
            validation_fraction=args.validation_fraction,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=args.stride,
            hidden_size=args.hidden_size,
            n_components=args.n_components,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            energy_samples=args.energy_samples,
            seed=args.seed,
            device=args.device,
            anomaly_score_method=args.anomaly_score_method,
            threshold_strategy=args.threshold_strategy,
        ),
        overwrite=args.overwrite,
    )
    return 0


def smd_mdn_model_config(dataset: WindowedDataset, config: SMDMDNConfig) -> MDNLSTMConfig:
    """Create an MDN-LSTM model config from SMD dataset dimensions."""

    return MDNLSTMConfig(
        input_size=dataset.windows.shape[-1],
        output_size=dataset.targets.shape[-1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        horizon=config.horizon,
        n_components=config.n_components,
    )


def smd_mdn_training_config(config: SMDMDNConfig) -> TrainingConfig:
    """Create training loop parameters for an SMD MDN-LSTM run."""

    return TrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
        device=config.device,
    )


def train_smd_mdn_model(
    train_dataset: WindowedDataset,
    validation_dataset: WindowedDataset,
    config: SMDMDNConfig,
) -> TrainingResult:
    """Train the non-regime MDN-LSTM on SMD windows."""

    return train_mdn_lstm(
        train_dataset,
        smd_mdn_model_config(train_dataset, config),
        smd_mdn_training_config(config),
        validation_dataset=validation_dataset,
    )


def predict_smd_mdn_splits(
    result: TrainingResult,
    validation_dataset: WindowedDataset,
    test_dataset: WindowedDataset,
    *,
    device: str = "cpu",
) -> tuple[MixtureForecast, MixtureForecast]:
    """Predict validation and test splits for a trained non-regime MDN-LSTM."""

    return (
        predict_mdn_lstm(result.model, validation_dataset, device=device),
        predict_mdn_lstm(result.model, test_dataset, device=device),
    )


def write_smd_mdn_forecast_artifact(
    path: Path,
    observed: FloatArray,
    forecast: MixtureForecast,
) -> None:
    """Write an SMD MDN-LSTM forecast artifact compatible with aaf-evaluate."""

    np.savez(
        path,
        observed=observed,
        weights=forecast.weights,
        means=forecast.means,
        stds=forecast.stds,
    )


def write_smd_mdn_anomaly_artifact(
    path: Path,
    dataset: WindowedDataset,
    forecast: MixtureForecast,
    *,
    method: str = "mean_nll",
) -> None:
    """Write SMD MDN-LSTM anomaly scores from forecast likelihoods."""

    scores = forecast_anomaly_scores(dataset.targets, forecast, method=method)
    np.savez(path, scores=scores, labels=dataset.anomaly_labels)


def _write_training_artifacts(
    output_dir: Path,
    result: TrainingResult,
    model_config: MDNLSTMConfig,
    standardizers: tuple[Standardizer, ...],
) -> None:
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
    np.savez(
        output_dir / "standardizers.npz",
        mean=np.stack([standardizer.mean for standardizer in standardizers], axis=0),
        std=np.stack([standardizer.std for standardizer in standardizers], axis=0),
    )


def _write_config(path: Path, config: SMDMDNConfig) -> None:
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
