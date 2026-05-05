import json

import numpy as np

from aaf.pipelines.mdn_synthetic import MDNSyntheticConfig, main, run_mdn_synthetic


def tiny_config(seed: int = 0) -> MDNSyntheticConfig:
    return MDNSyntheticConfig(
        seed=seed,
        n_train_configs=1,
        n_validation_configs=1,
        n_test_configs=1,
        series_length=80,
        burn_in=10,
        lookback=8,
        horizon=1,
        stride=4,
        hidden_size=6,
        num_layers=1,
        n_components=2,
        epochs=2,
        batch_size=8,
        learning_rate=0.01,
        energy_samples=16,
    )


def test_run_mdn_synthetic_writes_training_and_evaluation_artifacts(tmp_path) -> None:
    report = run_mdn_synthetic(tmp_path, tiny_config(seed=31))

    expected = {
        "config.json",
        "training_history.json",
        "model.pt",
        "standardizer.npz",
        "forecast.npz",
        "mixture_diagnostics.json",
        "anomaly_validation.npz",
        "anomaly_test.npz",
        "regime.npz",
        "metrics.json",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    assert report.forecast is not None
    assert report.anomaly is not None

    history = json.loads((tmp_path / "training_history.json").read_text(encoding="utf-8"))
    assert len(history["train_loss"]) == 2
    assert len(history["validation_loss"]) == 2
    diagnostics = json.loads((tmp_path / "mixture_diagnostics.json").read_text(encoding="utf-8"))
    assert set(diagnostics) == {"test", "validation"}


def test_run_mdn_synthetic_forecast_artifact_matches_contract(tmp_path) -> None:
    run_mdn_synthetic(tmp_path, tiny_config(seed=32))

    with np.load(tmp_path / "forecast.npz") as forecast:
        observed = forecast["observed"]
        weights = forecast["weights"]
        means = forecast["means"]
        stds = forecast["stds"]

    assert weights.shape == observed.shape[:-1] + (2,)
    assert means.shape == observed.shape[:-1] + (2, 1)
    assert stds.shape == means.shape
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_mdn_synthetic_cli_writes_metrics(tmp_path) -> None:
    exit_code = main(
        [
            str(tmp_path),
            "--seed",
            "33",
            "--series-length",
            "80",
            "--lookback",
            "8",
            "--stride",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--hidden-size",
            "6",
            "--n-components",
            "2",
            "--n-train-configs",
            "1",
            "--n-validation-configs",
            "1",
            "--n-test-configs",
            "1",
            "--energy-samples",
            "16",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "metrics.json").exists()
