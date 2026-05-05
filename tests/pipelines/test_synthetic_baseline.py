import json

import numpy as np
import pytest

from aaf.pipelines.synthetic_baseline import SyntheticBaselineConfig, main, run_synthetic_baseline


def tiny_config(seed: int = 0) -> SyntheticBaselineConfig:
    return SyntheticBaselineConfig(
        seed=seed,
        n_train_configs=2,
        n_validation_configs=1,
        n_test_configs=1,
        series_length=80,
        burn_in=10,
        lookback=8,
        horizon=2,
        stride=3,
        season_length=1,
        n_regimes=3,
        n_channels=2,
        ar_order=2,
        energy_samples=16,
    )


def test_run_synthetic_baseline_writes_evaluation_artifacts(tmp_path) -> None:
    report = run_synthetic_baseline(tmp_path, tiny_config(seed=11))

    expected_files = {
        "config.json",
        "standardizer.npz",
        "forecast.npz",
        "mixture_diagnostics.json",
        "anomaly_validation.npz",
        "anomaly_test.npz",
        "regime.npz",
        "metrics.json",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})
    assert report.forecast is not None
    assert report.anomaly is not None
    assert report.regime is not None
    diagnostics = json.loads((tmp_path / "mixture_diagnostics.json").read_text(encoding="utf-8"))
    assert set(diagnostics) == {"test", "validation"}


def test_run_synthetic_baseline_artifacts_match_expected_shapes(tmp_path) -> None:
    config = tiny_config(seed=12)

    run_synthetic_baseline(tmp_path, config)

    with np.load(tmp_path / "forecast.npz") as forecast:
        observed = forecast["observed"]
        weights = forecast["weights"]
        means = forecast["means"]
        stds = forecast["stds"]

    assert observed.ndim == 3
    assert observed.shape[1:] == (config.horizon, config.n_channels)
    assert weights.shape == observed.shape[:-1] + (1,)
    assert means.shape == observed.shape[:-1] + (1, config.n_channels)
    assert stds.shape == means.shape

    with np.load(tmp_path / "anomaly_test.npz") as anomaly:
        assert anomaly["scores"].shape == anomaly["labels"].shape
        assert anomaly["labels"].sum() > 0


def test_run_synthetic_baseline_is_reproducible_for_seed(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    run_synthetic_baseline(first_dir, tiny_config(seed=22))
    run_synthetic_baseline(second_dir, tiny_config(seed=22))

    first_metrics = json.loads((first_dir / "metrics.json").read_text(encoding="utf-8"))
    second_metrics = json.loads((second_dir / "metrics.json").read_text(encoding="utf-8"))
    assert first_metrics == second_metrics

    with (
        np.load(first_dir / "forecast.npz") as first,
        np.load(second_dir / "forecast.npz") as second,
    ):
        assert np.array_equal(first["observed"], second["observed"])
        assert np.array_equal(first["means"], second["means"])


def test_run_synthetic_baseline_refuses_non_empty_directory_without_overwrite(tmp_path) -> None:
    (tmp_path / "existing.txt").write_text("already here", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        run_synthetic_baseline(tmp_path, tiny_config())


def test_synthetic_baseline_cli_writes_metrics(tmp_path) -> None:
    exit_code = main(
        [
            str(tmp_path),
            "--seed",
            "5",
            "--series-length",
            "80",
            "--lookback",
            "8",
            "--horizon",
            "2",
            "--stride",
            "4",
            "--n-train-configs",
            "2",
            "--n-validation-configs",
            "1",
            "--n-test-configs",
            "1",
            "--n-channels",
            "2",
            "--energy-samples",
            "16",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "metrics.json").exists()
