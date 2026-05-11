from pathlib import Path

import numpy as np
import pytest

from aaf.pipelines.smd_baseline import (
    SMDBaselineConfig,
    build_smd_baseline_datasets,
    fit_smd_baseline,
    main,
    run_smd_baseline,
    write_smd_anomaly_artifact,
    write_smd_forecast_artifact,
)


def write_smd_fixture(root: Path, machine_id: str = "machine-1-1") -> None:
    for directory in ("train", "test", "test_label"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "train" / f"{machine_id}.txt").write_text(
        "\n".join(f"{idx},{idx + 1}" for idx in range(12)),
        encoding="utf-8",
    )
    (root / "test" / f"{machine_id}.txt").write_text(
        "\n".join(f"{idx},{idx + 1}" for idx in range(8)),
        encoding="utf-8",
    )
    (root / "test_label" / f"{machine_id}.txt").write_text(
        "0\n0\n1\n0\n0\n0\n0\n0\n",
        encoding="utf-8",
    )


def test_smd_baseline_config_accepts_valid_values(tmp_path) -> None:
    SMDBaselineConfig(root=tmp_path).validate()


def test_smd_baseline_config_rejects_invalid_validation_fraction(tmp_path) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        SMDBaselineConfig(root=tmp_path, validation_fraction=1.0).validate()


def test_smd_baseline_config_keeps_machine_subset(tmp_path: Path) -> None:
    config = SMDBaselineConfig(root=tmp_path, machine_ids=("machine-1-1",))

    assert config.machine_ids == ("machine-1-1",)


def test_build_smd_baseline_datasets_returns_windowed_splits(tmp_path) -> None:
    write_smd_fixture(tmp_path)

    train, validation, test, standardizers = build_smd_baseline_datasets(
        SMDBaselineConfig(
            root=tmp_path,
            lookback=2,
            horizon=1,
            validation_fraction=0.25,
        )
    )

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0
    assert len(standardizers) == 1


def test_fit_smd_baseline_predicts_validation_windows(tmp_path) -> None:
    write_smd_fixture(tmp_path)
    config = SMDBaselineConfig(root=tmp_path, lookback=2, horizon=1, validation_fraction=0.25)
    train, validation, _test, _standardizers = build_smd_baseline_datasets(config)

    forecaster = fit_smd_baseline(train, config)
    forecast = forecaster.predict(validation.windows, horizon=config.horizon)

    assert forecast.weights.shape[0] == len(validation)


def test_smd_baseline_artifact_writers_emit_npz_files(tmp_path) -> None:
    write_smd_fixture(tmp_path)
    config = SMDBaselineConfig(root=tmp_path, lookback=2, horizon=1, validation_fraction=0.25)
    train, validation, _test, _standardizers = build_smd_baseline_datasets(config)
    forecast = fit_smd_baseline(train, config).predict(
        validation.windows,
        horizon=config.horizon,
    )

    write_smd_forecast_artifact(tmp_path / "forecast.npz", validation.targets, forecast)
    write_smd_anomaly_artifact(tmp_path / "anomaly.npz", validation, forecast)

    with np.load(tmp_path / "forecast.npz") as artifact:
        assert artifact["observed"].shape == validation.targets.shape
    assert (tmp_path / "anomaly.npz").exists()


def test_run_smd_baseline_writes_evaluation_artifacts(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "run"
    write_smd_fixture(dataset_root)

    report = run_smd_baseline(
        output_dir,
        SMDBaselineConfig(
            root=dataset_root,
            lookback=2,
            horizon=1,
            validation_fraction=0.25,
            energy_samples=16,
        ),
    )

    expected = {
        "config.json",
        "standardizers.npz",
        "forecast.npz",
        "anomaly_validation.npz",
        "anomaly_test.npz",
        "regime.npz",
        "mixture_diagnostics.json",
        "metrics.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    assert report.forecast is not None
    assert report.anomaly is not None


def test_smd_baseline_cli_writes_metrics(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "run"
    write_smd_fixture(dataset_root)

    exit_code = main(
        [
            str(dataset_root),
            str(output_dir),
            "--machine-id",
            "machine-1-1",
            "--lookback",
            "2",
            "--validation-fraction",
            "0.25",
            "--energy-samples",
            "16",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "metrics.json").exists()
