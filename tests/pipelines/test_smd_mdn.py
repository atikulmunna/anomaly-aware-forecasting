from pathlib import Path

import numpy as np
import pytest

import aaf.pipelines.smd_mdn as smd_mdn_module
from aaf.pipelines.smd_mdn import (
    SMDMDNConfig,
    build_smd_mdn_datasets,
    main,
    predict_smd_mdn_splits,
    run_smd_mdn,
    smd_mdn_model_config,
    smd_mdn_training_config,
    train_smd_mdn_model,
    write_smd_mdn_anomaly_artifact,
    write_smd_mdn_forecast_artifact,
)


def write_smd_fixture(root: Path, machine_id: str = "machine-1-1") -> None:
    for directory in ("train", "test", "test_label"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "train" / f"{machine_id}.txt").write_text(
        "\n".join(f"{idx},{idx + 1}" for idx in range(16)),
        encoding="utf-8",
    )
    (root / "test" / f"{machine_id}.txt").write_text(
        "\n".join(f"{idx},{idx + 1}" for idx in range(10)),
        encoding="utf-8",
    )
    (root / "test_label" / f"{machine_id}.txt").write_text(
        "0\n0\n1\n0\n0\n0\n0\n0\n0\n0\n",
        encoding="utf-8",
    )


def tiny_config(root: Path) -> SMDMDNConfig:
    return SMDMDNConfig(
        root=root,
        validation_fraction=0.25,
        lookback=3,
        horizon=1,
        stride=1,
        hidden_size=6,
        n_components=2,
        epochs=1,
        batch_size=4,
        learning_rate=0.01,
        energy_samples=16,
        device="cpu",
    )


def test_smd_mdn_config_accepts_valid_values(tmp_path) -> None:
    tiny_config(tmp_path).validate()


def test_smd_mdn_config_rejects_invalid_component_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="n_components"):
        SMDMDNConfig(root=tmp_path, n_components=0).validate()


def test_smd_mdn_config_rejects_unknown_anomaly_score_method(tmp_path) -> None:
    with pytest.raises(ValueError, match="anomaly_score_method"):
        SMDMDNConfig(root=tmp_path, anomaly_score_method="unknown").validate()


def test_smd_mdn_config_rejects_unknown_threshold_strategy(tmp_path) -> None:
    with pytest.raises(ValueError, match="threshold_strategy"):
        SMDMDNConfig(root=tmp_path, threshold_strategy="unknown").validate()


def test_smd_mdn_config_rejects_invalid_persistence(tmp_path) -> None:
    with pytest.raises(ValueError, match="persistence_count"):
        SMDMDNConfig(
            root=tmp_path,
            anomaly_persistence_window=2,
            anomaly_persistence_count=3,
        ).validate()


def test_build_smd_mdn_datasets_returns_windowed_splits(tmp_path) -> None:
    write_smd_fixture(tmp_path)

    train, validation, test, standardizers = build_smd_mdn_datasets(tiny_config(tmp_path))

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0
    assert len(standardizers) == 1


def test_smd_mdn_config_helpers_match_dataset_dimensions(tmp_path) -> None:
    write_smd_fixture(tmp_path)
    config = tiny_config(tmp_path)
    train, _validation, _test, _standardizers = build_smd_mdn_datasets(config)

    model_config = smd_mdn_model_config(train, config)
    training_config = smd_mdn_training_config(config)

    assert model_config.input_size == 2
    assert model_config.n_components == 2
    assert training_config.epochs == 1
    assert training_config.device == "cpu"
    assert config.anomaly_score_method == "mean_nll"


def test_train_smd_mdn_model_returns_predictions(tmp_path) -> None:
    write_smd_fixture(tmp_path)
    config = tiny_config(tmp_path)
    train, validation, test, _standardizers = build_smd_mdn_datasets(config)

    result = train_smd_mdn_model(train, validation, config)
    validation_forecast, test_forecast = predict_smd_mdn_splits(result, validation, test)

    assert len(result.history.train_loss) == 1
    assert validation_forecast.weights.shape[0] == len(validation)
    assert test_forecast.weights.shape[0] == len(test)


def test_predict_smd_mdn_splits_forwards_device(monkeypatch) -> None:
    seen_devices: list[str] = []

    def fake_predict(_model, dataset, *, device="cpu"):
        seen_devices.append(device)
        return dataset

    monkeypatch.setattr(smd_mdn_module, "predict_mdn_lstm", fake_predict)

    validation_forecast, test_forecast = smd_mdn_module.predict_smd_mdn_splits(
        result=type("Result", (), {"model": object()})(),
        validation_dataset="validation",
        test_dataset="test",
        device="cuda",
    )

    assert validation_forecast == "validation"
    assert test_forecast == "test"
    assert seen_devices == ["cuda", "cuda"]


def test_smd_mdn_artifact_writers_emit_npz_files(tmp_path) -> None:
    write_smd_fixture(tmp_path)
    config = tiny_config(tmp_path)
    train, validation, test, _standardizers = build_smd_mdn_datasets(config)
    result = train_smd_mdn_model(train, validation, config)
    _validation_forecast, test_forecast = predict_smd_mdn_splits(result, validation, test)

    write_smd_mdn_forecast_artifact(tmp_path / "forecast.npz", test.targets, test_forecast)
    write_smd_mdn_anomaly_artifact(
        tmp_path / "anomaly.npz",
        test,
        test_forecast,
        method="channel_max_nll",
    )

    with np.load(tmp_path / "forecast.npz") as artifact:
        assert artifact["weights"].shape[0] == len(test)
    with np.load(tmp_path / "anomaly.npz") as artifact:
        assert artifact["scores"].shape == test.anomaly_labels.shape
    assert (tmp_path / "anomaly.npz").exists()


def test_run_smd_mdn_writes_full_run_directory(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "run"
    write_smd_fixture(dataset_root)

    report = run_smd_mdn(output_dir, tiny_config(dataset_root))

    expected = {
        "config.json",
        "training_history.json",
        "model.pt",
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
    assert report.regime is not None


def test_smd_mdn_cli_writes_metrics(tmp_path) -> None:
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
            "3",
            "--validation-fraction",
            "0.25",
            "--hidden-size",
            "6",
            "--n-components",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.01",
            "--energy-samples",
            "16",
            "--device",
            "cpu",
            "--anomaly-score-method",
            "channel_mean_nll",
            "--threshold-strategy",
            "validation_quantile_95",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "metrics.json").exists()
