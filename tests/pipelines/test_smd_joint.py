from pathlib import Path

import numpy as np
import pytest

from aaf.pipelines.smd_joint import (
    SMDJointConfig,
    build_smd_joint_datasets,
    predict_smd_joint_splits,
    run_smd_joint,
    smd_joint_loss_config,
    smd_joint_model_config,
    smd_joint_training_config,
    train_smd_joint_model,
    write_smd_joint_anomaly_artifact,
    write_smd_joint_forecast_artifact,
    write_smd_joint_regime_artifact,
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


def tiny_config(root: Path) -> SMDJointConfig:
    return SMDJointConfig(
        root=root,
        validation_fraction=0.25,
        lookback=3,
        horizon=1,
        stride=1,
        n_regimes=2,
        hidden_size=6,
        n_components=2,
        epochs=1,
        batch_size=4,
        learning_rate=0.01,
        energy_samples=16,
    )


def test_smd_joint_config_accepts_valid_values(tmp_path) -> None:
    tiny_config(tmp_path).validate()


def test_smd_joint_config_rejects_invalid_regime_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="n_regimes"):
        SMDJointConfig(root=tmp_path, n_regimes=1).validate()


def test_build_smd_joint_datasets_returns_windowed_splits(tmp_path) -> None:
    write_smd_fixture(tmp_path)

    train, validation, test, standardizers = build_smd_joint_datasets(tiny_config(tmp_path))

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0
    assert len(standardizers) == 1


def test_smd_joint_config_helpers_match_dataset_dimensions(tmp_path) -> None:
    write_smd_fixture(tmp_path)
    config = tiny_config(tmp_path)
    train, _validation, _test, _standardizers = build_smd_joint_datasets(config)

    model_config = smd_joint_model_config(train, config)
    loss_config = smd_joint_loss_config(config)
    training_config = smd_joint_training_config(config)

    assert model_config.input_size == 2
    assert model_config.n_regimes == 2
    assert loss_config.smoothness_weight == config.smoothness_weight
    assert training_config.epochs == 1


def test_train_smd_joint_model_returns_predictions(tmp_path) -> None:
    write_smd_fixture(tmp_path)
    config = tiny_config(tmp_path)
    train, validation, test, _standardizers = build_smd_joint_datasets(config)

    result = train_smd_joint_model(train, validation, config)
    validation_prediction, test_prediction = predict_smd_joint_splits(result, validation, test)

    assert len(result.history.train_loss) == 1
    assert validation_prediction.forecast.weights.shape[0] == len(validation)
    assert test_prediction.regime_probs.shape == test.regime_labels.shape + (config.n_regimes,)


def test_smd_joint_artifact_writers_emit_npz_files(tmp_path) -> None:
    write_smd_fixture(tmp_path)
    config = tiny_config(tmp_path)
    train, validation, test, _standardizers = build_smd_joint_datasets(config)
    result = train_smd_joint_model(train, validation, config)
    _validation_prediction, test_prediction = predict_smd_joint_splits(result, validation, test)

    write_smd_joint_forecast_artifact(
        tmp_path / "forecast.npz",
        test.targets,
        test_prediction.forecast,
    )
    write_smd_joint_anomaly_artifact(tmp_path / "anomaly.npz", test, test_prediction.forecast)
    write_smd_joint_regime_artifact(tmp_path / "regime.npz", test, test_prediction)

    with np.load(tmp_path / "regime.npz") as artifact:
        assert artifact["posterior_probs"].shape == test.regime_labels.shape + (config.n_regimes,)
    assert (tmp_path / "forecast.npz").exists()
    assert (tmp_path / "anomaly.npz").exists()


def test_run_smd_joint_writes_full_run_directory(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "run"
    write_smd_fixture(dataset_root)

    report = run_smd_joint(output_dir, tiny_config(dataset_root))

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
        "regime_diagnostics.json",
        "metrics.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    assert report.forecast is not None
    assert report.anomaly is not None
    assert report.regime is not None
