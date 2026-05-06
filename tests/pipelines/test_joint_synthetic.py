import json

import numpy as np
import pytest

from aaf.eval.forecasting import MixtureForecast
from aaf.pipelines.joint_synthetic import (
    JointSyntheticConfig,
    build_joint_synthetic_datasets,
    joint_loss_config,
    joint_model_config,
    joint_training_config,
    train_joint_synthetic_model,
    write_joint_anomaly_artifact,
    write_joint_config_artifact,
    write_joint_forecast_artifact,
    write_joint_regime_artifact,
)
from aaf.train.joint_loop import JointPrediction


def test_joint_synthetic_config_accepts_valid_values() -> None:
    JointSyntheticConfig().validate()


def test_joint_synthetic_config_rejects_negative_loss_weights() -> None:
    with pytest.raises(ValueError, match="smoothness"):
        JointSyntheticConfig(smoothness_weight=-0.1).validate()


def test_build_joint_synthetic_datasets_returns_windowed_splits() -> None:
    config = JointSyntheticConfig(
        n_train_configs=1,
        n_validation_configs=1,
        n_test_configs=1,
        series_length=80,
        burn_in=10,
        lookback=8,
        stride=4,
    )

    train, validation, test, standardizer = build_joint_synthetic_datasets(config)

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0
    assert standardizer.mean.shape == (1,)


def test_joint_model_config_matches_pipeline_dimensions() -> None:
    config = JointSyntheticConfig(
        n_channels=2,
        n_regimes=4,
        hidden_size=9,
        n_components=5,
    )

    model_config = joint_model_config(config)

    assert model_config.input_size == 2
    assert model_config.output_size == 2
    assert model_config.n_regimes == 4
    assert model_config.hidden_size == 9
    assert model_config.n_components == 5


def test_joint_loss_config_uses_pipeline_weights() -> None:
    config = JointSyntheticConfig(smoothness_weight=0.25, supervised_regime_weight=0.5)

    loss_config = joint_loss_config(config)

    assert loss_config.smoothness_weight == 0.25
    assert loss_config.supervised_regime_weight == 0.5


def test_joint_training_config_uses_pipeline_training_values() -> None:
    config = JointSyntheticConfig(seed=11, epochs=3, batch_size=12, learning_rate=0.02)

    training_config = joint_training_config(config)

    assert training_config.seed == 11
    assert training_config.epochs == 3
    assert training_config.batch_size == 12
    assert training_config.learning_rate == 0.02


def test_train_joint_synthetic_model_returns_history() -> None:
    config = JointSyntheticConfig(
        seed=12,
        n_train_configs=1,
        n_validation_configs=1,
        n_test_configs=1,
        series_length=80,
        burn_in=10,
        lookback=8,
        stride=4,
        hidden_size=6,
        n_components=2,
        epochs=1,
        batch_size=8,
        learning_rate=0.01,
        supervised_regime_weight=0.1,
    )
    train, validation, _test, _standardizer = build_joint_synthetic_datasets(config)

    result = train_joint_synthetic_model(train, validation, config)

    assert len(result.history.train_loss) == 1
    assert len(result.history.validation_loss) == 1


def test_joint_artifact_writers_emit_expected_npz_files(tmp_path) -> None:
    _train, validation, _test, _standardizer = build_joint_synthetic_datasets(
        JointSyntheticConfig(series_length=80, burn_in=10, lookback=8, stride=4)
    )
    forecast = MixtureForecast.from_arrays(
        weights=np.ones(validation.targets.shape[:-1] + (1,)),
        means=validation.targets[..., np.newaxis, :],
        stds=np.ones(validation.targets.shape[:-1] + (1, validation.targets.shape[-1])),
    )

    write_joint_forecast_artifact(tmp_path / "forecast.npz", validation.targets, forecast)
    write_joint_anomaly_artifact(tmp_path / "anomaly.npz", validation, forecast)

    assert (tmp_path / "forecast.npz").exists()
    assert (tmp_path / "anomaly.npz").exists()


def test_joint_regime_artifact_writer_includes_posteriors(tmp_path) -> None:
    _train, validation, _test, _standardizer = build_joint_synthetic_datasets(
        JointSyntheticConfig(series_length=80, burn_in=10, lookback=8, stride=4)
    )
    probs = np.zeros(validation.regime_labels.shape + (3,))
    probs[..., 0] = 1.0
    prediction = JointPrediction(
        forecast=MixtureForecast.from_arrays(
            weights=np.ones(validation.targets.shape[:-1] + (1,)),
            means=validation.targets[..., np.newaxis, :],
            stds=np.ones(validation.targets.shape[:-1] + (1, validation.targets.shape[-1])),
        ),
        regime_probs=probs,
        regime_labels=np.zeros_like(validation.regime_labels),
    )

    write_joint_regime_artifact(tmp_path / "regime.npz", validation, prediction)

    with np.load(tmp_path / "regime.npz") as artifact:
        assert artifact["true_labels"].shape == validation.regime_labels.shape
        assert artifact["pred_labels"].shape == validation.regime_labels.shape
        assert artifact["posterior_probs"].shape == probs.shape


def test_joint_config_artifact_writer_serializes_config(tmp_path) -> None:
    config = JointSyntheticConfig(seed=42, hidden_size=7)

    write_joint_config_artifact(tmp_path / "config.json", config)

    payload = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert payload["hidden_size"] == 7
