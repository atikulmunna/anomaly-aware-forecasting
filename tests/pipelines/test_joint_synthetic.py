import numpy as np
import pytest

from aaf.eval.forecasting import MixtureForecast
from aaf.pipelines.joint_synthetic import (
    JointSyntheticConfig,
    build_joint_synthetic_datasets,
    write_joint_anomaly_artifact,
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
