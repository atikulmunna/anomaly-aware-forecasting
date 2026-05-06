import pytest

from aaf.pipelines.joint_synthetic import JointSyntheticConfig, build_joint_synthetic_datasets


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
