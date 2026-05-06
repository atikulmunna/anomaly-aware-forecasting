import pytest

from aaf.models.joint_loss import JointLossComponents, JointLossConfig


def test_joint_loss_config_accepts_non_negative_weights() -> None:
    JointLossConfig(smoothness_weight=0.0, supervised_regime_weight=1.0).validate()


def test_joint_loss_config_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="smoothness"):
        JointLossConfig(smoothness_weight=-0.1).validate()


def test_joint_loss_components_is_plain_scalar_container() -> None:
    components = JointLossComponents(
        total=1.0,
        forecast_nll=0.8,
        smoothness=0.1,
        supervised_regime=0.1,
    )

    assert components.total == pytest.approx(1.0)
