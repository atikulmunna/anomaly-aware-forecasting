import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from aaf.models.joint import JointOutput  # noqa: E402
from aaf.models.joint_loss import (  # noqa: E402
    JointLossComponents,
    JointLossConfig,
    forecast_nll_loss,
    joint_loss,
    regime_smoothness_loss,
    supervised_regime_loss,
)
from aaf.models.mixture import MixtureParams  # noqa: E402


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


def test_forecast_nll_loss_delegates_to_mixture_nll() -> None:
    output = JointOutput(
        forecast=MixtureParams(
            logits=torch.zeros(1, 1, 1, 1),
            means=torch.zeros(1, 1, 1, 1, 1),
            raw_stds=torch.zeros(1, 1, 1, 1, 1),
        ),
        regime_logits=torch.zeros(1, 1, 2),
    )
    target = torch.zeros(1, 1, 1, 1)

    assert forecast_nll_loss(target, output).item() > 0.0


def test_regime_smoothness_loss_is_zero_for_constant_logits() -> None:
    logits = torch.zeros(2, 4, 3)

    assert regime_smoothness_loss(logits).item() == pytest.approx(0.0)


def test_regime_smoothness_loss_is_positive_for_changes() -> None:
    logits = torch.zeros(1, 2, 2)
    logits[:, 0, 0] = 5.0
    logits[:, 1, 1] = 5.0

    assert regime_smoothness_loss(logits).item() > 0.0


def test_supervised_regime_loss_is_small_for_correct_confident_logits() -> None:
    logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    labels = torch.tensor([[0, 1]])

    assert supervised_regime_loss(logits, labels).item() < 0.01


def test_supervised_regime_loss_rejects_bad_label_shape() -> None:
    with pytest.raises(ValueError, match="regime_labels"):
        supervised_regime_loss(torch.zeros(2, 3, 4), torch.zeros(2, dtype=torch.long))


def test_joint_loss_combines_forecast_smoothness_and_supervision() -> None:
    output = JointOutput(
        forecast=MixtureParams(
            logits=torch.zeros(1, 2, 1, 1),
            means=torch.zeros(1, 2, 1, 1, 1),
            raw_stds=torch.zeros(1, 2, 1, 1, 1),
        ),
        regime_logits=torch.zeros(1, 2, 2),
    )
    target = torch.zeros(1, 2, 1, 1)

    total, components = joint_loss(
        target,
        output,
        JointLossConfig(smoothness_weight=0.1, supervised_regime_weight=0.5),
        regime_labels=torch.zeros(1, 2, dtype=torch.long),
    )

    assert torch.isfinite(total)
    assert components.total >= components.forecast_nll
