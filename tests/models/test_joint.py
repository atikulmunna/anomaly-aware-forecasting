import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from aaf.models.joint import JointOutput  # noqa: E402
from aaf.models.mixture import MixtureParams  # noqa: E402


def test_joint_output_validates_matching_batch_time_dimensions() -> None:
    output = JointOutput(
        forecast=MixtureParams(
            logits=torch.zeros(2, 3, 1, 2),
            means=torch.zeros(2, 3, 1, 2, 1),
            raw_stds=torch.zeros(2, 3, 1, 2, 1),
        ),
        regime_logits=torch.zeros(2, 3, 4),
    )

    output.validate()
    assert tuple(output.regime_probs.shape) == (2, 3, 4)
    assert torch.allclose(output.regime_probs.sum(dim=-1), torch.ones(2, 3))


def test_joint_output_rejects_mismatched_regime_shape() -> None:
    output = JointOutput(
        forecast=MixtureParams(
            logits=torch.zeros(2, 3, 1, 2),
            means=torch.zeros(2, 3, 1, 2, 1),
            raw_stds=torch.zeros(2, 3, 1, 2, 1),
        ),
        regime_logits=torch.zeros(2, 4, 4),
    )

    with pytest.raises(ValueError, match="batch and time"):
        output.validate()
