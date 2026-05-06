import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from aaf.models.joint import JointMDNLSTMConfig, JointMDNLSTMForecaster, JointOutput  # noqa: E402
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


def test_joint_config_accepts_valid_dimensions() -> None:
    JointMDNLSTMConfig(input_size=2, output_size=2, n_regimes=3).validate()


def test_joint_config_rejects_single_regime() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        JointMDNLSTMConfig(input_size=1, output_size=1, n_regimes=1).validate()


def test_joint_model_initializes_regime_and_forecast_heads() -> None:
    model = JointMDNLSTMForecaster(
        JointMDNLSTMConfig(input_size=2, output_size=2, n_regimes=3, hidden_size=8)
    )

    assert model.regime_head.out_features == 3
    assert model.logit_head.in_features == 11


def test_joint_model_forward_shapes() -> None:
    model = JointMDNLSTMForecaster(
        JointMDNLSTMConfig(
            input_size=2,
            output_size=2,
            n_regimes=3,
            hidden_size=8,
            num_layers=1,
            horizon=2,
            n_components=4,
        )
    )

    output = model(torch.zeros(5, 7, 2))

    assert tuple(output.regime_logits.shape) == (5, 7, 3)
    assert tuple(output.forecast.logits.shape) == (5, 7, 2, 4)
    assert tuple(output.forecast.means.shape) == (5, 7, 2, 4, 2)


def test_joint_model_forecast_last_keeps_single_time_dimension() -> None:
    model = JointMDNLSTMForecaster(
        JointMDNLSTMConfig(input_size=1, output_size=1, n_regimes=2, hidden_size=4, num_layers=1)
    )

    output = model.forecast_last(torch.zeros(3, 6, 1))

    assert tuple(output.regime_logits.shape) == (3, 1, 2)
    assert tuple(output.forecast.logits.shape) == (3, 1, 1, 3)
