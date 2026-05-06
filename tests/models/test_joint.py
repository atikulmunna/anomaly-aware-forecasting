import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from aaf.models.joint import JointMDNLSTMConfig, JointMDNLSTMForecaster, JointOutput  # noqa: E402
from aaf.models.mixture import (  # noqa: E402
    MixtureParams,  # noqa: E402
    mixture_nll,  # noqa: E402
)


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


def test_joint_model_accepts_regime_logit_override() -> None:
    model = JointMDNLSTMForecaster(
        JointMDNLSTMConfig(input_size=1, output_size=1, n_regimes=2, hidden_size=4, num_layers=1)
    )
    history = torch.zeros(2, 5, 1)
    override = torch.zeros(2, 5, 2)
    override[..., 1] = 5.0

    output = model(history, regime_logits_override=override)

    assert torch.allclose(output.regime_logits, override)


def test_joint_model_rejects_bad_regime_logit_override_shape() -> None:
    model = JointMDNLSTMForecaster(
        JointMDNLSTMConfig(input_size=1, output_size=1, n_regimes=2, hidden_size=4, num_layers=1)
    )

    with pytest.raises(ValueError, match="override"):
        model(torch.zeros(2, 5, 1), regime_logits_override=torch.zeros(2, 4, 2))


def test_forecast_nll_depends_on_regime_logits() -> None:
    model = JointMDNLSTMForecaster(
        JointMDNLSTMConfig(
            input_size=1,
            output_size=1,
            n_regimes=2,
            hidden_size=8,
            num_layers=1,
            horizon=1,
            n_components=2,
        )
    )
    history = torch.randn(4, 6, 1)
    target = torch.randn(4, 1, 1, 1)
    baseline = model.forecast_last(history)
    override = baseline.regime_logits.repeat(1, history.shape[1], 1)
    override = override.clone()
    override[..., 0] += 10.0
    perturbed = model.forecast_last(history, regime_logits_override=override)

    baseline_nll = mixture_nll(target, baseline.forecast)
    perturbed_nll = mixture_nll(target, perturbed.forecast)

    assert torch.abs(perturbed_nll - baseline_nll).item() > 1e-6
