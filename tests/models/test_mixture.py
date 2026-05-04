import math

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from aaf.models.mixture import (  # noqa: E402
    MixtureParams,
    mixture_mean,
    mixture_nll,
    mixture_nll_values,
)


def test_single_standard_normal_nll_matches_known_value() -> None:
    params = MixtureParams(
        logits=torch.zeros(1, 1, 1, 1),
        means=torch.zeros(1, 1, 1, 1, 1),
        raw_stds=torch.log(torch.expm1(torch.ones(1, 1, 1, 1, 1) - 1e-3)),
    )
    target = torch.zeros(1, 1, 1, 1)

    assert mixture_nll(target, params).item() == pytest.approx(0.5 * math.log(2.0 * math.pi))


def test_mixture_nll_values_preserve_batch_time_horizon_shape() -> None:
    params = MixtureParams(
        logits=torch.zeros(2, 3, 4, 1),
        means=torch.zeros(2, 3, 4, 1, 2),
        raw_stds=torch.zeros(2, 3, 4, 1, 2),
    )
    target = torch.zeros(2, 3, 4, 2)

    values = mixture_nll_values(target, params)

    assert tuple(values.shape) == (2, 3, 4)
    assert mixture_nll(target, params).item() == pytest.approx(values.mean().item())


def test_mixture_mean_uses_softmax_weights() -> None:
    params = MixtureParams(
        logits=torch.tensor([[[[0.0, math.log(3.0)]]]]),
        means=torch.tensor([[[[[0.0], [4.0]]]]]),
        raw_stds=torch.zeros(1, 1, 1, 2, 1),
    )

    assert mixture_mean(params).item() == pytest.approx(3.0)


def test_invalid_target_shape_is_rejected() -> None:
    params = MixtureParams(
        logits=torch.zeros(1, 1, 1, 1),
        means=torch.zeros(1, 1, 1, 1, 1),
        raw_stds=torch.zeros(1, 1, 1, 1, 1),
    )

    with pytest.raises(ValueError, match="target"):
        mixture_nll(torch.zeros(1, 1, 1), params)


def test_invalid_parameter_shapes_are_rejected() -> None:
    params = MixtureParams(
        logits=torch.zeros(1, 1, 1, 2),
        means=torch.zeros(1, 1, 1, 1, 1),
        raw_stds=torch.zeros(1, 1, 1, 1, 1),
    )

    with pytest.raises(ValueError, match="logits shape"):
        params.validate()
