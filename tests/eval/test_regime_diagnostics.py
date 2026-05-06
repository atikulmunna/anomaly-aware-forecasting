import math

import numpy as np
import pytest

from aaf.eval.regime_diagnostics import (
    mean_regime_confidence,
    normalized_regime_entropy_values,
    posterior_argmax_labels,
    posterior_switch_count,
    regime_confidence_values,
    regime_posterior_diagnostics,
    regime_posterior_entropy_values,
)


def test_regime_entropy_matches_uniform_known_value() -> None:
    values = regime_posterior_entropy_values(np.array([[0.5, 0.5]]))

    assert values.item() == pytest.approx(math.log(2.0))


def test_normalized_regime_entropy_scales_by_regime_count() -> None:
    values = normalized_regime_entropy_values(np.array([[0.25, 0.25, 0.25, 0.25]]))

    assert values.item() == pytest.approx(1.0)


def test_regime_entropy_rejects_non_normalized_probs() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        regime_posterior_entropy_values(np.array([[1.0, 1.0]]))


def test_regime_confidence_uses_max_posterior_probability() -> None:
    probs = np.array([[0.1, 0.9], [0.4, 0.6]])

    assert regime_confidence_values(probs).tolist() == [0.9, 0.6]
    assert mean_regime_confidence(probs) == pytest.approx(0.75)


def test_posterior_switch_count_uses_argmax_labels() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3]])

    assert posterior_argmax_labels(probs).tolist() == [0, 1, 1, 0]
    assert posterior_switch_count(probs) == 2


def test_regime_posterior_diagnostics_returns_serializable_summary() -> None:
    probs = np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])

    diagnostics = regime_posterior_diagnostics(probs)
    payload = diagnostics.to_dict()

    assert payload["n_regimes"] == 2
    assert payload["switch_count"] == 1
    assert payload["confidence_mean"] == pytest.approx((0.8 + 0.7 + 0.9) / 3.0)
