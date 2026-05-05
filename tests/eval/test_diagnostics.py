import math

import numpy as np
import pytest

from aaf.eval.diagnostics import (
    mixture_entropy_values,
    normalized_mixture_entropy_values,
    normalized_weights,
)


def test_normalized_weights_rescales_component_axis() -> None:
    weights = normalized_weights(np.array([[2.0, 2.0], [1.0, 3.0]]))

    assert weights.tolist() == [[0.5, 0.5], [0.25, 0.75]]


def test_mixture_entropy_matches_uniform_known_value() -> None:
    entropy = mixture_entropy_values(np.array([[1.0, 1.0, 1.0]]))

    assert entropy.item() == pytest.approx(math.log(3.0))


def test_normalized_entropy_is_zero_for_single_component() -> None:
    entropy = normalized_mixture_entropy_values(np.ones((2, 1)))

    assert entropy.tolist() == [0.0, 0.0]


def test_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        normalized_weights(np.array([1.0, -1.0]))
