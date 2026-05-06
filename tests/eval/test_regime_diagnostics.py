import math

import numpy as np
import pytest

from aaf.eval.regime_diagnostics import (
    normalized_regime_entropy_values,
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
