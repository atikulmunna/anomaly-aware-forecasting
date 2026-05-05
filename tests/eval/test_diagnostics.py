import math

import numpy as np
import pytest

from aaf.eval.diagnostics import (
    active_component_count,
    component_mean_weights,
    effective_component_count,
    mean_pairwise_distance,
    mixture_entropy_values,
    normalized_mixture_entropy_values,
    normalized_weights,
    std_summary,
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


def test_component_mean_weights_average_over_forecasts() -> None:
    weights = np.array([[1.0, 0.0], [0.5, 0.5]])

    assert component_mean_weights(weights).tolist() == [0.75, 0.25]


def test_active_component_count_uses_mean_assignment_threshold() -> None:
    weights = np.array([[0.99, 0.01, 0.0], [0.99, 0.01, 0.0]])

    assert active_component_count(weights, threshold=0.01) == 2
    assert active_component_count(weights, threshold=0.02) == 1


def test_effective_component_count_is_one_for_collapsed_weights() -> None:
    weights = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    assert effective_component_count(weights) == pytest.approx(1.0)


def test_std_summary_reports_distribution_statistics() -> None:
    summary = std_summary(np.array([1.0, 2.0, 3.0]))

    assert summary["min"] == pytest.approx(1.0)
    assert summary["median"] == pytest.approx(2.0)
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["max"] == pytest.approx(3.0)


def test_std_summary_rejects_non_positive_stds() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        std_summary(np.array([1.0, 0.0]))


def test_mean_pairwise_distance_is_zero_for_single_component() -> None:
    assert mean_pairwise_distance(np.zeros((4, 1, 2))) == pytest.approx(0.0)


def test_mean_pairwise_distance_uses_euclidean_component_distance() -> None:
    means = np.array([[[0.0, 0.0], [3.0, 4.0]]])

    assert mean_pairwise_distance(means) == pytest.approx(5.0)


def test_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        normalized_weights(np.array([1.0, -1.0]))
