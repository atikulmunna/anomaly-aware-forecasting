import numpy as np
import pytest

from aaf.eval.regime import (
    adjusted_rand_index,
    align_regime_labels,
    false_switch_rate_per_1000,
    switch_points,
)


def test_adjusted_rand_index_is_one_for_permuted_labels() -> None:
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1])

    assert adjusted_rand_index(true, pred) == pytest.approx(1.0)


def test_adjusted_rand_index_is_less_than_one_for_mixed_clusters() -> None:
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 0, 1, 0, 1])

    assert adjusted_rand_index(true, pred) < 1.0


def test_align_regime_labels_uses_hungarian_assignment() -> None:
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1])

    alignment = align_regime_labels(true, pred)

    assert alignment.aligned_pred.tolist() == true.tolist()
    assert alignment.mapping == {2: 0, 0: 1, 1: 2}
    assert alignment.confusion.tolist() == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_switch_points_returns_change_timesteps() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 1])

    assert switch_points(labels).tolist() == [2, 4, 7]


def test_false_switch_rate_counts_switches_during_true_stable_transitions() -> None:
    true = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 1, 1, 1, 2, 2])

    assert false_switch_rate_per_1000(true, pred) == pytest.approx(500.0)


def test_rejects_mismatched_label_lengths() -> None:
    with pytest.raises(ValueError, match="same shape"):
        adjusted_rand_index(np.array([0, 1]), np.array([0]))
