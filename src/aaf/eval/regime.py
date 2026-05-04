"""Regime-detection evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class RegimeAlignment:
    aligned_pred: IntArray
    mapping: dict[int, int]
    confusion: IntArray


def adjusted_rand_index(true_labels: ArrayLike, pred_labels: ArrayLike) -> float:
    """Compute Adjusted Rand Index without requiring scikit-learn at runtime."""

    true = _label_array(true_labels, name="true_labels")
    pred = _label_array(pred_labels, name="pred_labels")
    _check_same_length(true, pred)

    contingency = _contingency_matrix(true, pred)
    n = int(contingency.sum())
    if n < 2:
        return 1.0

    sum_comb = float(np.sum(_comb2(contingency)))
    row_comb = float(np.sum(_comb2(contingency.sum(axis=1))))
    col_comb = float(np.sum(_comb2(contingency.sum(axis=0))))
    total_comb = float(_comb2(np.asarray(n, dtype=np.int64)))
    expected = row_comb * col_comb / total_comb if total_comb > 0.0 else 0.0
    maximum = 0.5 * (row_comb + col_comb)
    denominator = maximum - expected
    if denominator == 0.0:
        return 1.0
    return float((sum_comb - expected) / denominator)


def align_regime_labels(true_labels: ArrayLike, pred_labels: ArrayLike) -> RegimeAlignment:
    """Align predicted regime IDs to true regime IDs via Hungarian assignment."""

    true = _label_array(true_labels, name="true_labels")
    pred = _label_array(pred_labels, name="pred_labels")
    _check_same_length(true, pred)

    true_values = np.unique(true)
    pred_values = np.unique(pred)
    counts = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    for i, true_value in enumerate(true_values):
        for j, pred_value in enumerate(pred_values):
            counts[i, j] = int(np.sum((true == true_value) & (pred == pred_value)))

    row_ind, col_ind = linear_sum_assignment(-counts)
    mapping = {
        int(pred_values[col]): int(true_values[row])
        for row, col in zip(row_ind, col_ind, strict=True)
    }
    fallback = int(true_values[0])
    aligned = np.asarray([mapping.get(int(label), fallback) for label in pred], dtype=np.int64)
    confusion = _fixed_label_confusion(true, aligned, true_values)
    return RegimeAlignment(aligned_pred=aligned, mapping=mapping, confusion=confusion)


def switch_points(labels: ArrayLike) -> IntArray:
    """Return timesteps t where labels[t] differs from labels[t - 1]."""

    y = _label_array(labels, name="labels")
    if y.size < 2:
        return np.asarray([], dtype=np.int64)
    return np.flatnonzero(y[1:] != y[:-1]).astype(np.int64) + 1


def false_switch_rate_per_1000(true_labels: ArrayLike, pred_labels: ArrayLike) -> float:
    """Return predicted switches not matched by true switches per 1,000 stable transitions."""

    true = _label_array(true_labels, name="true_labels")
    pred = _label_array(pred_labels, name="pred_labels")
    _check_same_length(true, pred)
    if true.size < 2:
        return 0.0

    true_switches = set(int(value) for value in switch_points(true))
    pred_switches = set(int(value) for value in switch_points(pred))
    stable_transitions = (true.size - 1) - len(true_switches)
    if stable_transitions <= 0:
        return 0.0
    false_switches = len(pred_switches - true_switches)
    return float(false_switches * 1000.0 / stable_transitions)


def _label_array(values: ArrayLike, *, name: str) -> IntArray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if not np.all(array == np.floor(array)):
        raise ValueError(f"{name} must contain integer labels")
    return np.asarray(array, dtype=np.int64)


def _check_same_length(true: IntArray, pred: IntArray) -> None:
    if true.shape != pred.shape:
        raise ValueError("true and predicted labels must have the same shape")


def _contingency_matrix(true: IntArray, pred: IntArray) -> IntArray:
    true_inverse = np.unique(true, return_inverse=True)[1]
    pred_inverse = np.unique(pred, return_inverse=True)[1]
    contingency = np.zeros((true_inverse.max() + 1, pred_inverse.max() + 1), dtype=np.int64)
    np.add.at(contingency, (true_inverse, pred_inverse), 1)
    return contingency


def _fixed_label_confusion(true: IntArray, pred: IntArray, labels: IntArray) -> IntArray:
    positions = {int(label): idx for idx, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_value, pred_value in zip(true, pred, strict=True):
        confusion[positions[int(true_value)], positions[int(pred_value)]] += 1
    return confusion


def _comb2(values: IntArray) -> NDArray[np.float64]:
    values_float = values.astype(np.float64)
    return values_float * (values_float - 1.0) / 2.0
