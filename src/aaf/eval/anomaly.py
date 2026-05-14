"""Range-based anomaly detection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, order=True)
class Range:
    """Half-open interval [start, end) for contiguous anomalous regions."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("range start must be non-negative")
        if self.end <= self.start:
            raise ValueError("range end must be greater than start")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap(self, other: Range) -> int:
        return max(0, min(self.end, other.end) - max(self.start, other.start))


@dataclass(frozen=True)
class RangeMetrics:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class DetectionDelay:
    mean: float
    median: float
    missed: int
    detected: int


@dataclass(frozen=True)
class BinaryConfusion:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def precision(self) -> float:
        predicted_positive = self.true_positive + self.false_positive
        if predicted_positive == 0:
            return 1.0
        return float(self.true_positive / predicted_positive)

    @property
    def recall(self) -> float:
        actual_positive = self.true_positive + self.false_negative
        if actual_positive == 0:
            return 1.0
        return float(self.true_positive / actual_positive)

    @property
    def true_positive_rate(self) -> float:
        return self.recall

    @property
    def false_positive_rate(self) -> float:
        actual_negative = self.false_positive + self.true_negative
        if actual_negative == 0:
            return 0.0
        return float(self.false_positive / actual_negative)


def binary_ranges(labels: ArrayLike) -> list[Range]:
    """Convert a one-dimensional binary label sequence into half-open ranges."""

    y = _binary_array(labels, name="labels")
    ranges: list[Range] = []
    start: int | None = None
    for idx, value in enumerate(y):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            ranges.append(Range(start, idx))
            start = None
    if start is not None:
        ranges.append(Range(start, len(y)))
    return ranges


def threshold_scores(scores: ArrayLike, threshold: float) -> BoolArray:
    """Return anomaly predictions where larger scores are more anomalous."""

    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if np.any(~np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    return np.asarray(score_array >= threshold, dtype=np.bool_)


def threshold_candidates(scores: ArrayLike) -> FloatArray:
    """Return deterministic score thresholds for threshold-free sweeps."""

    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if np.any(~np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    return _threshold_candidates(score_array)


def binary_confusion(true_labels: ArrayLike, pred_labels: ArrayLike) -> BinaryConfusion:
    """Return binary confusion counts for one-dimensional anomaly labels."""

    true = _binary_array(true_labels, name="true_labels")
    pred = _binary_array(pred_labels, name="pred_labels")
    if true.shape != pred.shape:
        raise ValueError("true_labels and pred_labels must have the same shape")
    return BinaryConfusion(
        true_positive=int(np.sum(true & pred)),
        false_positive=int(np.sum(~true & pred)),
        true_negative=int(np.sum(~true & ~pred)),
        false_negative=int(np.sum(true & ~pred)),
    )


def range_precision_recall(true_labels: ArrayLike, pred_labels: ArrayLike) -> RangeMetrics:
    """Compute overlap-based range precision, recall, and F1.

    This intentionally scores contiguous anomalous regions instead of pointwise labels. Recall is
    the average fraction of each true range covered by predictions; precision is the average
    fraction of each predicted range covered by true anomalies.
    """

    true = binary_ranges(true_labels)
    pred = binary_ranges(pred_labels)
    recall = _average_overlap_fraction(true, pred)
    precision = _average_overlap_fraction(pred, true)
    f1 = _f1(precision, recall)
    return RangeMetrics(precision=precision, recall=recall, f1=f1)


def select_threshold_by_range_f1(
    scores: ArrayLike,
    true_labels: ArrayLike,
) -> tuple[float, RangeMetrics]:
    """Select a validation threshold that maximizes range-based F1."""

    score_array = np.asarray(scores, dtype=np.float64)
    _ = _binary_array(true_labels, name="true_labels")
    if score_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if np.any(~np.isfinite(score_array)):
        raise ValueError("scores must be finite")

    candidates = _threshold_candidates(score_array)
    best_threshold = float(candidates[0])
    best_metrics = range_precision_recall(
        true_labels,
        threshold_scores(score_array, best_threshold),
    )
    for threshold in candidates[1:]:
        metrics = range_precision_recall(
            true_labels,
            threshold_scores(score_array, float(threshold)),
        )
        if _is_better(metrics, best_metrics):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def detection_delay(true_labels: ArrayLike, pred_labels: ArrayLike) -> DetectionDelay:
    """Return detection delay statistics for true anomalous ranges."""

    true = binary_ranges(true_labels)
    predictions = _binary_array(pred_labels, name="pred_labels")
    delays: list[int] = []
    missed = 0
    for region in true:
        hits = np.flatnonzero(predictions[region.start : region.end])
        if hits.size == 0:
            missed += 1
        else:
            delays.append(int(hits[0]))

    if not delays:
        return DetectionDelay(mean=float("nan"), median=float("nan"), missed=missed, detected=0)
    delay_array = np.asarray(delays, dtype=np.float64)
    return DetectionDelay(
        mean=float(np.mean(delay_array)),
        median=float(np.median(delay_array)),
        missed=missed,
        detected=len(delays),
    )


def false_alarm_rate_per_1000(true_labels: ArrayLike, pred_labels: ArrayLike) -> float:
    """Count predicted anomaly-range starts in stable regions per 1,000 stable timesteps."""

    true = _binary_array(true_labels, name="true_labels")
    pred_ranges = binary_ranges(pred_labels)
    stable_count = int(np.sum(~true))
    if stable_count == 0:
        return 0.0

    false_starts = sum(1 for region in pred_ranges if not true[region.start])
    return float(false_starts * 1000.0 / stable_count)


def _binary_array(values: ArrayLike, *, name: str) -> BoolArray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isin(array, [0, 1, False, True]).all():
        raise ValueError(f"{name} must contain only binary values")
    return np.asarray(array, dtype=np.bool_)


def _average_overlap_fraction(subject: list[Range], reference: list[Range]) -> float:
    if not subject:
        return 1.0 if not reference else 0.0
    if not reference:
        return 0.0

    fractions = []
    for region in subject:
        overlap = sum(region.overlap(other) for other in reference)
        fractions.append(min(1.0, overlap / region.length))
    return float(np.mean(fractions))


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _threshold_candidates(scores: FloatArray) -> FloatArray:
    unique = np.unique(scores)
    if unique.size == 1:
        return unique
    above_max = np.nextafter(unique[-1], np.inf)
    return np.concatenate(([above_max], unique[::-1]))


def _is_better(candidate: RangeMetrics, incumbent: RangeMetrics) -> bool:
    if candidate.f1 != incumbent.f1:
        return candidate.f1 > incumbent.f1
    if candidate.recall != incumbent.recall:
        return candidate.recall > incumbent.recall
    return candidate.precision > incumbent.precision
