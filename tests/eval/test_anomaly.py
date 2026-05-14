import math

import numpy as np
import pytest

from aaf.eval.anomaly import (
    Range,
    average_precision_score,
    binary_confusion,
    binary_ranges,
    detection_delay,
    false_alarm_rate_per_1000,
    precision_recall_curve,
    range_curve,
    range_precision_recall,
    roc_auc_score,
    roc_curve,
    select_threshold_by_range_f1,
    threshold_candidates,
    threshold_free_metrics,
    threshold_scores,
    trapezoidal_area,
    vus_pr_score,
    vus_roc_score,
)


def test_binary_ranges_uses_half_open_intervals() -> None:
    labels = np.array([0, 1, 1, 0, 1, 1, 1])

    assert binary_ranges(labels) == [Range(1, 3), Range(4, 7)]


def test_range_precision_recall_is_perfect_for_identical_ranges() -> None:
    labels = np.array([0, 1, 1, 0, 1])

    metrics = range_precision_recall(labels, labels)

    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)


def test_range_precision_recall_penalizes_partial_overlap() -> None:
    true = np.array([0, 1, 1, 1, 1, 0])
    pred = np.array([0, 0, 1, 1, 0, 0])

    metrics = range_precision_recall(true, pred)

    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(2.0 / 3.0)


def test_threshold_scores_marks_large_scores_as_anomalous() -> None:
    scores = np.array([0.1, 0.5, 0.9])

    assert threshold_scores(scores, 0.5).tolist() == [False, True, True]


def test_threshold_candidates_include_above_max_and_descending_scores() -> None:
    candidates = threshold_candidates(np.array([0.2, 0.8, 0.2]))

    assert candidates[0] > 0.8
    assert candidates[1:].tolist() == [0.8, 0.2]


def test_binary_confusion_counts_predictions() -> None:
    counts = binary_confusion(
        np.array([1, 0, 1, 0]),
        np.array([1, 1, 0, 0]),
    )

    assert counts.true_positive == 1
    assert counts.false_positive == 1
    assert counts.true_negative == 1
    assert counts.false_negative == 1


def test_binary_confusion_exposes_precision_and_recall() -> None:
    counts = binary_confusion(
        np.array([1, 1, 0, 0]),
        np.array([1, 0, 1, 0]),
    )

    assert counts.precision == pytest.approx(0.5)
    assert counts.recall == pytest.approx(0.5)


def test_binary_confusion_exposes_roc_rates() -> None:
    counts = binary_confusion(
        np.array([1, 1, 0, 0]),
        np.array([1, 0, 1, 0]),
    )

    assert counts.true_positive_rate == pytest.approx(0.5)
    assert counts.false_positive_rate == pytest.approx(0.5)


def test_precision_recall_curve_sweeps_thresholds() -> None:
    curve = precision_recall_curve(
        np.array([0.9, 0.1, 0.8]),
        np.array([1, 0, 1]),
    )

    assert curve[0].recall == pytest.approx(0.0)
    assert curve[-1].precision == pytest.approx(2.0 / 3.0)
    assert curve[-1].recall == pytest.approx(1.0)


def test_roc_curve_sweeps_thresholds() -> None:
    curve = roc_curve(
        np.array([0.9, 0.1, 0.8, 0.2]),
        np.array([1, 0, 1, 0]),
    )

    assert curve[0].true_positive_rate == pytest.approx(0.0)
    assert curve[0].false_positive_rate == pytest.approx(0.0)
    assert curve[-1].true_positive_rate == pytest.approx(1.0)
    assert curve[-1].false_positive_rate == pytest.approx(1.0)


def test_trapezoidal_area_sorts_x_coordinates() -> None:
    area = trapezoidal_area(
        np.array([1.0, 0.0, 0.5]),
        np.array([1.0, 0.0, 0.5]),
    )

    assert area == pytest.approx(0.5)


def test_pointwise_auc_scores_are_perfect_for_separable_scores() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])

    assert average_precision_score(scores, labels) == pytest.approx(1.0)
    assert roc_auc_score(scores, labels) == pytest.approx(1.0)


def test_range_curve_uses_range_recall_and_pointwise_false_positive_rate() -> None:
    curve = range_curve(
        np.array([0.1, 0.9, 0.8, 0.2]),
        np.array([0, 1, 1, 0]),
    )

    assert curve[0].recall == pytest.approx(0.0)
    assert curve[-1].recall == pytest.approx(1.0)
    assert curve[-1].false_positive_rate == pytest.approx(1.0)


def test_vus_scores_are_perfect_for_separable_range_scores() -> None:
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    labels = np.array([0, 1, 1, 0])

    assert vus_pr_score(scores, labels) == pytest.approx(1.0)
    assert vus_roc_score(scores, labels) == pytest.approx(1.0)


def test_threshold_free_metrics_groups_pointwise_and_range_scores() -> None:
    metrics = threshold_free_metrics(
        np.array([0.1, 0.9, 0.8, 0.2]),
        np.array([0, 1, 1, 0]),
    )

    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.vus_pr == pytest.approx(1.0)
    assert metrics.vus_roc == pytest.approx(1.0)


def test_threshold_free_metrics_handles_no_anomalies() -> None:
    metrics = threshold_free_metrics(
        np.array([0.1, 0.2, 0.3]),
        np.array([0, 0, 0]),
    )

    assert 0.0 <= metrics.average_precision <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.vus_pr <= 1.0
    assert 0.0 <= metrics.vus_roc <= 1.0


def test_threshold_free_metrics_handles_all_anomalies() -> None:
    metrics = threshold_free_metrics(
        np.array([0.1, 0.2, 0.3]),
        np.array([1, 1, 1]),
    )

    assert 0.0 <= metrics.average_precision <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.vus_pr <= 1.0
    assert 0.0 <= metrics.vus_roc <= 1.0


def test_threshold_free_metrics_handles_tied_scores() -> None:
    metrics = threshold_free_metrics(
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([1, 0, 1, 0]),
    )

    assert metrics.average_precision == pytest.approx(0.5)
    assert metrics.roc_auc == pytest.approx(0.5)
    assert metrics.vus_pr == pytest.approx(0.5)
    assert metrics.vus_roc == pytest.approx(0.5)


def test_select_threshold_uses_range_f1_objective() -> None:
    scores = np.array([0.1, 0.8, 0.7, 0.2, 0.3])
    labels = np.array([0, 1, 1, 0, 0])

    threshold, metrics = select_threshold_by_range_f1(scores, labels)

    assert threshold == pytest.approx(0.7)
    assert metrics.f1 == pytest.approx(1.0)


def test_detection_delay_reports_first_hit_inside_each_range() -> None:
    true = np.array([0, 1, 1, 1, 0, 1, 1])
    pred = np.array([0, 0, 0, 1, 0, 0, 0])

    delay = detection_delay(true, pred)

    assert delay.detected == 1
    assert delay.missed == 1
    assert delay.mean == pytest.approx(2.0)
    assert delay.median == pytest.approx(2.0)


def test_detection_delay_is_nan_when_all_ranges_are_missed() -> None:
    delay = detection_delay(np.array([0, 1, 1]), np.array([0, 0, 0]))

    assert delay.detected == 0
    assert delay.missed == 1
    assert math.isnan(delay.mean)
    assert math.isnan(delay.median)


def test_false_alarm_rate_counts_predicted_range_starts_in_stable_regions() -> None:
    true = np.array([0, 0, 1, 1, 0, 0])
    pred = np.array([1, 1, 0, 0, 1, 0])

    assert false_alarm_rate_per_1000(true, pred) == pytest.approx(500.0)


def test_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        binary_ranges(np.array([0, 2, 1]))
