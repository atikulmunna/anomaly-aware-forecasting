import json

import numpy as np
import pytest

from aaf.eval.cli import main
from aaf.eval.forecasting import MixtureForecast
from aaf.eval.report import (
    evaluate_anomaly,
    evaluate_forecast,
    evaluate_regime,
    evaluate_run_directory,
)


def test_evaluate_forecast_returns_expected_point_metrics() -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.array([[1.0]]),
        means=np.array([[[1.0]]]),
        stds=np.array([[[1.0]]]),
    )

    report = evaluate_forecast(np.array([[3.0]]), forecast, energy_samples=32, seed=0)

    assert report.mae == pytest.approx(2.0)
    assert report.rmse == pytest.approx(2.0)
    assert report.nll > 0.0
    assert report.channelwise_crps > 0.0
    assert report.energy_score > 0.0
    assert report.diagnostics.active_components_1pct == 1


def test_evaluate_anomaly_freezes_validation_threshold_for_test() -> None:
    validation_scores = np.array([0.1, 0.8, 0.7, 0.2])
    validation_labels = np.array([0, 1, 1, 0])
    test_scores = np.array([0.9, 0.1, 0.7, 0.3])
    test_labels = np.array([1, 0, 1, 0])

    report = evaluate_anomaly(
        validation_scores,
        validation_labels,
        test_scores,
        test_labels,
    )

    assert report.threshold == pytest.approx(0.7)
    assert report.threshold_strategy == "max_validation_f1"
    assert report.validation.f1 == pytest.approx(1.0)
    assert report.test.f1 == pytest.approx(1.0)
    assert report.threshold_free.vus_pr == pytest.approx(1.0)
    assert report.threshold_free.vus_roc == pytest.approx(1.0)


def test_evaluate_anomaly_supports_quantile_threshold_strategy() -> None:
    report = evaluate_anomaly(
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0, 0, 0, 0]),
        np.array([0.1, 0.35, 0.45]),
        np.array([0, 1, 1]),
        threshold_strategy="validation_quantile_95",
    )

    assert report.threshold_strategy == "validation_quantile_95"
    assert report.threshold == pytest.approx(0.385)
    assert report.test.recall == pytest.approx(0.5)


def test_evaluate_anomaly_applies_persistence_filter_after_threshold_selection() -> None:
    report = evaluate_anomaly(
        np.array([0.1, 0.9, 0.1, 0.9, 0.9]),
        np.array([0, 0, 0, 0, 0]),
        np.array([0.1, 0.9, 0.1, 0.9, 0.9]),
        np.array([0, 0, 0, 1, 1]),
        threshold_strategy="validation_quantile_95",
        persistence_window=3,
        persistence_count=2,
    )

    assert report.persistence_window == 3
    assert report.persistence_count == 2
    assert report.test.precision == pytest.approx(1.0)
    assert report.test.recall == pytest.approx(1.0)


def test_evaluate_anomaly_supports_per_machine_quantile_thresholds() -> None:
    report = evaluate_anomaly(
        np.array([0.0, 1.0, 10.0, 20.0]),
        np.array([0, 0, 0, 0]),
        np.array([1.0, 0.4, 20.0, 14.0]),
        np.array([1, 0, 1, 0]),
        threshold_strategy="per_machine_validation_quantile_98",
        validation_groups=np.array([0, 0, 1, 1]),
        test_groups=np.array([0, 0, 1, 1]),
    )

    assert report.threshold_strategy == "per_machine_validation_quantile_98"
    assert report.threshold == pytest.approx({0: 0.98, 1: 19.8})
    assert report.test.f1 == pytest.approx(1.0)


def test_evaluate_anomaly_requires_groups_for_per_machine_strategy() -> None:
    with pytest.raises(ValueError, match="group labels"):
        evaluate_anomaly(
            np.array([0.0, 1.0]),
            np.array([0, 0]),
            np.array([0.5, 1.5]),
            np.array([0, 1]),
            threshold_strategy="per_machine_validation_quantile_99",
        )


def test_evaluate_regime_aligns_permuted_labels() -> None:
    true = np.array([0, 0, 1, 1])
    pred = np.array([1, 1, 0, 0])

    report = evaluate_regime(true, pred)

    assert report.adjusted_rand_index == pytest.approx(1.0)
    assert report.confusion == [[2, 0], [0, 2]]
    assert report.label_mapping == {1: 0, 0: 1}


def test_evaluate_run_directory_writes_metrics_json(tmp_path) -> None:
    np.savez(
        tmp_path / "forecast.npz",
        observed=np.array([[0.0]]),
        weights=np.array([[1.0]]),
        means=np.array([[[0.0]]]),
        stds=np.array([[[1.0]]]),
    )
    np.savez(
        tmp_path / "anomaly_validation.npz",
        scores=np.array([0.1, 0.8, 0.7, 0.2]),
        labels=np.array([0, 1, 1, 0]),
        groups=np.array([0, 0, 1, 1]),
    )
    np.savez(
        tmp_path / "anomaly_test.npz",
        scores=np.array([0.9, 0.1, 0.7, 0.3]),
        labels=np.array([1, 0, 1, 0]),
        groups=np.array([0, 0, 1, 1]),
    )
    np.savez(
        tmp_path / "regime.npz",
        true_labels=np.array([0, 0, 1, 1]),
        pred_labels=np.array([1, 1, 0, 0]),
    )
    output = tmp_path / "metrics.json"

    report = evaluate_run_directory(
        tmp_path,
        output_path=output,
        energy_samples=16,
        seed=0,
        threshold_strategy="validation_quantile_95",
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert report.forecast is not None
    assert report.anomaly is not None
    assert report.regime is not None
    assert saved["forecast"]["nll"] == pytest.approx(report.forecast.nll)
    assert saved["forecast"]["diagnostics"]["active_components_1pct"] == 1
    assert saved["anomaly"]["threshold_strategy"] == "validation_quantile_95"
    assert saved["anomaly"]["persistence_window"] == 1
    assert saved["anomaly"]["persistence_count"] == 1
    assert saved["anomaly"]["threshold_free"]["vus_pr"] == pytest.approx(1.0)
    assert saved["anomaly"]["threshold_free"]["vus_roc"] == pytest.approx(1.0)
    assert saved["regime"]["adjusted_rand_index"] == pytest.approx(1.0)


def test_evaluate_run_directory_uses_artifact_groups_for_per_machine_thresholds(tmp_path) -> None:
    np.savez(
        tmp_path / "anomaly_validation.npz",
        scores=np.array([0.0, 1.0, 10.0, 20.0]),
        labels=np.array([0, 0, 0, 0]),
        groups=np.array([0, 0, 1, 1]),
    )
    np.savez(
        tmp_path / "anomaly_test.npz",
        scores=np.array([1.0, 0.4, 20.0, 14.0]),
        labels=np.array([1, 0, 1, 0]),
        groups=np.array([0, 0, 1, 1]),
    )

    report = evaluate_run_directory(
        tmp_path,
        threshold_strategy="per_machine_validation_quantile_98",
    )

    assert report.anomaly is not None
    assert report.anomaly.test.f1 == pytest.approx(1.0)


def test_cli_writes_default_metrics_path(tmp_path) -> None:
    np.savez(
        tmp_path / "forecast.npz",
        observed=np.array([[0.0]]),
        weights=np.array([[1.0]]),
        means=np.array([[[0.0]]]),
        stds=np.array([[[1.0]]]),
    )

    np.savez(
        tmp_path / "anomaly_validation.npz",
        scores=np.array([0.1, 0.2, 0.3]),
        labels=np.array([0, 0, 0]),
    )
    np.savez(
        tmp_path / "anomaly_test.npz",
        scores=np.array([0.1, 0.35, 0.45]),
        labels=np.array([0, 1, 1]),
    )

    exit_code = main(
        [
            str(tmp_path),
            "--energy-samples",
            "16",
            "--threshold-strategy",
            "validation_quantile_95",
            "--persistence-window",
            "2",
            "--persistence-count",
            "1",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "metrics.json").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["anomaly"]["threshold_strategy"] == "validation_quantile_95"
    assert saved["anomaly"]["persistence_window"] == 2
    assert saved["anomaly"]["persistence_count"] == 1
