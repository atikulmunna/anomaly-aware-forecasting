"""Evaluation report assembly for archived run artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aaf.eval.anomaly import (
    DetectionDelay,
    RangeMetrics,
    detection_delay,
    false_alarm_rate_per_1000,
    range_precision_recall,
    select_threshold_by_range_f1,
    threshold_scores,
)
from aaf.eval.diagnostics import MixtureDiagnostics, mixture_diagnostics
from aaf.eval.forecasting import (
    MixtureForecast,
    central_interval_coverage,
    channelwise_crps,
    energy_score,
    mean_absolute_error,
    negative_log_likelihood,
    root_mean_squared_error,
)
from aaf.eval.regime import (
    adjusted_rand_index,
    align_regime_labels,
    false_switch_rate_per_1000,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ForecastReport:
    nll: float
    channelwise_crps: float
    energy_score: float
    mae: float
    rmse: float
    interval_coverage_90: float
    diagnostics: MixtureDiagnostics


@dataclass(frozen=True)
class AnomalyReport:
    threshold: float
    validation: RangeMetrics
    test: RangeMetrics
    detection_delay: DetectionDelay
    false_alarm_rate_per_1000: float


@dataclass(frozen=True)
class RegimeReport:
    adjusted_rand_index: float
    false_switch_rate_per_1000: float
    confusion: list[list[int]]
    label_mapping: dict[int, int]


@dataclass(frozen=True)
class EvaluationReport:
    forecast: ForecastReport | None = None
    anomaly: AnomalyReport | None = None
    regime: RegimeReport | None = None

    def to_dict(self) -> dict[str, Any]:
        value = _json_ready(asdict(self))
        if not isinstance(value, dict):
            raise TypeError("evaluation report serialization did not produce a dictionary")
        return value

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def evaluate_forecast(
    observed: FloatArray,
    forecast: MixtureForecast,
    *,
    energy_samples: int = 256,
    seed: int = 0,
) -> ForecastReport:
    """Evaluate probabilistic forecasts against observations."""

    return ForecastReport(
        nll=negative_log_likelihood(observed, forecast),
        channelwise_crps=channelwise_crps(observed, forecast),
        energy_score=energy_score(observed, forecast, n_samples=energy_samples, seed=seed),
        mae=mean_absolute_error(observed, forecast),
        rmse=root_mean_squared_error(observed, forecast),
        interval_coverage_90=central_interval_coverage(observed, forecast, level=0.9),
        diagnostics=mixture_diagnostics(forecast),
    )


def evaluate_anomaly(
    validation_scores: FloatArray,
    validation_labels: IntArray,
    test_scores: FloatArray,
    test_labels: IntArray,
) -> AnomalyReport:
    """Select an anomaly threshold on validation data and evaluate on test data."""

    threshold, validation_metrics = select_threshold_by_range_f1(
        validation_scores,
        validation_labels,
    )
    test_predictions = threshold_scores(test_scores, threshold)
    test_metrics = range_precision_recall(test_labels, test_predictions)
    return AnomalyReport(
        threshold=threshold,
        validation=validation_metrics,
        test=test_metrics,
        detection_delay=detection_delay(test_labels, test_predictions),
        false_alarm_rate_per_1000=false_alarm_rate_per_1000(test_labels, test_predictions),
    )


def evaluate_regime(true_labels: IntArray, pred_labels: IntArray) -> RegimeReport:
    """Evaluate regime assignments after permutation alignment."""

    alignment = align_regime_labels(true_labels, pred_labels)
    return RegimeReport(
        adjusted_rand_index=adjusted_rand_index(true_labels, pred_labels),
        false_switch_rate_per_1000=false_switch_rate_per_1000(
            true_labels,
            alignment.aligned_pred,
        ),
        confusion=alignment.confusion.tolist(),
        label_mapping=alignment.mapping,
    )


def evaluate_run_directory(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    energy_samples: int = 256,
    seed: int = 0,
) -> EvaluationReport:
    """Evaluate any supported artifacts found in a run directory.

    Supported files:
    - forecast.npz: observed, weights, means, stds
    - anomaly_validation.npz: scores, labels
    - anomaly_test.npz: scores, labels
    - regime.npz: true_labels, pred_labels
    """

    forecast_report = None
    forecast_path = run_dir / "forecast.npz"
    if forecast_path.exists():
        with np.load(forecast_path) as data:
            forecast_report = evaluate_forecast(
                np.asarray(data["observed"]),
                MixtureForecast.from_arrays(data["weights"], data["means"], data["stds"]),
                energy_samples=energy_samples,
                seed=seed,
            )

    anomaly_report = None
    anomaly_validation_path = run_dir / "anomaly_validation.npz"
    anomaly_test_path = run_dir / "anomaly_test.npz"
    if anomaly_validation_path.exists() and anomaly_test_path.exists():
        with np.load(anomaly_validation_path) as validation_data:
            validation_scores = np.asarray(validation_data["scores"])
            validation_labels = np.asarray(validation_data["labels"])
        with np.load(anomaly_test_path) as test_data:
            test_scores = np.asarray(test_data["scores"])
            test_labels = np.asarray(test_data["labels"])
        anomaly_report = evaluate_anomaly(
            validation_scores,
            validation_labels,
            test_scores,
            test_labels,
        )

    regime_report = None
    regime_path = run_dir / "regime.npz"
    if regime_path.exists():
        with np.load(regime_path) as data:
            regime_report = evaluate_regime(
                np.asarray(data["true_labels"]),
                np.asarray(data["pred_labels"]),
            )

    report = EvaluationReport(
        forecast=forecast_report,
        anomaly=anomaly_report,
        regime=regime_report,
    )
    if output_path is not None:
        report.write_json(output_path)
    return report


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
