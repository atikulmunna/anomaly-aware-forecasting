"""Anomaly-only rescoring for archived run artifacts."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aaf.eval.anomaly import threshold_free_metrics
from aaf.eval.report import EvaluationReport, evaluate_anomaly
from aaf.experiments.compare import collect_run_rows, write_comparison_csv, write_comparison_json
from aaf.experiments.manifest import RunManifest, load_run_manifest, write_run_manifest


@dataclass(frozen=True)
class AnomalyRescoreJob:
    """One anomaly threshold and persistence setting for an archived run."""

    run_id: str
    threshold_strategy: str
    persistence_window: int = 1
    persistence_count: int = 1
    notes: str | None = None

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.persistence_window <= 0:
            raise ValueError("persistence_window must be positive")
        if self.persistence_count <= 0:
            raise ValueError("persistence_count must be positive")
        if self.persistence_count > self.persistence_window:
            raise ValueError("persistence_count must be <= persistence_window")


def rescore_anomaly_run(
    source_run_dir: Path,
    output_root: Path,
    jobs: tuple[AnomalyRescoreJob, ...],
    *,
    pipeline: str | None = None,
    dataset: str | None = None,
    seed: int | None = None,
    compare_output: Path | None = None,
    overwrite: bool = False,
) -> dict[str, EvaluationReport]:
    """Evaluate anomaly artifacts from one archived run under multiple settings."""

    if not jobs:
        raise ValueError("at least one rescore job is required")
    if not source_run_dir.exists():
        raise FileNotFoundError(source_run_dir)
    _require_anomaly_artifacts(source_run_dir)
    source_manifest = _load_optional_manifest(source_run_dir)
    resolved_pipeline = pipeline or (
        source_manifest.pipeline if source_manifest else "anomaly-rescore"
    )
    resolved_dataset = dataset or (source_manifest.dataset if source_manifest else "unknown")
    resolved_seed = seed if seed is not None else (
        source_manifest.seed if source_manifest else None
    )
    artifacts = _load_anomaly_artifacts(source_run_dir)
    cached_threshold_free = threshold_free_metrics(
        artifacts.test_scores,
        artifacts.test_labels,
    )

    reports: dict[str, EvaluationReport] = {}
    for job in jobs:
        job.validate()
        output_dir = output_root / job.run_id
        _prepare_output_dir(output_dir, overwrite=overwrite)
        report = EvaluationReport(
            anomaly=evaluate_anomaly(
                artifacts.validation_scores,
                artifacts.validation_labels,
                artifacts.test_scores,
                artifacts.test_labels,
                threshold_strategy=job.threshold_strategy,
                persistence_window=job.persistence_window,
                persistence_count=job.persistence_count,
                validation_groups=artifacts.validation_groups,
                test_groups=artifacts.test_groups,
                threshold_free=cached_threshold_free,
            ),
        )
        report.write_json(output_dir / "metrics.json")
        write_run_manifest(
            output_dir / "manifest.json",
            RunManifest(
                run_id=job.run_id,
                pipeline=resolved_pipeline,
                dataset=resolved_dataset,
                seed=resolved_seed,
                notes=job.notes,
            ),
        )
        reports[job.run_id] = report

    if compare_output is not None:
        _write_comparison(compare_output, output_root)
    return reports


@dataclass(frozen=True)
class _AnomalyArtifacts:
    validation_scores: NDArray[np.float64]
    validation_labels: NDArray[np.int64]
    test_scores: NDArray[np.float64]
    test_labels: NDArray[np.int64]
    validation_groups: NDArray[np.int64] | None
    test_groups: NDArray[np.int64] | None


def make_rescore_run_id(
    source_run_dir: Path,
    threshold_strategy: str,
    persistence_window: int,
    persistence_count: int,
    *,
    prefix: str | None = None,
) -> str:
    """Create a stable run id for one rescoring setting."""

    base = prefix or source_run_dir.name
    return (
        f"{_slug(base)}-{_slug(threshold_strategy)}"
        f"-w{persistence_window}-c{persistence_count}"
    )


def _load_optional_manifest(run_dir: Path) -> RunManifest | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return load_run_manifest(manifest_path)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _require_anomaly_artifacts(source_run_dir: Path) -> None:
    missing = [
        name
        for name in ("anomaly_validation.npz", "anomaly_test.npz")
        if not (source_run_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing anomaly artifacts: {', '.join(missing)}")


def _load_anomaly_artifacts(source_run_dir: Path) -> _AnomalyArtifacts:
    with np.load(source_run_dir / "anomaly_validation.npz") as validation_data:
        validation_scores = np.asarray(validation_data["scores"])
        validation_labels = np.asarray(validation_data["labels"])
        validation_groups = (
            np.asarray(validation_data["groups"]) if "groups" in validation_data else None
        )
    with np.load(source_run_dir / "anomaly_test.npz") as test_data:
        test_scores = np.asarray(test_data["scores"])
        test_labels = np.asarray(test_data["labels"])
        test_groups = np.asarray(test_data["groups"]) if "groups" in test_data else None
    return _AnomalyArtifacts(
        validation_scores=validation_scores,
        validation_labels=validation_labels,
        test_scores=test_scores,
        test_labels=test_labels,
        validation_groups=validation_groups,
        test_groups=test_groups,
    )


def _write_comparison(path: Path, output_root: Path) -> None:
    rows = collect_run_rows(output_root)
    if path.suffix.lower() == ".json":
        write_comparison_json(path, rows)
    elif path.suffix.lower() == ".csv":
        write_comparison_csv(path, rows)
    else:
        raise ValueError("rescore comparison output must end with .csv or .json")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "run"
