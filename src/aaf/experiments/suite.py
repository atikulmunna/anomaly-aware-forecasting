"""Experiment suite execution for reproducible run matrices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from aaf.eval.report import EvaluationReport
from aaf.experiments.compare import collect_run_rows, write_comparison_csv, write_comparison_json
from aaf.experiments.manifest import RunManifest, write_run_manifest
from aaf.pipelines.joint_synthetic import JointSyntheticConfig, run_joint_synthetic
from aaf.pipelines.mdn_synthetic import MDNSyntheticConfig, run_mdn_synthetic
from aaf.pipelines.smd_baseline import SMDBaselineConfig, run_smd_baseline
from aaf.pipelines.smd_joint import SMDJointConfig, run_smd_joint
from aaf.pipelines.synthetic_baseline import SyntheticBaselineConfig, run_synthetic_baseline


class PipelineRunFunction(Protocol):
    def __call__(
        self,
        output_dir: Path,
        config: Any,
        *,
        overwrite: bool = False,
    ) -> EvaluationReport: ...


PipelineRunner = tuple[type[Any], PipelineRunFunction]

PIPELINES: dict[str, PipelineRunner] = {
    "synthetic-baseline": (SyntheticBaselineConfig, run_synthetic_baseline),
    "synthetic-mdn": (MDNSyntheticConfig, run_mdn_synthetic),
    "synthetic-joint": (JointSyntheticConfig, run_joint_synthetic),
    "smd-baseline": (SMDBaselineConfig, run_smd_baseline),
    "smd-joint": (SMDJointConfig, run_smd_joint),
}


@dataclass(frozen=True)
class SuiteJob:
    """One pipeline invocation in an experiment suite."""

    run_id: str
    pipeline: str
    dataset: str
    params: dict[str, Any]
    notes: str | None = None

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.pipeline:
            raise ValueError("pipeline must be non-empty")
        if not self.dataset:
            raise ValueError("dataset must be non-empty")


@dataclass(frozen=True)
class ExperimentSuite:
    """Named collection of experiment jobs."""

    name: str
    jobs: tuple[SuiteJob, ...]

    def validate(self) -> None:
        if not self.name:
            raise ValueError("suite name must be non-empty")
        if len(self.jobs) == 0:
            raise ValueError("suite must contain at least one job")
        seen_run_ids: set[str] = set()
        for job in self.jobs:
            job.validate()
            if job.run_id in seen_run_ids:
                raise ValueError(f"duplicate run_id in suite: {job.run_id}")
            seen_run_ids.add(job.run_id)


@dataclass(frozen=True)
class SuiteRunResult:
    """Run directories and reports produced by a suite execution."""

    output_root: Path
    reports: dict[str, EvaluationReport]


def load_experiment_suite(path: Path) -> ExperimentSuite:
    """Load an experiment suite JSON file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment suite must be a JSON object")
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError("experiment suite jobs must be a list")
    jobs = tuple(_suite_job_from_payload(item) for item in raw_jobs)
    suite = ExperimentSuite(name=str(payload["name"]), jobs=jobs)
    suite.validate()
    return suite


def validate_suite_job_config(job: SuiteJob) -> None:
    """Validate one suite job against its target pipeline config type."""

    job.validate()
    if job.pipeline not in PIPELINES:
        raise ValueError(f"unsupported pipeline: {job.pipeline}")
    config_type, _runner = PIPELINES[job.pipeline]
    config = config_type(**_normalize_config_params(job.params))
    config.validate()


def validate_suite_configs(suite: ExperimentSuite) -> None:
    """Validate every job's params against the configured pipeline."""

    suite.validate()
    for job in suite.jobs:
        validate_suite_job_config(job)


def run_experiment_suite(
    suite: ExperimentSuite,
    output_root: Path,
    *,
    compare_output: Path | None = None,
    overwrite: bool = False,
) -> SuiteRunResult:
    """Run every job in a suite and write manifests beside run artifacts."""

    suite.validate()
    reports: dict[str, EvaluationReport] = {}
    for job in suite.jobs:
        report = run_suite_job(job, output_root, overwrite=overwrite)
        reports[job.run_id] = report
        write_run_manifest(
            output_root / job.run_id / "manifest.json",
            RunManifest(
                run_id=job.run_id,
                pipeline=job.pipeline,
                dataset=job.dataset,
                seed=_seed_from_params(job.params),
                notes=job.notes,
            ),
        )
    if compare_output is not None:
        _write_comparison(compare_output, output_root)
    return SuiteRunResult(output_root=output_root, reports=reports)


def run_suite_job(
    job: SuiteJob,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> EvaluationReport:
    """Run one suite job and return its evaluation report."""

    job.validate()
    if job.pipeline not in PIPELINES:
        raise ValueError(f"unsupported pipeline: {job.pipeline}")
    config_type, runner = PIPELINES[job.pipeline]
    config = config_type(**_normalize_config_params(job.params))
    return runner(output_root / job.run_id, config, overwrite=overwrite)


def _suite_job_from_payload(payload: object) -> SuiteJob:
    if not isinstance(payload, dict):
        raise ValueError("suite job must be a JSON object")
    raw_params = payload.get("params", {})
    if not isinstance(raw_params, dict):
        raise ValueError("suite job params must be a JSON object")
    return SuiteJob(
        run_id=str(payload["run_id"]),
        pipeline=str(payload["pipeline"]),
        dataset=str(payload.get("dataset", payload["pipeline"])),
        params=dict(raw_params),
        notes=None if payload.get("notes") is None else str(payload["notes"]),
    )


def _normalize_config_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    if "root" in normalized:
        normalized["root"] = Path(str(normalized["root"]))
    return normalized


def _seed_from_params(params: dict[str, Any]) -> int | None:
    if "seed" not in params or params["seed"] is None:
        return None
    return int(params["seed"])


def _write_comparison(path: Path, output_root: Path) -> None:
    rows = collect_run_rows(output_root)
    if path.suffix.lower() == ".json":
        write_comparison_json(path, rows)
    elif path.suffix.lower() == ".csv":
        write_comparison_csv(path, rows)
    else:
        raise ValueError("suite comparison output must end with .csv or .json")
