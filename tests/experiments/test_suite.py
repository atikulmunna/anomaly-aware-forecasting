import json
from pathlib import Path

import pytest

from aaf.experiments import load_run_manifest
from aaf.experiments.suite import (
    ExperimentSuite,
    SuiteJob,
    load_experiment_suite,
    run_experiment_suite,
    run_suite_job,
    validate_suite_configs,
)


def write_smd_fixture(root) -> None:
    for directory in ("train", "test", "test_label"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "train" / "machine-1-1.txt").write_text(
        "\n".join(f"{idx},{idx + 1}" for idx in range(16)),
        encoding="utf-8",
    )
    (root / "test" / "machine-1-1.txt").write_text(
        "\n".join(f"{idx},{idx + 1}" for idx in range(10)),
        encoding="utf-8",
    )
    (root / "test_label" / "machine-1-1.txt").write_text(
        "0\n0\n1\n0\n0\n0\n0\n0\n0\n0\n",
        encoding="utf-8",
    )


def test_experiment_suite_rejects_duplicate_run_ids() -> None:
    suite = ExperimentSuite(
        name="dup",
        jobs=(
            SuiteJob(run_id="same", pipeline="synthetic-baseline", dataset="synthetic", params={}),
            SuiteJob(run_id="same", pipeline="synthetic-joint", dataset="synthetic", params={}),
        ),
    )

    with pytest.raises(ValueError, match="duplicate"):
        suite.validate()


def test_load_experiment_suite_reads_jobs(tmp_path) -> None:
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "name": "smoke",
                "jobs": [
                    {
                        "run_id": "synthetic-baseline",
                        "pipeline": "synthetic-baseline",
                        "dataset": "synthetic",
                        "params": {"seed": 3},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite = load_experiment_suite(path)

    assert suite.name == "smoke"
    assert suite.jobs[0].run_id == "synthetic-baseline"
    assert suite.jobs[0].params == {"seed": 3}


def test_checked_in_smoke_suite_loads() -> None:
    suite = load_experiment_suite(Path("experiments/smoke.synthetic.json"))

    assert suite.name == "synthetic-smoke"
    assert [job.pipeline for job in suite.jobs] == [
        "synthetic-baseline",
        "synthetic-mdn",
        "synthetic-joint",
    ]


def test_checked_in_headline_synthetic_suite_loads() -> None:
    suite = load_experiment_suite(Path("experiments/headline.synthetic.json"))

    assert suite.name == "synthetic-headline"
    assert [job.pipeline for job in suite.jobs] == [
        "synthetic-baseline",
        "synthetic-mdn",
        "synthetic-joint",
    ]
    assert all(job.params["n_test_configs"] == 10 for job in suite.jobs)


def test_checked_in_synthetic_suites_match_pipeline_configs() -> None:
    for path in (
        Path("experiments/smoke.synthetic.json"),
        Path("experiments/headline.synthetic.json"),
    ):
        validate_suite_configs(load_experiment_suite(path))


def test_run_suite_job_dispatches_smd_baseline(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    write_smd_fixture(dataset_root)
    job = SuiteJob(
        run_id="smd-baseline",
        pipeline="smd-baseline",
        dataset="smd",
        params={
            "root": str(dataset_root),
            "lookback": 3,
            "validation_fraction": 0.25,
            "energy_samples": 16,
        },
    )

    report = run_suite_job(job, tmp_path / "runs")

    assert report.anomaly is not None
    assert (tmp_path / "runs" / "smd-baseline" / "metrics.json").exists()


def test_run_suite_job_accepts_machine_id_lists(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    write_smd_fixture(dataset_root)
    job = SuiteJob(
        run_id="smd-baseline",
        pipeline="smd-baseline",
        dataset="smd",
        params={
            "root": str(dataset_root),
            "machine_ids": ["machine-1-1"],
            "lookback": 3,
            "validation_fraction": 0.25,
            "energy_samples": 16,
        },
    )

    report = run_suite_job(job, tmp_path / "runs")

    assert report.anomaly is not None


def test_run_experiment_suite_writes_manifests(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    write_smd_fixture(dataset_root)
    suite = ExperimentSuite(
        name="smoke",
        jobs=(
            SuiteJob(
                run_id="smd-baseline",
                pipeline="smd-baseline",
                dataset="smd",
                params={
                    "root": str(dataset_root),
                    "lookback": 3,
                    "validation_fraction": 0.25,
                    "energy_samples": 16,
                    "seed": 11,
                },
                notes="tiny",
            ),
        ),
    )

    result = run_experiment_suite(suite, tmp_path / "runs")
    manifest = load_run_manifest(tmp_path / "runs" / "smd-baseline" / "manifest.json")

    assert set(result.reports) == {"smd-baseline"}
    assert manifest.run_id == "smd-baseline"
    assert manifest.seed == 11
    assert manifest.notes == "tiny"


def test_run_experiment_suite_writes_comparison_output(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    write_smd_fixture(dataset_root)
    suite = ExperimentSuite(
        name="smoke",
        jobs=(
            SuiteJob(
                run_id="smd-baseline",
                pipeline="smd-baseline",
                dataset="smd",
                params={
                    "root": str(dataset_root),
                    "lookback": 3,
                    "validation_fraction": 0.25,
                    "energy_samples": 16,
                },
            ),
        ),
    )

    run_experiment_suite(
        suite,
        tmp_path / "runs",
        compare_output=tmp_path / "reports" / "comparison.csv",
    )

    assert (tmp_path / "reports" / "comparison.csv").exists()
