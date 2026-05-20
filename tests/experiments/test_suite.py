import json

import pytest

from aaf.experiments.suite import ExperimentSuite, SuiteJob, load_experiment_suite


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
