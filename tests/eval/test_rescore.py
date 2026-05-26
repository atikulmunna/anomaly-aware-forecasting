import csv
import json

import numpy as np
import pytest

from aaf.eval.rescore import AnomalyRescoreJob, make_rescore_run_id, rescore_anomaly_run
from aaf.eval.rescore_cli import main
from aaf.experiments import RunManifest, write_run_manifest


def test_rescore_anomaly_run_writes_comparison_ready_outputs(tmp_path) -> None:
    source = tmp_path / "source-run"
    source.mkdir()
    np.savez(
        source / "anomaly_validation.npz",
        scores=np.array([0.1, 0.2, 0.3]),
        labels=np.array([0, 0, 0]),
    )
    np.savez(
        source / "anomaly_test.npz",
        scores=np.array([0.1, 0.35, 0.45]),
        labels=np.array([0, 1, 1]),
    )
    write_run_manifest(
        source / "manifest.json",
        RunManifest(run_id="source-run", pipeline="smd-joint", dataset="smd", seed=7),
    )

    reports = rescore_anomaly_run(
        source,
        tmp_path / "rescored",
        (
            AnomalyRescoreJob(
                run_id="q95-w1-c1",
                threshold_strategy="validation_quantile_95",
            ),
        ),
        compare_output=tmp_path / "comparison.csv",
    )

    metrics = json.loads((tmp_path / "rescored" / "q95-w1-c1" / "metrics.json").read_text())
    manifest = json.loads((tmp_path / "rescored" / "q95-w1-c1" / "manifest.json").read_text())
    with (tmp_path / "comparison.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert reports["q95-w1-c1"].forecast is None
    assert reports["q95-w1-c1"].regime is None
    assert metrics["anomaly"]["threshold_strategy"] == "validation_quantile_95"
    assert manifest["pipeline"] == "smd-joint"
    assert manifest["dataset"] == "smd"
    assert rows[0]["manifest.run_id"] == "q95-w1-c1"
    assert rows[0]["forecast"] == ""


def test_rescore_anomaly_run_rejects_missing_anomaly_artifacts(tmp_path) -> None:
    source = tmp_path / "source-run"
    source.mkdir()

    with pytest.raises(FileNotFoundError, match="anomaly_validation"):
        rescore_anomaly_run(
            source,
            tmp_path / "rescored",
            (AnomalyRescoreJob(run_id="q99", threshold_strategy="validation_quantile_99"),),
        )


def test_make_rescore_run_id_sanitizes_strategy_names(tmp_path) -> None:
    run_id = make_rescore_run_id(
        tmp_path / "Run A",
        "per_machine_validation_quantile_98",
        5,
        2,
    )

    assert run_id == "run-a-per-machine-validation-quantile-98-w5-c2"


def test_rescore_cli_writes_strategy_persistence_matrix(tmp_path) -> None:
    source = tmp_path / "source-run"
    source.mkdir()
    np.savez(
        source / "anomaly_validation.npz",
        scores=np.array([0.1, 0.2, 0.3, 0.4]),
        labels=np.array([0, 0, 0, 0]),
    )
    np.savez(
        source / "anomaly_test.npz",
        scores=np.array([0.1, 0.35, 0.45]),
        labels=np.array([0, 1, 1]),
    )

    exit_code = main(
        [
            str(source),
            "--output-root",
            str(tmp_path / "rescored"),
            "--compare",
            str(tmp_path / "comparison.csv"),
            "--strategy",
            "validation_quantile_95",
            "--strategy",
            "validation_quantile_99",
            "--persistence",
            "1:1",
            "--persistence",
            "2:1",
        ]
    )

    assert exit_code == 0
    with (tmp_path / "comparison.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert {
        row["anomaly.threshold_strategy"]
        for row in rows
    } == {"validation_quantile_95", "validation_quantile_99"}
