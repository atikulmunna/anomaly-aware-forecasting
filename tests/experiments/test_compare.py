import json

from aaf.experiments import RunManifest, write_run_manifest
from aaf.experiments.compare import collect_run_row, flatten_mapping


def test_flatten_mapping_uses_dotted_metric_keys() -> None:
    flattened = flatten_mapping(
        {
            "forecast": {"nll": 1.5},
            "anomaly": {"threshold_free": {"vus_pr": 0.8}},
        }
    )

    assert flattened == {
        "forecast.nll": 1.5,
        "anomaly.threshold_free.vus_pr": 0.8,
    }


def test_collect_run_row_flattens_metrics_and_manifest(tmp_path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"forecast": {"nll": 2.5}, "anomaly": {"test": {"f1": 0.7}}}),
        encoding="utf-8",
    )
    write_run_manifest(
        run_dir / "manifest.json",
        RunManifest(run_id="run-a", pipeline="joint", dataset="synthetic", seed=3),
    )

    row = collect_run_row(run_dir)

    assert row.values["manifest.run_id"] == "run-a"
    assert row.values["manifest.pipeline"] == "joint"
    assert row.values["forecast.nll"] == 2.5
    assert row.values["anomaly.test.f1"] == 0.7
