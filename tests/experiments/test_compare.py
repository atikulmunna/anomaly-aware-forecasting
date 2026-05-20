import csv
import json

from aaf.experiments import RunManifest, write_run_manifest
from aaf.experiments.compare import (
    collect_run_row,
    collect_run_rows,
    comparison_columns,
    comparison_table,
    discover_run_dirs,
    flatten_mapping,
    write_comparison_csv,
    write_comparison_json,
)


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


def test_collect_run_rows_discovers_nested_metrics(tmp_path) -> None:
    for name in ("a", "nested/b"):
        run_dir = tmp_path / name
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps({"forecast": {"nll": len(name)}}),
            encoding="utf-8",
        )

    run_dirs = discover_run_dirs(tmp_path)
    rows = collect_run_rows(tmp_path)

    assert [path.name for path in run_dirs] == ["a", "b"]
    assert [row.values["forecast.nll"] for row in rows] == [1, 8]


def test_comparison_table_pads_discovered_columns(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "metrics.json").write_text(json.dumps({"forecast": {"nll": 1.0}}), encoding="utf-8")
    (second / "metrics.json").write_text(
        json.dumps({"anomaly": {"test": {"f1": 0.4}}}),
        encoding="utf-8",
    )
    rows = collect_run_rows(tmp_path)

    columns = comparison_columns(rows)
    table = comparison_table(rows)

    assert "forecast.nll" in columns
    assert "anomaly.test.f1" in columns
    assert table[0]["anomaly.test.f1"] is None
    assert table[1]["forecast.nll"] is None


def test_comparison_exporters_write_json_and_csv(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps({"forecast": {"nll": 1.2}}), encoding="utf-8")
    rows = collect_run_rows(tmp_path)

    write_comparison_json(tmp_path / "reports" / "comparison.json", rows)
    write_comparison_csv(tmp_path / "reports" / "comparison.csv", rows)

    payload = json.loads((tmp_path / "reports" / "comparison.json").read_text(encoding="utf-8"))
    with (tmp_path / "reports" / "comparison.csv").open(encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))

    assert payload[0]["forecast.nll"] == 1.2
    assert csv_rows[0]["forecast.nll"] == "1.2"
