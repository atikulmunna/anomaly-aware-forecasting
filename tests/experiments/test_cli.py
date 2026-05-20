import csv
import json

from aaf.experiments.cli import main


def test_compare_runs_cli_writes_csv(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"forecast": {"nll": 1.25}}),
        encoding="utf-8",
    )
    output = tmp_path / "reports" / "comparison.csv"

    exit_code = main([str(tmp_path / "runs"), "--output", str(output)])

    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert exit_code == 0
    assert rows[0]["forecast.nll"] == "1.25"


def test_compare_runs_cli_writes_json(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"anomaly": {"test": {"f1": 0.5}}}),
        encoding="utf-8",
    )
    output = tmp_path / "reports" / "comparison.json"

    exit_code = main([str(tmp_path / "runs"), "--output", str(output)])

    rows = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert rows[0]["anomaly.test.f1"] == 0.5
