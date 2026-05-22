import json

from aaf.experiments.suite_cli import main


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


def test_run_suite_cli_writes_runs_and_comparison(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    write_smd_fixture(dataset_root)
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "smoke",
                "jobs": [
                    {
                        "run_id": "smd-baseline",
                        "pipeline": "smd-baseline",
                        "dataset": "smd",
                        "params": {
                            "root": str(dataset_root),
                            "lookback": 3,
                            "validation_fraction": 0.25,
                            "energy_samples": 16,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            str(suite_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--compare",
            str(tmp_path / "reports" / "comparison.json"),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "runs" / "smd-baseline" / "metrics.json").exists()
    assert (tmp_path / "reports" / "comparison.json").exists()


def test_run_suite_cli_dry_run_validates_without_outputs(tmp_path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "dry",
                "jobs": [
                    {
                        "run_id": "synthetic-baseline",
                        "pipeline": "synthetic-baseline",
                        "dataset": "synthetic",
                        "params": {
                            "n_train_configs": 1,
                            "n_validation_configs": 1,
                            "n_test_configs": 1,
                            "series_length": 80,
                            "lookback": 8,
                            "energy_samples": 16,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            str(suite_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "runs").exists()


def test_run_suite_cli_applies_set_overrides(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    write_smd_fixture(dataset_root)
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "override",
                "jobs": [
                    {
                        "run_id": "smd-baseline",
                        "pipeline": "smd-baseline",
                        "dataset": "smd",
                        "params": {
                            "root": "placeholder",
                            "lookback": 3,
                            "validation_fraction": 0.25,
                            "energy_samples": 16,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            str(suite_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--set",
            f"root={dataset_root}",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "runs" / "smd-baseline" / "metrics.json").exists()
