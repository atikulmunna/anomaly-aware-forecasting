import json
from pathlib import Path

from aaf.pipelines.smd_baseline import SMDBaselineConfig, run_smd_baseline
from aaf.pipelines.smd_joint import SMDJointConfig, run_smd_joint


def write_smd_fixture(root: Path, machine_id: str = "machine-1-1") -> None:
    for directory in ("train", "test", "test_label"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "train" / f"{machine_id}.txt").write_text(
        "\n".join(f"{idx},{idx + 1}" for idx in range(16)),
        encoding="utf-8",
    )
    (root / "test" / f"{machine_id}.txt").write_text(
        "\n".join(f"{idx},{idx + 1}" for idx in range(10)),
        encoding="utf-8",
    )
    (root / "test_label" / f"{machine_id}.txt").write_text(
        "0\n0\n1\n0\n0\n0\n0\n0\n0\n0\n",
        encoding="utf-8",
    )


def test_smd_pipelines_emit_metrics_contract(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    write_smd_fixture(dataset_root)

    run_smd_baseline(
        tmp_path / "baseline",
        SMDBaselineConfig(
            root=dataset_root,
            lookback=3,
            validation_fraction=0.25,
            energy_samples=16,
        ),
    )
    run_smd_joint(
        tmp_path / "joint",
        SMDJointConfig(
            root=dataset_root,
            lookback=3,
            validation_fraction=0.25,
            n_regimes=2,
            hidden_size=6,
            n_components=2,
            epochs=1,
            batch_size=4,
            learning_rate=0.01,
            energy_samples=16,
        ),
    )

    baseline_metrics = json.loads((tmp_path / "baseline" / "metrics.json").read_text())
    joint_metrics = json.loads((tmp_path / "joint" / "metrics.json").read_text())
    assert set(baseline_metrics) == {"anomaly", "forecast", "regime"}
    assert set(joint_metrics) == {"anomaly", "forecast", "regime"}
