from pathlib import Path

import pytest

from aaf.pipelines.smd_joint import SMDJointConfig


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


def tiny_config(root: Path) -> SMDJointConfig:
    return SMDJointConfig(
        root=root,
        validation_fraction=0.25,
        lookback=3,
        horizon=1,
        stride=1,
        n_regimes=2,
        hidden_size=6,
        n_components=2,
        epochs=1,
        batch_size=4,
        learning_rate=0.01,
        energy_samples=16,
    )


def test_smd_joint_config_accepts_valid_values(tmp_path) -> None:
    tiny_config(tmp_path).validate()


def test_smd_joint_config_rejects_invalid_regime_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="n_regimes"):
        SMDJointConfig(root=tmp_path, n_regimes=1).validate()
