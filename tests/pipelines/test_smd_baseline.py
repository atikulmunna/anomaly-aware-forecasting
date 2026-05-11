from pathlib import Path

import pytest

from aaf.pipelines.smd_baseline import SMDBaselineConfig, build_smd_baseline_datasets


def write_smd_fixture(root: Path, machine_id: str = "machine-1-1") -> None:
    for directory in ("train", "test", "test_label"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "train" / f"{machine_id}.txt").write_text(
        "\n".join(f"{idx},{idx + 1}" for idx in range(12)),
        encoding="utf-8",
    )
    (root / "test" / f"{machine_id}.txt").write_text(
        "\n".join(f"{idx},{idx + 1}" for idx in range(8)),
        encoding="utf-8",
    )
    (root / "test_label" / f"{machine_id}.txt").write_text(
        "0\n0\n1\n0\n0\n0\n0\n0\n",
        encoding="utf-8",
    )


def test_smd_baseline_config_accepts_valid_values(tmp_path) -> None:
    SMDBaselineConfig(root=tmp_path).validate()


def test_smd_baseline_config_rejects_invalid_validation_fraction(tmp_path) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        SMDBaselineConfig(root=tmp_path, validation_fraction=1.0).validate()


def test_smd_baseline_config_keeps_machine_subset(tmp_path: Path) -> None:
    config = SMDBaselineConfig(root=tmp_path, machine_ids=("machine-1-1",))

    assert config.machine_ids == ("machine-1-1",)


def test_build_smd_baseline_datasets_returns_windowed_splits(tmp_path) -> None:
    write_smd_fixture(tmp_path)

    train, validation, test, standardizers = build_smd_baseline_datasets(
        SMDBaselineConfig(
            root=tmp_path,
            lookback=2,
            horizon=1,
            validation_fraction=0.25,
        )
    )

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0
    assert len(standardizers) == 1
