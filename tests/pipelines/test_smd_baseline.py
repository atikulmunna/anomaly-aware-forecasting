from pathlib import Path

import pytest

from aaf.pipelines.smd_baseline import SMDBaselineConfig


def test_smd_baseline_config_accepts_valid_values(tmp_path) -> None:
    SMDBaselineConfig(root=tmp_path).validate()


def test_smd_baseline_config_rejects_invalid_validation_fraction(tmp_path) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        SMDBaselineConfig(root=tmp_path, validation_fraction=1.0).validate()


def test_smd_baseline_config_keeps_machine_subset(tmp_path: Path) -> None:
    config = SMDBaselineConfig(root=tmp_path, machine_ids=("machine-1-1",))

    assert config.machine_ids == ("machine-1-1",)
