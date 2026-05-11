from pathlib import Path

import pytest

from aaf.data.smd import list_smd_machine_ids


def write_smd_fixture(root: Path, machine_id: str = "machine-1-1") -> None:
    for directory in ("train", "test", "test_label"):
        (root / directory).mkdir(parents=True, exist_ok=True)
        (root / directory / f"{machine_id}.txt").write_text("1,2\n3,4\n", encoding="utf-8")


def test_list_smd_machine_ids_returns_complete_machine_intersection(tmp_path) -> None:
    write_smd_fixture(tmp_path, "machine-1-1")
    write_smd_fixture(tmp_path, "machine-1-2")
    (tmp_path / "train" / "train-only.txt").write_text("1,2\n", encoding="utf-8")

    assert list_smd_machine_ids(tmp_path) == ("machine-1-1", "machine-1-2")


def test_list_smd_machine_ids_requires_expected_directories(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="train"):
        list_smd_machine_ids(tmp_path)
