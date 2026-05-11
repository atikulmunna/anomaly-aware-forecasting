from pathlib import Path

import numpy as np
import pytest

from aaf.data.smd import (
    SMDMachineSplit,
    list_smd_machine_ids,
    load_smd_labels,
    load_smd_machine,
    load_smd_machines,
    load_smd_matrix,
)


def write_smd_fixture(root: Path, machine_id: str = "machine-1-1") -> None:
    for directory in ("train", "test", "test_label"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "train" / f"{machine_id}.txt").write_text("1,2\n3,4\n", encoding="utf-8")
    (root / "test" / f"{machine_id}.txt").write_text("1,2\n3,4\n", encoding="utf-8")
    (root / "test_label" / f"{machine_id}.txt").write_text("0\n1\n", encoding="utf-8")


def test_list_smd_machine_ids_returns_complete_machine_intersection(tmp_path) -> None:
    write_smd_fixture(tmp_path, "machine-1-1")
    write_smd_fixture(tmp_path, "machine-1-2")
    (tmp_path / "train" / "train-only.txt").write_text("1,2\n", encoding="utf-8")

    assert list_smd_machine_ids(tmp_path) == ("machine-1-1", "machine-1-2")


def test_list_smd_machine_ids_requires_expected_directories(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="train"):
        list_smd_machine_ids(tmp_path)


def test_load_smd_matrix_reads_comma_separated_observations(tmp_path) -> None:
    path = tmp_path / "machine.txt"
    path.write_text("1.0,2.0\n3.0,4.0\n", encoding="utf-8")

    matrix = load_smd_matrix(path)

    assert matrix.shape == (2, 2)
    assert matrix.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_load_smd_matrix_promotes_univariate_files(tmp_path) -> None:
    path = tmp_path / "machine.txt"
    path.write_text("1.0\n2.0\n", encoding="utf-8")

    assert load_smd_matrix(path).shape == (2, 1)


def test_load_smd_labels_reads_binary_vector(tmp_path) -> None:
    path = tmp_path / "labels.txt"
    path.write_text("0\n1\n0\n", encoding="utf-8")

    assert load_smd_labels(path).tolist() == [0, 1, 0]


def test_load_smd_labels_rejects_non_binary_values(tmp_path) -> None:
    path = tmp_path / "labels.txt"
    path.write_text("0\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="binary"):
        load_smd_labels(path)


def test_smd_machine_split_validates_shapes() -> None:
    split = SMDMachineSplit(
        machine_id="machine-1-1",
        train=np.ones((4, 2)),
        test=np.ones((3, 2)),
        test_labels=np.array([0, 1, 0]),
    )

    split.validate()


def test_smd_machine_split_rejects_label_length_mismatch() -> None:
    split = SMDMachineSplit(
        machine_id="machine-1-1",
        train=np.ones((4, 2)),
        test=np.ones((3, 2)),
        test_labels=np.array([0, 1]),
    )

    with pytest.raises(ValueError, match="test_labels"):
        split.validate()


def test_load_smd_machine_reads_standard_directory_tree(tmp_path) -> None:
    write_smd_fixture(tmp_path, "machine-1-1")
    (tmp_path / "test_label" / "machine-1-1.txt").write_text("0\n1\n", encoding="utf-8")

    split = load_smd_machine(tmp_path, "machine-1-1")

    assert split.machine_id == "machine-1-1"
    assert split.train.shape == (2, 2)
    assert split.test.shape == (2, 2)
    assert split.test_labels.tolist() == [0, 1]


def test_load_smd_machines_uses_discovered_order(tmp_path) -> None:
    write_smd_fixture(tmp_path, "machine-1-2")
    write_smd_fixture(tmp_path, "machine-1-1")

    splits = load_smd_machines(tmp_path)

    assert [split.machine_id for split in splits] == ["machine-1-1", "machine-1-2"]


def test_load_smd_machines_accepts_explicit_subset(tmp_path) -> None:
    write_smd_fixture(tmp_path, "machine-1-1")
    write_smd_fixture(tmp_path, "machine-1-2")

    splits = load_smd_machines(tmp_path, ("machine-1-2",))

    assert [split.machine_id for split in splits] == ["machine-1-2"]
