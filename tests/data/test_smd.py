from pathlib import Path

import pytest

from aaf.data.smd import list_smd_machine_ids, load_smd_matrix


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
