from pathlib import Path

import numpy as np
import pytest

from aaf.data.smd import (
    SMDMachineSplit,
    SMDPreparedMachine,
    fit_smd_standardizer,
    list_smd_machine_ids,
    load_smd_labels,
    load_smd_machine,
    load_smd_machines,
    load_smd_matrix,
    make_smd_series,
    prepare_smd_machine,
    standardize_smd_machine,
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


def test_fit_smd_standardizer_uses_train_split_only() -> None:
    split = SMDMachineSplit(
        machine_id="machine-1-1",
        train=np.array([[0.0], [2.0], [4.0]]),
        test=np.array([[100.0]]),
        test_labels=np.array([0]),
    )

    standardizer = fit_smd_standardizer(split)

    assert standardizer.mean.tolist() == [2.0]


def test_standardize_smd_machine_preserves_labels() -> None:
    split = SMDMachineSplit(
        machine_id="machine-1-1",
        train=np.array([[0.0], [2.0], [4.0]]),
        test=np.array([[2.0], [4.0]]),
        test_labels=np.array([0, 1]),
    )

    standardized = standardize_smd_machine(split, fit_smd_standardizer(split))

    assert standardized.machine_id == split.machine_id
    assert standardized.test_labels.tolist() == [0, 1]
    assert standardized.train.mean() == pytest.approx(0.0)


def test_smd_prepared_machine_validates_split_lengths() -> None:
    prepared = SMDPreparedMachine(
        machine_id="machine-1-1",
        train=np.ones((4, 2)),
        validation=np.ones((2, 2)),
        test=np.ones((3, 2)),
        validation_labels=np.zeros(2, dtype=np.int64),
        test_labels=np.array([0, 1, 0]),
    )

    prepared.validate()


def test_prepare_smd_machine_splits_validation_from_training_tail() -> None:
    split = SMDMachineSplit(
        machine_id="machine-1-1",
        train=np.arange(20, dtype=np.float64).reshape(10, 2),
        test=np.ones((4, 2)),
        test_labels=np.array([0, 1, 0, 0]),
    )

    prepared, standardizer = prepare_smd_machine(split, validation_fraction=0.3)

    assert prepared.train.shape[0] == 7
    assert prepared.validation.shape[0] == 3
    assert prepared.validation_labels.tolist() == [0, 0, 0]
    assert standardizer.mean.shape == (2,)


def test_make_smd_series_uses_dummy_regime_labels() -> None:
    series = make_smd_series(
        np.ones((3, 2)),
        np.array([0, 1, 0]),
        config_id="machine-1-1-test",
    )

    assert series.config_id == "machine-1-1-test"
    assert series.regime_labels.tolist() == [0, 0, 0]
    assert series.anomaly_labels.tolist() == [0, 1, 0]
