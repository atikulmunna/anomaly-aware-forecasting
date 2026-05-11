"""Server Machine Dataset loading and preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aaf.data.preprocessing import Standardizer
from aaf.data.synthetic import FloatArray, IntArray


@dataclass(frozen=True)
class SMDMachineSplit:
    """Train/test arrays and test anomaly labels for one SMD machine."""

    machine_id: str
    train: FloatArray
    test: FloatArray
    test_labels: IntArray

    def validate(self) -> None:
        if not self.machine_id:
            raise ValueError("machine_id must be non-empty")
        if self.train.ndim != 2 or self.test.ndim != 2:
            raise ValueError("SMD observations must have shape (T, D)")
        if self.train.shape[1] != self.test.shape[1]:
            raise ValueError("train and test channel counts must match")
        if self.test_labels.shape != (self.test.shape[0],):
            raise ValueError("test_labels must match test length")
        if np.any(~np.isfinite(self.train)) or np.any(~np.isfinite(self.test)):
            raise ValueError("SMD observations must be finite")
        if np.any((self.test_labels != 0) & (self.test_labels != 1)):
            raise ValueError("test labels must be binary")


def list_smd_machine_ids(root: Path) -> tuple[str, ...]:
    """Return machine ids that have train, test, and test-label files."""

    train_dir = root / "train"
    test_dir = root / "test"
    label_dir = root / "test_label"
    if not train_dir.exists():
        raise FileNotFoundError(f"missing SMD train directory: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"missing SMD test directory: {test_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"missing SMD test_label directory: {label_dir}")

    train_ids = _machine_ids(train_dir)
    test_ids = _machine_ids(test_dir)
    label_ids = _machine_ids(label_dir)
    return tuple(sorted(train_ids & test_ids & label_ids))


def load_smd_machine(root: Path, machine_id: str) -> SMDMachineSplit:
    """Load one machine split from a standard SMD directory tree."""

    split = SMDMachineSplit(
        machine_id=machine_id,
        train=load_smd_matrix(root / "train" / f"{machine_id}.txt"),
        test=load_smd_matrix(root / "test" / f"{machine_id}.txt"),
        test_labels=load_smd_labels(root / "test_label" / f"{machine_id}.txt"),
    )
    split.validate()
    return split


def load_smd_machines(
    root: Path,
    machine_ids: tuple[str, ...] | None = None,
) -> tuple[SMDMachineSplit, ...]:
    """Load multiple SMD machine splits in deterministic id order."""

    selected_ids = list_smd_machine_ids(root) if machine_ids is None else machine_ids
    if len(selected_ids) == 0:
        raise ValueError("at least one SMD machine id is required")
    return tuple(load_smd_machine(root, machine_id) for machine_id in selected_ids)


def fit_smd_standardizer(split: SMDMachineSplit) -> Standardizer:
    """Fit a per-machine scaler using the SMD training split only."""

    split.validate()
    return Standardizer.fit(split.train)


def standardize_smd_machine(
    split: SMDMachineSplit,
    standardizer: Standardizer,
) -> SMDMachineSplit:
    """Apply a fitted scaler to train and test observations without touching labels."""

    standardized = SMDMachineSplit(
        machine_id=split.machine_id,
        train=standardizer.transform(split.train),
        test=standardizer.transform(split.test),
        test_labels=split.test_labels.copy(),
    )
    standardized.validate()
    return standardized


def _machine_ids(directory: Path) -> set[str]:
    return {path.stem for path in directory.iterdir() if path.is_file() and path.suffix == ".txt"}


def load_smd_matrix(path: Path) -> FloatArray:
    """Load one SMD observation matrix from a comma-separated text file."""

    if not path.exists():
        raise FileNotFoundError(path)
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("SMD matrix must have shape (T, D)")
    if np.any(~np.isfinite(values)):
        raise ValueError("SMD matrix must contain only finite values")
    return np.asarray(values, dtype=np.float64)


def load_smd_labels(path: Path) -> IntArray:
    """Load one SMD anomaly-label vector from a text file."""

    if not path.exists():
        raise FileNotFoundError(path)
    values = np.loadtxt(path, delimiter=",", dtype=np.int64)
    labels = np.asarray(values, dtype=np.int64).reshape(-1)
    if labels.shape[0] == 0:
        raise ValueError("SMD labels must be non-empty")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("SMD labels must be binary")
    return labels
