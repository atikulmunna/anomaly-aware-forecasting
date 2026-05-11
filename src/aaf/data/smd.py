"""Server Machine Dataset loading and preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
