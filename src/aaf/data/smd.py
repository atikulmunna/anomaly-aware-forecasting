"""Server Machine Dataset loading and preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aaf.data.preprocessing import Standardizer, WindowedDataset, make_windowed_dataset
from aaf.data.synthetic import FloatArray, IntArray, SyntheticSeries


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


@dataclass(frozen=True)
class SMDPreparedMachine:
    """Standardized train/validation/test arrays for one SMD machine."""

    machine_id: str
    train: FloatArray
    validation: FloatArray
    test: FloatArray
    validation_labels: IntArray
    test_labels: IntArray

    def validate(self) -> None:
        if self.train.ndim != 2 or self.validation.ndim != 2 or self.test.ndim != 2:
            raise ValueError("prepared SMD observations must have shape (T, D)")
        if (
            self.train.shape[1] != self.validation.shape[1]
            or self.train.shape[1] != self.test.shape[1]
        ):
            raise ValueError("prepared SMD channel counts must match")
        if self.validation_labels.shape != (self.validation.shape[0],):
            raise ValueError("validation_labels must match validation length")
        if self.test_labels.shape != (self.test.shape[0],):
            raise ValueError("test_labels must match test length")


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


def prepare_smd_machine(
    split: SMDMachineSplit,
    *,
    validation_fraction: float = 0.2,
) -> tuple[SMDPreparedMachine, Standardizer]:
    """Standardize one SMD machine and carve validation from the training tail."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    standardizer = fit_smd_standardizer(split)
    standardized = standardize_smd_machine(split, standardizer)
    validation_length = max(1, int(round(standardized.train.shape[0] * validation_fraction)))
    if validation_length >= standardized.train.shape[0]:
        raise ValueError("validation split leaves no training observations")
    train_end = standardized.train.shape[0] - validation_length
    prepared = SMDPreparedMachine(
        machine_id=standardized.machine_id,
        train=standardized.train[:train_end],
        validation=standardized.train[train_end:],
        test=standardized.test,
        validation_labels=np.zeros(validation_length, dtype=np.int64),
        test_labels=standardized.test_labels,
    )
    prepared.validate()
    return prepared, standardizer


def make_smd_series(
    observations: FloatArray,
    anomaly_labels: IntArray,
    *,
    config_id: str,
) -> SyntheticSeries:
    """Wrap SMD arrays in the common series container with a dummy regime label."""

    series = SyntheticSeries(
        observations=np.asarray(observations, dtype=np.float64),
        regime_labels=np.zeros(observations.shape[0], dtype=np.int64),
        anomaly_labels=np.asarray(anomaly_labels, dtype=np.int64),
        config_id=config_id,
    )
    series.validate()
    return series


def make_smd_windowed_splits(
    prepared: SMDPreparedMachine,
    *,
    lookback: int,
    horizon: int,
    stride: int = 1,
) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset]:
    """Create train, validation, and test windows for a prepared SMD machine."""

    prepared.validate()
    return (
        make_windowed_dataset(
            make_smd_series(
                prepared.train,
                np.zeros(prepared.train.shape[0], dtype=np.int64),
                config_id=f"{prepared.machine_id}-train",
            ),
            lookback=lookback,
            horizon=horizon,
            stride=stride,
            dtype=np.float32,
        ),
        make_windowed_dataset(
            make_smd_series(
                prepared.validation,
                prepared.validation_labels,
                config_id=f"{prepared.machine_id}-validation",
            ),
            lookback=lookback,
            horizon=horizon,
            stride=stride,
            dtype=np.float32,
        ),
        make_windowed_dataset(
            make_smd_series(
                prepared.test,
                prepared.test_labels,
                config_id=f"{prepared.machine_id}-test",
            ),
            lookback=lookback,
            horizon=horizon,
            stride=stride,
            dtype=np.float32,
        ),
    )


def concat_windowed_datasets(datasets: tuple[WindowedDataset, ...]) -> WindowedDataset:
    """Concatenate windowed datasets produced for multiple SMD machines."""

    if len(datasets) == 0:
        raise ValueError("at least one windowed dataset is required")
    return WindowedDataset(
        windows=np.concatenate([dataset.windows for dataset in datasets], axis=0),
        targets=np.concatenate([dataset.targets for dataset in datasets], axis=0),
        regime_labels=np.concatenate([dataset.regime_labels for dataset in datasets], axis=0),
        anomaly_labels=np.concatenate([dataset.anomaly_labels for dataset in datasets], axis=0),
    )


def prepare_smd_windowed_datasets(
    root: Path,
    *,
    machine_ids: tuple[str, ...] | None = None,
    validation_fraction: float = 0.2,
    lookback: int,
    horizon: int,
    stride: int = 1,
) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset, tuple[Standardizer, ...]]:
    """Load, standardize, split, and window SMD machines."""

    splits = load_smd_machines(root, machine_ids)
    prepared = tuple(
        prepare_smd_machine(split, validation_fraction=validation_fraction) for split in splits
    )
    windowed = tuple(
        make_smd_windowed_splits(
            machine,
            lookback=lookback,
            horizon=horizon,
            stride=stride,
        )
        for machine, _standardizer in prepared
    )
    return (
        concat_windowed_datasets(tuple(item[0] for item in windowed)),
        concat_windowed_datasets(tuple(item[1] for item in windowed)),
        concat_windowed_datasets(tuple(item[2] for item in windowed)),
        tuple(standardizer for _machine, standardizer in prepared),
    )


def prepare_smd_windowed_datasets_with_machine_ids(
    root: Path,
    *,
    machine_ids: tuple[str, ...] | None = None,
    validation_fraction: float = 0.2,
    lookback: int,
    horizon: int,
    stride: int = 1,
) -> tuple[
    WindowedDataset,
    WindowedDataset,
    WindowedDataset,
    tuple[Standardizer, ...],
    IntArray,
    IntArray,
    IntArray,
]:
    """Load SMD windowed datasets and integer machine ids for each window."""

    splits = load_smd_machines(root, machine_ids)
    prepared = tuple(
        prepare_smd_machine(split, validation_fraction=validation_fraction) for split in splits
    )
    windowed = tuple(
        make_smd_windowed_splits(
            machine,
            lookback=lookback,
            horizon=horizon,
            stride=stride,
        )
        for machine, _standardizer in prepared
    )
    train_splits = tuple(item[0] for item in windowed)
    validation_splits = tuple(item[1] for item in windowed)
    test_splits = tuple(item[2] for item in windowed)
    return (
        concat_windowed_datasets(train_splits),
        concat_windowed_datasets(validation_splits),
        concat_windowed_datasets(test_splits),
        tuple(standardizer for _machine, standardizer in prepared),
        concat_machine_id_arrays(train_splits),
        concat_machine_id_arrays(validation_splits),
        concat_machine_id_arrays(test_splits),
    )


def concat_machine_id_arrays(datasets: tuple[WindowedDataset, ...]) -> IntArray:
    """Return one integer machine id per window for concatenated SMD splits."""

    if len(datasets) == 0:
        raise ValueError("at least one windowed dataset is required")
    return np.concatenate(
        [
            np.full(len(dataset), machine_index, dtype=np.int64)
            for machine_index, dataset in enumerate(datasets)
        ],
        axis=0,
    )


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
