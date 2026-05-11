"""Server Machine Dataset loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path


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
