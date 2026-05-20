"""Collect and compare archived experiment run metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aaf.experiments.manifest import load_run_manifest

FlatMetrics = dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class RunComparisonRow:
    """Flattened metrics and metadata for one archived run."""

    run_dir: Path
    values: FlatMetrics


def flatten_mapping(
    payload: dict[str, Any],
    *,
    prefix: str = "",
) -> FlatMetrics:
    """Flatten nested metric dictionaries using dotted keys."""

    flattened: FlatMetrics = {}
    for key, value in payload.items():
        field = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_mapping(value, prefix=field))
        elif isinstance(value, str | int | float | bool) or value is None:
            flattened[field] = value
        else:
            flattened[field] = str(value)
    return flattened


def collect_run_row(run_dir: Path) -> RunComparisonRow:
    """Collect one run directory into a flattened comparison row."""

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.json in run directory: {run_dir}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metrics.json must contain a JSON object")
    values: FlatMetrics = {"run_dir": str(run_dir)}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        values.update(flatten_mapping({"manifest": load_run_manifest(manifest_path).to_dict()}))
    values.update(flatten_mapping(payload))
    return RunComparisonRow(run_dir=run_dir, values=values)
