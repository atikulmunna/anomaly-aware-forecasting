"""Run manifest metadata for reproducible experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunManifest:
    """Small metadata record stored beside run artifacts."""

    run_id: str
    pipeline: str
    dataset: str
    seed: int | None = None
    notes: str | None = None

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.pipeline:
            raise ValueError("pipeline must be non-empty")
        if not self.dataset:
            raise ValueError("dataset must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    """Write a run manifest JSON file."""

    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_run_manifest(path: Path) -> RunManifest:
    """Load a run manifest JSON file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run manifest must be a JSON object")
    manifest = RunManifest(
        run_id=str(payload["run_id"]),
        pipeline=str(payload["pipeline"]),
        dataset=str(payload["dataset"]),
        seed=None if payload.get("seed") is None else int(payload["seed"]),
        notes=None if payload.get("notes") is None else str(payload["notes"]),
    )
    manifest.validate()
    return manifest
