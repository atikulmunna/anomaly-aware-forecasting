import json

import pytest

from aaf.experiments import RunManifest, load_run_manifest, write_run_manifest


def test_run_manifest_validates_required_fields() -> None:
    with pytest.raises(ValueError, match="run_id"):
        RunManifest(run_id="", pipeline="joint", dataset="synthetic").validate()


def test_run_manifest_round_trips_json(tmp_path) -> None:
    manifest = RunManifest(
        run_id="run-001",
        pipeline="smd-joint",
        dataset="smd",
        seed=7,
        notes="smoke",
    )

    write_run_manifest(tmp_path / "manifest.json", manifest)
    loaded = load_run_manifest(tmp_path / "manifest.json")

    assert loaded == manifest
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["pipeline"] == "smd-joint"
