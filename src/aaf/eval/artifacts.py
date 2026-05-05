"""Helpers for writing evaluation artifact files."""

from __future__ import annotations

import json
from pathlib import Path

from aaf.eval.diagnostics import mixture_diagnostics
from aaf.eval.forecasting import MixtureForecast


def write_mixture_diagnostics_json(
    path: Path,
    *,
    validation: MixtureForecast | None = None,
    test: MixtureForecast | None = None,
) -> None:
    """Write split-level mixture diagnostics to JSON."""

    payload: dict[str, object] = {}
    if validation is not None:
        payload["validation"] = mixture_diagnostics(validation).to_dict()
    if test is not None:
        payload["test"] = mixture_diagnostics(test).to_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
