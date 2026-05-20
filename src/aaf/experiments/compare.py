"""Collect and compare archived experiment run metrics."""

from __future__ import annotations

from typing import Any

FlatMetrics = dict[str, str | int | float | bool | None]


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
