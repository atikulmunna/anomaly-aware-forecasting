"""Command-line entry point for running experiment suites."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aaf.experiments.suite import (
    apply_suite_param_overrides,
    load_experiment_suite,
    run_experiment_suite,
    validate_suite_configs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AAF experiment suite.")
    parser.add_argument("suite", type=Path, help="JSON suite definition.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a parameter for every suite job. VALUE may be JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate suite config without running jobs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite = apply_suite_param_overrides(
        load_experiment_suite(args.suite),
        _parse_overrides(tuple(args.set)),
    )
    validate_suite_configs(suite)
    if args.dry_run:
        return 0
    run_experiment_suite(
        suite,
        args.output_root,
        compare_output=args.compare,
        overwrite=args.overwrite,
    )
    return 0


def _parse_overrides(values: tuple[str, ...]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--set values must use KEY=VALUE")
        key, raw = value.split("=", 1)
        if not key:
            raise ValueError("--set key must be non-empty")
        overrides[key] = _parse_override_value(raw)
    return overrides


def _parse_override_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
