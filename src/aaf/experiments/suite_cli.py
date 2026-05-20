"""Command-line entry point for running experiment suites."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from aaf.experiments.suite import load_experiment_suite, run_experiment_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AAF experiment suite.")
    parser.add_argument("suite", type=Path, help="JSON suite definition.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_experiment_suite(
        load_experiment_suite(args.suite),
        args.output_root,
        compare_output=args.compare,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
