"""Command-line tools for experiment comparison."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from aaf.experiments.compare import (
    collect_run_rows,
    write_comparison_csv,
    write_comparison_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare archived AAF run metrics.")
    parser.add_argument("root", type=Path, help="Directory containing run subdirectories.")
    parser.add_argument("--output", type=Path, required=True, help="CSV or JSON output path.")
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default=None,
        help="Output format. Defaults to the output file extension.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = collect_run_rows(args.root)
    output_format = args.format or _format_from_output_path(args.output)
    if output_format == "csv":
        write_comparison_csv(args.output, rows)
    else:
        write_comparison_json(args.output, rows)
    return 0


def _format_from_output_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    raise ValueError("output path must end with .csv or .json when --format is omitted")


if __name__ == "__main__":
    raise SystemExit(main())
