"""Command-line entry points for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from aaf.eval.report import evaluate_run_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an anomaly-aware forecasting run.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing evaluation artifacts.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for metrics JSON. Defaults to <run_dir>/metrics.json.",
    )
    parser.add_argument(
        "--energy-samples",
        type=int,
        default=256,
        help="Monte Carlo samples used for Energy Score.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampled metrics.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output or args.run_dir / "metrics.json"
    evaluate_run_directory(
        args.run_dir,
        output_path=output,
        energy_samples=args.energy_samples,
        seed=args.seed,
    )
    return 0
