"""Command-line entry points for evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

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
    parser.add_argument(
        "--threshold-strategy",
        default="max_validation_f1",
        help="Validation-only anomaly threshold selection strategy.",
    )
    parser.add_argument(
        "--persistence-window",
        type=int,
        default=1,
        help="Trailing window for anomaly prediction persistence filtering.",
    )
    parser.add_argument(
        "--persistence-count",
        type=int,
        default=1,
        help="Minimum positives in the trailing persistence window.",
    )
    parser.add_argument(
        "--skip-forecast",
        action="store_true",
        help="Skip forecast metrics even if forecast artifacts are present.",
    )
    parser.add_argument(
        "--skip-anomaly",
        action="store_true",
        help="Skip anomaly metrics even if anomaly artifacts are present.",
    )
    parser.add_argument(
        "--skip-regime",
        action="store_true",
        help="Skip regime metrics even if regime artifacts are present.",
    )
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
        threshold_strategy=args.threshold_strategy,
        persistence_window=args.persistence_window,
        persistence_count=args.persistence_count,
        include_forecast=not args.skip_forecast,
        include_anomaly=not args.skip_anomaly,
        include_regime=not args.skip_regime,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
