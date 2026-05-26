"""Command-line entry point for anomaly-only rescoring."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from aaf.eval.anomaly import validate_persistence, validate_threshold_strategy
from aaf.eval.rescore import AnomalyRescoreJob, make_rescore_run_id, rescore_anomaly_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rescore archived anomaly artifacts without retraining.",
    )
    parser.add_argument("source_run_dir", type=Path, help="Run directory with anomaly artifacts.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        default=[],
        help="Threshold strategy to evaluate. May be repeated.",
    )
    parser.add_argument(
        "--persistence",
        action="append",
        default=[],
        metavar="WINDOW:COUNT",
        help="Persistence setting to evaluate. May be repeated.",
    )
    parser.add_argument(
        "--run-id-prefix",
        default=None,
        help="Optional prefix for generated rescore run ids.",
    )
    parser.add_argument("--pipeline", default=None, help="Manifest pipeline override.")
    parser.add_argument("--dataset", default=None, help="Manifest dataset override.")
    parser.add_argument("--seed", type=int, default=None, help="Manifest seed override.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    strategies = tuple(args.strategies) or ("validation_quantile_99",)
    _validate_strategies(strategies)
    persistence_settings = tuple(_parse_persistence(value) for value in args.persistence) or (
        (1, 1),
    )
    jobs = tuple(
        AnomalyRescoreJob(
            run_id=make_rescore_run_id(
                args.source_run_dir,
                strategy,
                window,
                count,
                prefix=args.run_id_prefix,
            ),
            threshold_strategy=strategy,
            persistence_window=window,
            persistence_count=count,
            notes=(
                f"Anomaly-only rescore of {args.source_run_dir} with "
                f"{strategy}, {count}-of-{window} persistence"
            ),
        )
        for strategy in strategies
        for window, count in persistence_settings
    )
    rescore_anomaly_run(
        args.source_run_dir,
        args.output_root,
        jobs,
        pipeline=args.pipeline,
        dataset=args.dataset,
        seed=args.seed,
        compare_output=args.compare,
        overwrite=args.overwrite,
    )
    return 0


def _parse_persistence(value: str) -> tuple[int, int]:
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise ValueError("--persistence must use WINDOW:COUNT")
    window = int(parts[0])
    count = int(parts[1])
    validate_persistence(window=window, count=count)
    return window, count


def _validate_strategies(strategies: tuple[str, ...]) -> None:
    for strategy in strategies:
        validate_threshold_strategy(strategy)


if __name__ == "__main__":
    raise SystemExit(main())
