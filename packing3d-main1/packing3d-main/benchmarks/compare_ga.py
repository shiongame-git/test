"""Run explicit BL-GA / PP-GA comparison trials.

The report is a research/benchmark artifact, not the standard packing-result
JSON and not a serializer for complete GA or PP state.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from packing import (
    BLPriority,
    BoxOrderMode,
    ComparisonDecoder,
    DecoderConfig,
    GAConfig,
    GAExecutionConfig,
    parse_box_sizes_bytes,
    run_comparison_trial,
    write_decoder_profile_json,
    write_comparison_report_json,
)

from packing.source import SOURCE_LABEL
REPORT_FORMAT = "packing3d-comparison-report"
REPORT_VERSION = 4

EXPECTED = {
    "box_sizes_cube.txt": "b57a8c73bc7b66b9807e55d58d17a23fa9294ce652c98597642ea0ed889d906a",
    "box_sizes_cuboid.txt": "908bb73f607d377e76b8e50c92f04a3c029e05cc3eee57e9b988c36352f3da9f",
}


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part) for part in value.split(",") if part != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not parsed or any(seed < 0 for seed in parsed):
        raise argparse.ArgumentTypeError("seeds must contain at least one integer >= 0")
    return parsed


def _support_percent(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "min-support-percent must be an integer from 0 to 100"
        ) from exc
    if not 0 <= parsed <= 100:
        raise argparse.ArgumentTypeError(
            "min-support-percent must be an integer from 0 to 100"
        )
    return parsed


def _choice_values(raw: str, enum_type):
    if raw == "both":
        return tuple(enum_type)
    return (enum_type(raw),)


def _trial_dict(trial) -> dict[str, object]:
    return {
        "run_metadata": trial.run_metadata.to_dict(),
        "best": {
            "generation_index": trial.best_generation_index,
            "population_index": trial.best_population_index,
            "genes": list(trial.best_genes),
            "fitness": trial.best_fitness,
            "placed_count": trial.placed_count,
            "failed_count": trial.failed_count,
            "placed_volume": trial.placed_volume,
            "density": trial.density,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", nargs="*", type=Path)
    parser.add_argument("--decoder", choices=("both", "BL", "PP"), default="both")
    parser.add_argument(
        "--candidate-policy",
        choices=("Cartesian", "LocalNeighbors"),
        default="Cartesian",
    )
    parser.add_argument("--min-support-percent", type=_support_percent, default=55)
    parser.add_argument(
        "--priority",
        choices=("both", *(priority.value for priority in BLPriority)),
        default=BLPriority.BOTTOM_FIRST.value,
    )
    parser.add_argument(
        "--order",
        choices=("both", *(mode.value for mode in BoxOrderMode)),
        default=BoxOrderMode.SOURCE.value,
    )
    parser.add_argument("--seeds", type=_csv_ints, default=(0,), help="comma-separated nonnegative integers")
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--crossover", type=float, default=1.0)
    parser.add_argument("--mutation", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="write one run-linked decoder profile JSON per trial into this directory",
    )
    args = parser.parse_args()
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[1]
    paths = args.fixtures or [root / "data" / name for name in EXPECTED]
    decoders = _choice_values(args.decoder, ComparisonDecoder)
    priorities = _choice_values(args.priority, BLPriority)
    orders = _choice_values(args.order, BoxOrderMode)
    config = GAConfig(
        population_size=args.population,
        generation_count=args.generations,
        crossover_rate=float(args.crossover),
        mutation_rate=float(args.mutation),
    )
    execution = GAExecutionConfig(workers=args.workers, cache=args.cache)
    decoder_config = DecoderConfig(
        candidate_policy=args.candidate_policy,
        candidate_policy_version=1,
        min_support_percent=args.min_support_percent,
        support_model_version="support-union-v1",
    )

    trials: list[dict[str, object]] = []
    for path in paths:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        expected = EXPECTED.get(path.name)
        if expected is not None and digest != expected:
            raise SystemExit(f"fixture hash mismatch for {path}: {digest}")
        container, boxes = parse_box_sizes_bytes(raw, source=str(path))
        for priority in priorities:
            for order_mode in orders:
                for seed in args.seeds:
                    for decoder in decoders:
                        trial = run_comparison_trial(
                            container,
                            boxes,
                            decoder=decoder,
                            priority=priority,
                            order_mode=order_mode,
                            config=config,
                            seed=seed,
                            execution=execution,
                            decoder_config=decoder_config,
                            implementation_version=SOURCE_LABEL,
                            origin="compare_cli",
                            problem_source=str(path),
                            fixture_sha256=digest,
                            profile=args.profile_dir is not None,
                        )
                        row = _trial_dict(trial)
                        trials.append(row)
                        if args.profile_dir is not None:
                            if trial.profile is None or trial.run_metadata is None:
                                raise RuntimeError("profile output requested but trial has no profile")
                            profile_path = args.profile_dir / f"{trial.profile.run_id}.json"
                            if profile_path.exists():
                                raise SystemExit(f"profile output already exists: {profile_path}")
                            write_decoder_profile_json(profile_path, trial.profile)
                        print(
                            f"{path.name} decoder={decoder.value} priority={priority.value} "
                            f"order={order_mode.value} seed={seed} "
                            f"density={trial.density:.9f} placed={trial.placed_count} "
                            f"elapsed_s={trial.elapsed_seconds:.6f}"
                        )

    report = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "implementation_version": SOURCE_LABEL,
        "trials": trials,
    }
    from packing.io_run_metadata import render_json, validate_comparison_report
    validate_comparison_report(report)
    rendered = render_json(report) + "\n"
    if args.output is not None:
        write_comparison_report_json(args.output, report)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
