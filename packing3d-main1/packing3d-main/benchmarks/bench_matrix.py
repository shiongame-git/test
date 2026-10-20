"""Collect non-overwriting packing performance measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from packing import (
    BLPriority,
    Box,
    ComparisonDecoder,
    Container,
    DecoderConfig,
    GAConfig,
    GAExecutionConfig,
    pack,
    read_box_sizes,
    run_comparison_trial,
    validate_result,
    validate_support,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = (
    ("cube", ROOT / "data" / "box_sizes_cube.txt", "b57a8c73bc7b66b9807e55d58d17a23fa9294ce652c98597642ea0ed889d906a"),
    ("cuboid", ROOT / "data" / "box_sizes_cuboid.txt", "908bb73f607d377e76b8e50c92f04a3c029e05cc3eee57e9b988c36352f3da9f"),
)


class _MeasurementTimeBudget:
    def __init__(self, seconds: int) -> None:
        if type(seconds) is not int or seconds <= 0:
            raise ValueError("time budget must be a positive integer number of seconds")
        self.seconds = seconds
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def checkpoint(self, label: str) -> None:
        if self.elapsed() >= self.seconds:
            raise TimeoutError(
                f"measurement time budget of {self.seconds}s expired at {label}; "
                "no acceptance artifacts were written"
            )

    def run(self, label: str, call: Callable[[], object]) -> tuple[object, float]:
        self.checkpoint(f"before {label}")
        started = time.perf_counter()
        result = call()
        elapsed = time.perf_counter() - started
        self.checkpoint(f"after {label}")
        return result, elapsed
DECODER_CONFIG = DecoderConfig(
    candidate_policy="Cartesian",
    candidate_policy_version=1,
    min_support_percent=55,
    support_model_version="support-union-v1",
)
GA_CONFIG = GAConfig(
    population_size=4,
    generation_count=3,
    crossover_rate=0.8,
    mutation_rate=0.1,
)


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _cpu_name() -> str:
    return platform.processor().strip() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")


def _decoder_config_dict() -> dict[str, object]:
    return {
        "candidate_policy": DECODER_CONFIG.candidate_policy,
        "candidate_policy_version": DECODER_CONFIG.candidate_policy_version,
        "min_support_percent": DECODER_CONFIG.min_support_percent,
        "support_model_version": DECODER_CONFIG.support_model_version,
    }


def _summary(result) -> dict[str, object]:
    return {
        "placed_count": result.placed_count,
        "failed_count": result.failed_count,
        "placed_volume": result.placed_volume,
        "density": result.density,
    }


def _measure_bl(budget: _MeasurementTimeBudget) -> list[dict[str, object]]:
    measurements: list[dict[str, object]] = []
    for fixture_name, path, expected_hash in FIXTURES:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise RuntimeError(f"fixture hash mismatch: {path}: {digest}")
        container, boxes = read_box_sizes(path)

        for priority in BLPriority:
            baseline, warmup_seconds = budget.run(
                f"BL {fixture_name}/{priority.value} warm-up",
                lambda: pack(
                    container,
                    boxes,
                    priority=priority,
                    decoder_config=DECODER_CONFIG,
                ),
            )
            errors = validate_result(
                baseline,
                expected_container=container,
                ordered_boxes=boxes,
            )
            support_errors = validate_support(baseline, settings=DECODER_CONFIG)
            if errors or support_errors:
                raise RuntimeError(f"invalid BL warm-up result: {errors + support_errors}")

            times: list[float] = []
            for repeat in range(3):
                result, elapsed = budget.run(
                    f"BL {fixture_name}/{priority.value} repeat {repeat + 1}",
                    lambda: pack(
                        container,
                        boxes,
                        priority=priority,
                        decoder_config=DECODER_CONFIG,
                    ),
                )
                times.append(elapsed)
                errors = validate_result(
                    result,
                    expected_container=container,
                    ordered_boxes=boxes,
                )
                support_errors = validate_support(result, settings=DECODER_CONFIG)
                if errors or support_errors:
                    raise RuntimeError(f"invalid BL measured result: {errors + support_errors}")
                if result != baseline:
                    raise RuntimeError("BL result differs from warm-up")

            measurements.append(
                {
                    "fixture": fixture_name,
                    "source": path.relative_to(ROOT).as_posix(),
                    "fixture_sha256": digest,
                    "container": [container.W, container.H, container.D],
                    "box_count": len(boxes),
                    "box_order": "source",
                    "priority": priority.value,
                    "decoder_config": _decoder_config_dict(),
                    "warmup_count": 1,
                    "warmup_seconds": warmup_seconds,
                    "repeat_count": len(times),
                    "times_seconds": times,
                    "min_seconds": min(times),
                    "median_seconds": statistics.median(times),
                    "max_seconds": max(times),
                    "results_match_warmup": True,
                    "result": _summary(baseline),
                }
            )
    return measurements


def _best_summary(trial) -> dict[str, object]:
    return {
        "generation_index": trial.best_generation_index,
        "population_index": trial.best_population_index,
        "genes": list(trial.best_genes),
        "fitness": trial.best_fitness,
        "placed_count": trial.placed_count,
        "failed_count": trial.failed_count,
        "placed_volume": trial.placed_volume,
        "density": trial.density,
    }


def _measure_workers(budget: _MeasurementTimeBudget) -> list[dict[str, object]]:
    problems = (
        ("small", Container(4, 4, 4), tuple(Box(i, 1, 1, 1) for i in range(4))),
        (
            "larger",
            Container(12, 12, 12),
            tuple(Box(i, 1 + i % 3, 1 + (i // 3) % 3, 1 + (i // 9) % 3) for i in range(48)),
        ),
    )
    measurements: list[dict[str, object]] = []
    best_by_problem: dict[str, dict[str, object]] = {}

    for problem_name, container, boxes in problems:
        for workers in (1, 2, 4):
            for cache in (False, True):
                execution = GAExecutionConfig(workers=workers, cache=cache)
                kwargs = {
                    "decoder": ComparisonDecoder.PP,
                    "priority": BLPriority.BOTTOM_FIRST,
                    "config": GA_CONFIG,
                    "seed": 17,
                    "execution": execution,
                    "decoder_config": DECODER_CONFIG,
                    "origin": "benchmark",
                    "problem_source": f"synthetic:{problem_name}",
                }
                warmup, warmup_wall_seconds = budget.run(
                    f"workers {problem_name}/{workers}/cache={cache} warm-up",
                    lambda: run_comparison_trial(container, boxes, **kwargs),
                )
                expected_best = _best_summary(warmup)
                prior_best = best_by_problem.setdefault(problem_name, expected_best)
                if expected_best != prior_best:
                    raise RuntimeError(f"worker/cache result differs for {problem_name}")

                trials = []
                times: list[float] = []
                for repeat in range(3):
                    trial, comparison_call_wall_seconds = budget.run(
                        f"workers {problem_name}/{workers}/cache={cache} repeat {repeat + 1}",
                        lambda: run_comparison_trial(container, boxes, **kwargs),
                    )
                    best = _best_summary(trial)
                    if best != expected_best:
                        raise RuntimeError(
                            f"worker/cache repeat differs for {problem_name}, {workers=}, {cache=}, {repeat=}"
                        )
                    metadata = trial.run_metadata.to_dict()
                    times.append(trial.elapsed_seconds)
                    trials.append(
                        {
                            "repeat": repeat + 1,
                            "run_id": metadata["run_id"],
                            "elapsed_seconds": trial.elapsed_seconds,
                            "comparison_call_wall_seconds": comparison_call_wall_seconds,
                            "best": best,
                            "workers": metadata["execution"]["workers"],
                            "statistics": metadata["statistics"],
                            "start_method": metadata["execution"]["start_method"],
                        }
                    )

                measurements.append(
                    {
                        "problem": problem_name,
                        "container": [container.W, container.H, container.D],
                        "box_count": len(boxes),
                        "source": f"synthetic:{problem_name}",
                        "decoder": "PP",
                        "priority": "BottomFirst",
                        "order_mode": "source",
                        "ga_config": {
                            "population_size": GA_CONFIG.population_size,
                            "generation_count": GA_CONFIG.generation_count,
                            "crossover_rate": GA_CONFIG.crossover_rate,
                            "mutation_rate": GA_CONFIG.mutation_rate,
                        },
                        "seed": 17,
                    "decoder_config": _decoder_config_dict(),
                        "requested_workers": workers,
                        "cache": cache,
                        "warmup_seconds": warmup.elapsed_seconds,
                        "warmup_comparison_call_wall_seconds": warmup_wall_seconds,
                        "warmup_workers": warmup.run_metadata.to_dict()["execution"]["workers"],
                        "repeat_count": len(times),
                        "times_seconds": times,
                        "min_seconds": min(times),
                        "median_seconds": statistics.median(times),
                        "max_seconds": max(times),
                        "best_results_match": True,
                        "trials": trials,
                    }
                )
                print(
                    f"workers: {problem_name}, {workers=}, {cache=}, "
                    f"median={statistics.median(times):.6f}s",
                    flush=True,
                )
    return measurements


def _markdown(report: dict[str, object], json_path: Path) -> str:
    budget = report["measurement_contract"]["time_budget_seconds"]
    lines = [
        "# packing performance measurements",
        "",
        f"- Measured source commit: `{report['source_commit']}`",
        f"- Measurement harness SHA-256: `{report['measurement_harness_sha256']}`",
        f"- Captured: {report['captured_at_local']}",
        f"- Environment: {report['environment']['os']}; {report['environment']['python']}; CPU `{report['environment']['cpu']}`; logical CPUs `{report['environment']['logical_cpu_count']}`",
        f"- Pre-run wall-clock budget: {budget} seconds; the budget clock starts after argument validation and runs through the final pre-write check. Elapsed time at that check: {report['measurement_contract']['measurement_duration_seconds_before_artifact_write']:.3f} seconds. Artifact file-writing time is excluded.",
        "- The budget is checked before and after every warm-up/timed execution; an in-flight call may finish, but expiration aborts the run before either acceptance artifact is written.",
        "- Abort without acceptance artifacts on time-budget expiration, result mismatch, or validation failure. No interrupted run is converted into a packing failure.",
        "- BL timing covers the complete `pack` call, excluding input loading and result validation. Each condition has one warm-up followed by three timed repeats.",
        "",
        "## BL single-call measurements",
        "",
        "Cartesian/version 1/support 55%, source order; time covers the complete `pack` call. Each measured result passed geometry and independent support validation and matched the warm-up result.",
        "",
        "| Fixture | Priority | SHA-256 | Warm-up s | Repeats s | Min s | Median s | Max s | Placed / failed / volume / density |",
        "|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in report["bl_measurements"]:
        result = row["result"]
        lines.append(
            f"| {row['fixture']} | {row['priority']} | `{row['fixture_sha256']}` | "
            f"{row['warmup_seconds']:.6f} | {', '.join(f'{value:.6f}' for value in row['times_seconds'])} | "
            f"{row['min_seconds']:.6f} | {row['median_seconds']:.6f} | {row['max_seconds']:.6f} | "
            f"{result['placed_count']} / {result['failed_count']} / {result['placed_volume']} / {result['density']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Workers/cache GA-call measurements",
            "",
            "PP-GA; Manual worker requests 1/2/4; cache OFF/ON; seed 17; P=4/G=3; BottomFirst/source; Cartesian/55%. The reported `elapsed_seconds` is the GA core interval from `begin_run`'s `perf_counter` start to `RunContext.complete`, after `EvaluationRuntime` shutdown. It includes environment/worker resolution and pool startup, transfer, evaluation, and shutdown; it excludes comparison-wrapper result validation and summary construction. `comparison_call_wall_seconds` records the outer comparison-call duration separately. Resolution and requested/computed/cache-hit counts are preserved per run in the JSON artifact.",
            "",
            "| Problem | Workers request | Cache | Warm-up GA core s | Repeats GA core s | Min GA core s | Median GA core s | Max GA core s | Resolved / requested / computed / cache hits |",
            "|---|---:|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in report["workers_measurements"]:
        last = row["trials"][-1]
        stats = last["statistics"]
        resolved = last["workers"]["resolved_workers"]
        lines.append(
            f"| {row['problem']} ({row['box_count']} boxes) | {row['requested_workers']} | "
            f"{'ON' if row['cache'] else 'OFF'} | {row['warmup_seconds']:.6f} | "
            f"{', '.join(f'{value:.6f}' for value in row['times_seconds'])} | "
            f"{row['min_seconds']:.6f} | {row['median_seconds']:.6f} | {row['max_seconds']:.6f} | "
            f"{resolved} / {stats['requested']} / {stats['computed']} / {stats['cache_hits']} |"
        )
    lines.extend(
        [
            "",
            "All three timed best summaries per workers/cache condition matched that condition's warm-up and the other conditions for the same problem. These observations describe only this commit and environment; they do not claim a general speedup, optimal workers value, Windows response-time guarantee, or absolute threshold.",
            "",
            f"Machine-readable run metadata and individual trials: [`{json_path.name}`]({json_path.name}).",
            "",
        ]
    )
    return "\n".join(lines)


def _output_paths(output_json: Path) -> tuple[Path, Path]:
    if output_json.suffix.lower() != ".json":
        raise ValueError("--output-json must use the .json extension")
    output_json = output_json.resolve()
    output_markdown = output_json.with_suffix(".md")
    if not output_json.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output_json.parent}")
    if output_json.exists() or output_markdown.exists():
        existing = output_json if output_json.exists() else output_markdown
        raise FileExistsError(f"refusing to overwrite existing acceptance artifact: {existing}")
    return output_json, output_markdown


def _write_artifacts(report: dict[str, object], output_json: Path) -> tuple[Path, Path]:
    output_json, output_markdown = _output_paths(output_json)

    created: list[Path] = []
    try:
        with output_json.open("x", encoding="utf-8", newline="\n") as json_file:
            created.append(output_json)
            with output_markdown.open("x", encoding="utf-8", newline="\n") as markdown_file:
                created.append(output_markdown)
                json_file.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                markdown_file.write(_markdown(report, output_json))
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return output_json, output_markdown


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="new JSON artifact path; a matching Markdown file is written beside it",
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=int,
        required=True,
        help="pre-run total wall-clock budget; expiration aborts before artifact creation",
    )
    args = parser.parse_args(argv)
    if args.time_budget_seconds <= 0:
        parser.error("--time-budget-seconds must be a positive integer")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_json, _ = _output_paths(args.output_json)
    budget = _MeasurementTimeBudget(args.time_budget_seconds)
    harness_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report: dict[str, object] = {
        "format": "packing3d-performance-measurements",
        "version": 1,
        "source_commit": _source_commit(),
        "captured_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": {
            "os": platform.platform(),
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "cpu": _cpu_name(),
            "logical_cpu_count": os.cpu_count(),
        },
        "measurement_contract": {
            "warmup_runs_per_condition": 1,
            "timed_runs_per_condition": 3,
            "time_budget_seconds": args.time_budget_seconds,
            "time_budget_policy": "wall-clock from immediately after argument validation to the final pre-write check; checked before and after each warm-up/timed execution; the active call may finish, but expiration aborts without writing acceptance artifacts; artifact file-writing time is excluded",
            "abort_conditions": "time-budget expiration, warm-up/repeat result mismatch, or geometry/support validation failure; failures abort without emitting acceptance artifacts and are never represented as a packing failure",
            "timed_scope_bl": "complete pack call, excluding input loading and result validation",
            "timed_scope_workers": "RunMetadata elapsed_seconds from begin_run perf_counter start to RunContext.complete after EvaluationRuntime shutdown; includes environment/worker resolution, pool startup, transfer, evaluation, and shutdown; excludes comparison-wrapper validation and summary construction",
            "workers_outer_call_timing": "comparison_call_wall_seconds measures the complete run_comparison_trial wall-clock call separately",
        },
        "measurement_harness_sha256": harness_sha256,
        "bl_measurements": _measure_bl(budget),
        "workers_measurements": _measure_workers(budget),
    }
    budget.checkpoint("before writing final artifacts")
    report["measurement_contract"]["measurement_duration_seconds_before_artifact_write"] = budget.elapsed()
    output_json, output_markdown = _write_artifacts(report, output_json)
    print(f"wrote {output_json}", flush=True)
    print(f"wrote {output_markdown}", flush=True)


if __name__ == "__main__":
    main()
