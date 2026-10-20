"""Benchmark the fixed 200-box fixtures for each BL candidate priority."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import statistics
import time
from pathlib import Path

from packing.source import source_revision

from packing import BLPriority, DecoderConfig, pack, read_box_sizes, validate_result

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "box_sizes_cube.txt": "b57a8c73bc7b66b9807e55d58d17a23fa9294ce652c98597642ea0ed889d906a",
    "box_sizes_cuboid.txt": "908bb73f607d377e76b8e50c92f04a3c029e05cc3eee57e9b988c36352f3da9f",
}


def _cpu_name() -> str:
    name = platform.processor().strip()
    if name:
        return name
    return os.environ.get("PROCESSOR_IDENTIFIER", "unknown")


def run(path: Path, priority: BLPriority) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = EXPECTED.get(path.name)
    if expected is not None and digest != expected:
        raise SystemExit(f"fixture hash mismatch for {path}: {digest}")
    container, boxes = read_box_sizes(path)

    # Warm-up is intentionally outside the reported measurements.
    warm = pack(container, boxes, priority=priority)
    errors = validate_result(warm, expected_container=container, ordered_boxes=boxes)
    if errors:
        raise SystemExit("warm-up result validation failed: " + "; ".join(errors))

    times: list[float] = []
    baseline = warm
    for _ in range(3):
        start = time.perf_counter()
        result = pack(container, boxes, priority=priority)
        elapsed = time.perf_counter() - start
        errors = validate_result(result, expected_container=container, ordered_boxes=boxes)
        if errors:
            raise SystemExit("result validation failed: " + "; ".join(errors))
        if result != baseline:
            raise SystemExit("non-deterministic result: repeated run differs from warm-up")
        times.append(elapsed)

    print(f"fixture: {path}")
    print(f"sha256: {digest}")
    print(f"container: {container.W}x{container.H}x{container.D}")
    print(f"boxes: {len(boxes)}")
    print(f"priority: {priority.value}")
    print(f"git_revision: {source_revision() or 'unavailable'}")
    decoder_config = DecoderConfig()
    print(
        "effective_decoder_config: "
        f"candidate_policy={decoder_config.candidate_policy}, "
        f"candidate_policy_version={decoder_config.candidate_policy_version}, "
        f"min_support_percent={decoder_config.min_support_percent}, "
        f"support_model_version={decoder_config.support_model_version}"
    )
    print(
        "measurement_boundary: warm-up excluded; each of 3 timed intervals covers "
        "pack() only; fixture loading and result validation are outside the timed intervals"
    )
    print(
        "validation_scope: validate_result checks expected boxes/order, container bounds, "
        "pairwise overlap, and result aggregates; support-threshold validation is not performed"
    )
    print(f"python: {platform.python_implementation()} {platform.python_version()}")
    print(f"cpu: {_cpu_name()}")
    print(f"os: {platform.platform()}")
    print("times_s: " + ", ".join(f"{t:.6f}" for t in times))
    print(f"median_s: {statistics.median(times):.6f}")
    print(f"placed_count: {baseline.placed_count}")
    print(f"failed_count: {baseline.failed_count}")
    print(f"placed_volume: {baseline.placed_volume}")
    print(f"density: {baseline.density:.9f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", nargs="*", type=Path)
    parser.add_argument(
        "--priority",
        choices=("both", *(priority.value for priority in BLPriority)),
        default="both",
        help="candidate priority to benchmark (default: both)",
    )
    args = parser.parse_args()
    paths = args.fixtures or [ROOT / "data" / name for name in EXPECTED]
    priorities = tuple(BLPriority) if args.priority == "both" else (BLPriority(args.priority),)

    first = True
    for path in paths:
        for priority in priorities:
            if not first:
                print()
            first = False
            run(path, priority)


if __name__ == "__main__":
    main()
