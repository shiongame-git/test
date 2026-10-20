"""Measure a few 200-box Research Record runs; not a performance acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

from packing import (
    BLPriority,
    DecoderConfig,
    GAConfig,
    RunOptions,
    WorkersRequest,
    parse_box_sizes_bytes,
    prepare_research_experiment,
    read_research_record,
    run_research_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (("BL-GA", "Cartesian"), ("BL-GA", "LocalNeighbors"),
              ("PP-GA", "Cartesian"), ("PP-GA", "LocalNeighbors"))


def _output_paths(json_path: Path, markdown_path: Path) -> tuple[Path, Path]:
    if json_path.suffix.lower() != ".json" or markdown_path.suffix.lower() != ".md":
        raise ValueError("output paths must end in .json and .md")
    json_path = json_path.resolve()
    markdown_path = markdown_path.resolve()
    if os.path.normcase(str(json_path)) == os.path.normcase(str(markdown_path)):
        raise ValueError("JSON and Markdown output paths must differ")
    if not json_path.parent.is_dir() or not markdown_path.parent.is_dir():
        raise FileNotFoundError("output parent directories must already exist")
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("refusing to overwrite an existing measurement artifact")
    return json_path, markdown_path


def _git_state() -> dict[str, object]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                                    check=True, capture_output=True, text=True, timeout=5).stdout)
        return {"head": head, "dirty_before_measurement": dirty}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"head": None, "dirty_before_measurement": None,
                "unavailable": f"{type(exc).__name__}: {exc}"}


def _measure(source: Path, record_type: str, policy: str, work_root: Path) -> dict[str, object]:
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    container, boxes = parse_box_sizes_bytes(source_bytes, source=str(source))
    if len(boxes) != 200:
        raise ValueError(f"expected a fixed 200-box input: {source}")
    options = RunOptions(
        origin="api", source_boxes=boxes, source=str(source), fixture_sha256=source_hash,
    )
    tracemalloc.start()
    started = time.perf_counter()
    try:
        plan = prepare_research_experiment(
            container, boxes, record_type=record_type,
            config=GAConfig(population_size=2, generation_count=2, crossover_rate=0.8, mutation_rate=0.1),
            priority=BLPriority.BOTTOM_FIRST,
            decoder_config=DecoderConfig(candidate_policy=policy, min_support_percent=55),
            start_seed=17, run_count=1, workers_request=WorkersRequest("manual", 1),
            cache=False, save_level="full", profile=False, run_options=options, source_bytes=source_bytes,
        )
        experiment = run_research_experiment(plan, work_root)
        total_seconds = time.perf_counter() - started
        _current, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    experiment_directory = work_root / experiment["experiment_id"]
    record_directory = experiment_directory / "runs" / "000"
    record = read_research_record(record_directory)
    artifacts = {item["role"]: item for item in record["artifacts"]}
    output_bytes = sum(path.stat().st_size for path in experiment_directory.rglob("*") if path.is_file())
    return {
        "record_type": record_type,
        "candidate_policy": policy,
        "support_percent": 55,
        "priority": "BottomFirst",
        "seed": 17,
        "population_size": 2,
        "generation_count": 2,
        "workers": 1,
        "cache": False,
        "profile": False,
        "save_level": "full",
        "box_count": len(boxes),
        "fixture": str(source.relative_to(ROOT)).replace(os.sep, "/"),
        "fixture_sha256": source_hash,
        "run_id": record["ga_run_id"],
        "ga_core_seconds": record["timing"]["execution_seconds"],
        "ga_wrapper_call_seconds": record["timing"]["call_wall_seconds"],
        "independent_validation_seconds": record["timing"]["validation_seconds"],
        "artifact_save_seconds": record["timing"]["save_seconds"],
        "experiment_total_seconds": total_seconds,
        "tracemalloc_peak_bytes": peak_bytes,
        "output_bytes": output_bytes,
        "validation_scope": record["validation"]["scope"],
        "checked_evaluations": record["validation"]["checked_evaluations"],
        "record_artifact_bytes": {role: item["bytes"] for role, item in artifacts.items()},
    }


def _markdown(report: dict[str, object], json_path: Path, markdown_path: Path) -> str:
    rows = report["measurements"]
    lines = [
        "# Research Record 200箱少数run計測",
        "",
        f"計測日: {report['measured_at']}。対象commit: `{report['source']['head']}`。",
        "200箱の固定入力を使い、BL-GA/PP-GA × Cartesian/LocalNeighborsを各1 run計測。",
        "P=2、G=2、seed=17、BottomFirst、支持55%、workers=1、cache/Profile OFF、Full保存。",
        "GA core、wrapper call、独立検査、artifact保存、Experiment全体、Python tracemalloc peak、保存bytesを分けて記録。",
        "tracemallocはPython割当のみを観測し、native/process RSSのpeakではない。1回ずつの観測で一般性能・speedupは主張しない。",
        "",
        "|decoder|policy|箱数|GA core s|call s|検査 s|保存 s|総 s|peak bytes|出力 bytes|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['record_type']}|{row['candidate_policy']}|{row['box_count']}|"
            f"{row['ga_core_seconds']:.6f}|{row['ga_wrapper_call_seconds']:.6f}|"
            f"{row['independent_validation_seconds']:.6f}|{row['artifact_save_seconds']:.6f}|"
            f"{row['experiment_total_seconds']:.6f}|{row['tracemalloc_peak_bytes']}|{row['output_bytes']}|"
        )
    relative = os.path.relpath(json_path, markdown_path.parent).replace(os.sep, "/")
    lines.extend(["", f"条件、fixture SHA-256、run IDs、検査件数、artifact別bytesは[`{json_path.name}`]({relative})。", ""])
    return "\n".join(lines)


def _write_exclusive(json_path: Path, markdown_path: Path, report: dict[str, object]) -> None:
    json_data = (json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    markdown_data = _markdown(report, json_path, markdown_path).encode("utf-8")
    created: list[Path] = []
    try:
        for path, data in ((json_path, json_data), (markdown_path, markdown_data)):
            with path.open("xb") as stream:
                created.append(path)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args()
    json_path, markdown_path = _output_paths(args.output_json, args.output_markdown)
    if not _git_state()["head"]:
        raise RuntimeError("could not identify source commit")
    measurements = []
    with tempfile.TemporaryDirectory(prefix="packing3d-research-measure-") as temporary:
        work_root = Path(temporary)
        for record_type, policy in CONDITIONS:
            source = ROOT / "data/box_sizes_cube.txt"
            measurements.append(_measure(source, record_type, policy, work_root))
    report = {
        "format": "packing3d-research-measurement",
        "version": 1,
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": _git_state(),
        "environment": {"os": platform.platform(), "python": platform.python_version(),
                        "implementation": platform.python_implementation(), "processor": platform.processor()},
        "measurement_repetitions": 1,
        "measurements": measurements,
        "limits": ["four small-run observations only", "tracemalloc excludes native and child-process RSS",
                   "not a speedup or optimal-worker claim"],
    }
    _write_exclusive(json_path, markdown_path, report)
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
