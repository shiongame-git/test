"""Research Record conversion, independent validation, and durable artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from ._decoder_config import DecoderConfig
from .bl import BLPriority
from .errors import InputFormatError
from .ga import GARunResult
from .io_profile import read_decoder_profile_json
from .io_result_json import _render_result_json, read_result_json
from .io_run_metadata import RunMetadata, validate_run_metadata
from .model import Box, Container, PackingResult, Placement, Point
from .pp_ga import PPGARunResult
from .pp_state import PPState
from .profile import DecoderProfileRecord
from .research import (
    ResearchRecord,
    _loads_strict_json,
    read_generations_jsonl,
    validate_research_record,
)
from .source import SOURCE_LABEL
from .validate import validate_result, validate_support
from .workers import observe_environment


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validate_source_bytes(source_bytes: bytes, problem: dict[str, Any]) -> None:
    from .io_box_sizes import parse_box_sizes_bytes

    container, boxes = parse_box_sizes_bytes(source_bytes, source=problem["source"])
    expected_container = tuple(problem["container"])
    expected_boxes = tuple(tuple(row) for row in problem["source_boxes"])
    if (container.W, container.H, container.D) != expected_container or tuple(
        (box.id, box.w, box.h, box.d) for box in boxes
    ) != expected_boxes:
        raise ValueError("captured source bytes do not describe the snapshot problem")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def _source_repository() -> Path | None:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    return None


def default_research_directory() -> Path:
    documents = Path.home() / "Documents"
    return (documents if documents.is_dir() else Path.home()) / "Packing3D-Experiments"


def validate_research_output_root(path: str | os.PathLike[str]) -> Path:
    """Allow external output roots, or exactly the ignored repository-local root."""
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("research output root must be a path")
    root = Path(path).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("research output root must be a directory")
    repository = _source_repository()
    if repository is None:
        return root
    repository = repository.resolve()
    if root == repository / "local-experiments":
        return root
    if root == repository or root.is_relative_to(repository):
        raise ValueError("research output must be outside the repository or exactly local-experiments")
    return root


def _git_capture() -> tuple[str | None, str, bool, list[str], Path | None]:
    root = _source_repository()
    unavailable: list[str] = []
    if root is None:
        return None, "unavailable", False, ["git checkout not found"], None
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True, timeout=3).stdout.strip()
        state_text = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
                                    check=True, capture_output=True, text=True, timeout=3).stdout
        git_state = "clean" if not state_text else "dirty"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, "unavailable", False, [f"git state unavailable: {type(exc).__name__}"], root
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        return None, "unavailable", False, ["git HEAD is not a full lowercase commit hash"], root
    try:
        loaded = Path(sys.modules["packing"].__file__).resolve()
        expected = (root / "src" / "packing" / "__init__.py").resolve()
        source_diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src/packing"], cwd=root,
                                     timeout=3).returncode
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", "src/packing"],
                                   cwd=root, check=True, capture_output=True, text=True, timeout=3).stdout
        verified = loaded == expected and source_diff == 0 and not untracked
    except (KeyError, OSError, subprocess.SubprocessError):
        verified = False
    if git_state == "dirty":
        unavailable.append("working tree is dirty; changed files are not archived in this Record")
    if not verified:
        unavailable.append("loaded packing source could not be verified against tracked HEAD files")
    return head, git_state, verified, unavailable, root


def make_provenance(environment: dict[str, Any] | None = None) -> dict[str, Any]:
    """Capture environment and code identity at execution time, not save time."""
    if environment is None:
        environment = observe_environment(implementation_version=SOURCE_LABEL)
    head, state, verified, unavailable, root = _git_capture()
    lock_hash = None
    if root is not None:
        lockfile = root / "uv.lock"
        try:
            lock_hash = _sha256(lockfile.read_bytes())
        except OSError as exc:
            unavailable.append(f"uv.lock unavailable: {type(exc).__name__}")
    else:
        unavailable.append("repository root unavailable for uv.lock hash")
    if lock_hash is None:
        unavailable.append("uv.lock SHA-256 unavailable")
    limits = list(unavailable)
    limits.append("source identity does not archive code; exact reproduction needs the matching checkout and environment")
    return {
        "distribution_version": SOURCE_LABEL,
        "git_head": head,
        "git_state": state,
        "loaded_code_verified": verified,
        "python": dict(environment["python"]),
        "os": dict(environment["os"]),
        "cpu": dict(environment["cpu"]),
        "lockfile_sha256": lock_hash,
        "unavailable": unavailable,
        "reproducibility_limits": limits,
    }


def snapshot_from_ga(run: GARunResult | PPGARunResult) -> dict[str, Any]:
    if type(run) not in (GARunResult, PPGARunResult):
        raise TypeError("run must be a GARunResult or PPGARunResult")
    if type(run.run_metadata) is not RunMetadata:
        raise ValueError("research GA Record requires completed run metadata")
    meta = run.run_metadata.to_dict()
    validate_run_metadata(meta)
    d = meta["decoder"]
    execution = meta["execution"]
    workers = execution["workers"]
    return {
        "problem": meta["problem"],
        "decoder": {
            "name": d["name"], "priority": d["priority"], "reference_version": d["reference_version"],
            "fitness_version": d["fitness_version"], "pp_initialization": d["pp_initialization"],
            "candidate_policy": d["candidate_policy"], "candidate_policy_version": d["candidate_policy_version"],
            "min_support_percent": d["min_support_percent"], "support_model_version": d["support_model_version"],
        },
        "ga": {key: meta["ga"][key] for key in ("population_size", "generation_count", "crossover_rate", "mutation_rate")},
        "execution": {
            "workers_request": {"mode": workers["requested_mode"], "manual_workers": workers["manual_workers"]},
            "workers": workers["resolved_workers"], "cache": execution["cache"], "start_method": execution["start_method"],
        },
        "profile": run.profile is not None,
        "pp_input_state": None,
    }


def snapshot_for_single(
    container: Container,
    boxes: tuple[Box, ...],
    *,
    record_type: str,
    priority: BLPriority,
    decoder_config: DecoderConfig,
    pp_input_state: PPState | None = None,
    source: str | None = None,
    fixture_sha256: str | None = None,
    order_mode: str = "source",
    source_boxes: tuple[Box, ...] | None = None,
) -> dict[str, Any]:
    if record_type not in ("BL", "PP"):
        raise ValueError("single record_type must be BL or PP")
    if type(container) is not Container or type(boxes) is not tuple or any(type(b) is not Box for b in boxes):
        raise TypeError("single snapshot needs a Container and tuple[Box]")
    if type(priority) is not BLPriority or type(decoder_config) is not DecoderConfig:
        raise TypeError("priority and decoder_config must be resolved core values")
    if order_mode not in ("source", "depth_desc_stable"):
        raise ValueError("unknown order_mode")
    source_boxes = boxes if source_boxes is None else source_boxes
    if type(source_boxes) is not tuple or any(type(box) is not Box for box in source_boxes):
        raise TypeError("source_boxes must be a tuple[Box]")
    source_by_id = {box.id: box for box in source_boxes}
    if len(source_by_id) != len(source_boxes) or set(source_by_id) != {box.id for box in boxes}:
        raise ValueError("source_boxes must contain the same unique problem IDs as boxes")
    if any(source_by_id[box.id] != box for box in boxes):
        raise ValueError("source_boxes dimensions differ from the executed boxes")
    effective = source_boxes if order_mode == "source" else tuple(sorted(source_boxes, key=lambda box: -box.d))
    if tuple(box.id for box in effective) != tuple(box.id for box in boxes):
        raise ValueError("boxes order differs from the requested order_mode")
    problem = {
        "container": [container.W, container.H, container.D],
        "source_boxes": [[b.id, b.w, b.h, b.d] for b in source_boxes],
        "source_box_order": [b.id for b in source_boxes],
        "effective_box_order": [b.id for b in effective],
        "order_mode": order_mode,
        "source": source,
        "fixture_sha256": fixture_sha256,
    }
    name = "PP" if record_type == "PP" else "BL"
    state = None
    if record_type == "PP":
        if type(pp_input_state) is not PPState:
            raise TypeError("single PP record requires PPState")
        state = [[box_id, None if point is None else [point.x, point.y, point.z]] for box_id, point in pp_input_state.items_tuple()]
    elif pp_input_state is not None:
        raise ValueError("BL record does not accept a PPState")
    return {
        "problem": problem,
        "decoder": {
            "name": name,
            "priority": priority.value,
            "reference_version": f"{name}-ref-v2",
            "fitness_version": None,
            "pp_initialization": "explicit_input_state" if name == "PP" else None,
            "candidate_policy": decoder_config.candidate_policy,
            "candidate_policy_version": decoder_config.candidate_policy_version,
            "min_support_percent": decoder_config.min_support_percent,
            "support_model_version": decoder_config.support_model_version,
        },
        "ga": None,
        "execution": None,
        "profile": False,
        "pp_input_state": state,
    }


@dataclass(frozen=True, slots=True, init=False)
class ResearchCapture:
    """Immutable execution capture; nested JSON values are returned as copies."""

    _snapshot_json: str
    _provenance_json: str
    execution_started_at: str
    execution_finished_at: str
    call_wall_seconds: float
    execution_seconds: float | None
    source_bytes: bytes | None

    def __init__(self, snapshot, provenance, execution_started_at, execution_finished_at,
                 call_wall_seconds, execution_seconds, source_bytes=None):
        if type(snapshot) is not dict or type(provenance) is not dict:
            raise TypeError("capture snapshot and provenance must be JSON objects")
        if source_bytes is not None and type(source_bytes) is not bytes:
            raise TypeError("capture source_bytes must be bytes or None")
        object.__setattr__(self, "_snapshot_json", json.dumps(snapshot, ensure_ascii=False, allow_nan=False))
        object.__setattr__(self, "_provenance_json", json.dumps(provenance, ensure_ascii=False, allow_nan=False))
        object.__setattr__(self, "execution_started_at", execution_started_at)
        object.__setattr__(self, "execution_finished_at", execution_finished_at)
        object.__setattr__(self, "call_wall_seconds", float(call_wall_seconds))
        object.__setattr__(self, "execution_seconds", None if execution_seconds is None else float(execution_seconds))
        object.__setattr__(self, "source_bytes", source_bytes)

    @property
    def snapshot(self) -> dict[str, Any]:
        return json.loads(self._snapshot_json)

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(self._provenance_json)


def capture_ga_run(
    run: GARunResult | PPGARunResult,
    *,
    execution_started_at: str,
    execution_finished_at: str,
    call_wall_seconds: float,
    source_bytes: bytes | None = None,
    execution_provenance: dict[str, Any] | None = None,
) -> ResearchCapture:
    if type(run.run_metadata) is not RunMetadata:
        raise ValueError("GA run metadata is unavailable")
    environment = run.run_metadata["environment"]
    provenance = make_provenance(environment) if execution_provenance is None else json.loads(
        json.dumps(execution_provenance, ensure_ascii=False)
    )
    provenance["distribution_version"] = environment["implementation"]["version"]
    provenance["python"] = dict(environment["python"])
    provenance["os"] = dict(environment["os"])
    provenance["cpu"] = dict(environment["cpu"])
    return ResearchCapture(
        snapshot_from_ga(run), provenance, execution_started_at,
        execution_finished_at, float(call_wall_seconds), float(run.run_metadata["statistics"]["elapsed_seconds"]),
        source_bytes,
    )


def capture_single_run(
    *,
    snapshot: dict[str, Any],
    execution_started_at: str,
    execution_finished_at: str,
    call_wall_seconds: float,
    environment: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    source_bytes: bytes | None = None,
) -> ResearchCapture:
    return ResearchCapture(snapshot, make_provenance(environment) if provenance is None else provenance, execution_started_at,
                           execution_finished_at, float(call_wall_seconds), float(call_wall_seconds), source_bytes)


def _state_data(state: PPState | None) -> list[list[Any]] | None:
    if state is None:
        return None
    return [[box_id, None if point is None else [point.x, point.y, point.z]] for box_id, point in state.items_tuple()]


def _result_data(result: PackingResult) -> dict[str, Any]:
    return json.loads(_render_result_json(result))


def _run_generations(run: GARunResult | PPGARunResult):
    for generation in run.generations:
        yield generation


def _evaluation_state(generation: Any, population_index: int) -> PPState | None:
    input_states = getattr(generation, "input_states", None)
    return None if input_states is None else input_states[population_index]


def _generation_decoder(run: GARunResult | PPGARunResult, generation_index: int) -> str:
    if type(run) is GARunResult:
        return "BL"
    return "BL" if generation_index == 0 else "PP"


def _build_summary(run: GARunResult | PPGARunResult) -> dict[str, Any]:
    best = run.best
    result = best.packing_result
    best_data = {
        "generation_index": run.best_generation_index,
        "population_index": run.best_population_index,
        "genes": list(best.individual.genes),
        "fitness": best.fitness,
        "placed_count": result.placed_count,
        "failed_count": result.failed_count,
        "placed_volume": result.placed_volume,
        "density": result.density,
    }
    generations = []
    cumulative = None
    for generation in _run_generations(run):
        fitnesses = [evaluation.fitness for evaluation in generation.evaluations]
        gen_best_index = generation.best_index
        gen_best = generation.evaluations[gen_best_index]
        if cumulative is None or gen_best.fitness > cumulative["fitness"]:
            cumulative = {"generation_index": generation.generation_index,
                          "population_index": gen_best_index, "fitness": gen_best.fitness}
        generations.append({
            "generation_index": generation.generation_index,
            "decoder": _generation_decoder(run, generation.generation_index),
            "count": len(fitnesses),
            "min": min(fitnesses),
            "max": max(fitnesses),
            "mean": math.fsum(fitnesses) / len(fitnesses),
            "best_population_index": gen_best_index,
            "best_fitness": gen_best.fitness,
            "cumulative_best": dict(cumulative),
        })
    return {
        "best": best_data,
        "generations": generations,
        "evaluation_stats": {
            "requested": run.execution_stats.requested,
            "computed": run.execution_stats.computed,
            "cache_hits": run.execution_stats.cache_hits,
        },
    }


def _build_trace(run: GARunResult | PPGARunResult, record_id: str) -> bytes:
    metadata = run.run_metadata
    assert type(metadata) is RunMetadata
    header = {
        "format": "packing3d-ga-state-trace", "version": 1, "record_id": record_id,
        "ga_run_id": metadata["run_id"], "generation_count": run.config.generation_count,
        "population_size": run.config.population_size, "box_count": len(run.boxes),
    }
    lines = [json.dumps(header, ensure_ascii=False, allow_nan=False)]
    for generation in _run_generations(run):
        for pop_index, evaluation in enumerate(generation.evaluations):
            result = evaluation.packing_result
            row = {
                "generation_index": generation.generation_index,
                "population_index": pop_index,
                "genes": list(evaluation.individual.genes),
                "fitness": evaluation.fitness,
                "decoder": _generation_decoder(run, generation.generation_index),
                "packing_result": _result_data(result),
                "pp_input_state": _state_data(_evaluation_state(generation, pop_index)),
                "placed_count": result.placed_count,
                "failed_count": result.failed_count,
                "placed_volume": result.placed_volume,
                "density": result.density,
            }
            lines.append(json.dumps(row, ensure_ascii=False, allow_nan=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _ordered_boxes(snapshot: dict[str, Any], genes: list[int] | tuple[int, ...] | None = None) -> tuple[Box, ...]:
    p = snapshot["problem"]
    by_id = {row[0]: Box(*row) for row in p["source_boxes"]}
    order = genes if genes is not None else p["effective_box_order"]
    return tuple(by_id[box_id] for box_id in order)


def _validate_result(result: PackingResult, snapshot: dict[str, Any], genes=None) -> None:
    expected_container = Container(*snapshot["problem"]["container"])
    expected_boxes = _ordered_boxes(snapshot, genes)
    geometry = validate_result(result, expected_container=expected_container, ordered_boxes=expected_boxes)
    if geometry:
        raise ValueError("independent result validation failed: " + "; ".join(geometry))
    decoder = snapshot["decoder"]
    settings = DecoderConfig(
        candidate_policy=decoder["candidate_policy"],
        candidate_policy_version=decoder["candidate_policy_version"],
        min_support_percent=decoder["min_support_percent"],
        support_model_version=decoder["support_model_version"],
    )
    support = validate_support(result, settings=settings)
    if support:
        raise ValueError("independent support validation failed: " + "; ".join(support))


def _validate_single_result(result: PackingResult, snapshot: dict[str, Any]) -> dict[str, Any]:
    _validate_result(result, snapshot)
    return {"placed_count": result.placed_count, "failed_count": result.failed_count,
            "placed_volume": result.placed_volume, "density": result.density}


def _validate_ga_run(run: GARunResult | PPGARunResult, snapshot: dict[str, Any], full: bool) -> tuple[dict[str, Any], int]:
    if type(run.run_metadata) is not RunMetadata:
        raise ValueError("GA result has no completed metadata")
    metadata = run.run_metadata.to_dict()
    validate_run_metadata(metadata)
    if snapshot_from_ga(run) != snapshot:
        raise ValueError("GA run no longer matches its captured snapshot")
    evaluation_count = 0
    if full:
        for generation in _run_generations(run):
            for evaluation in generation.evaluations:
                _validate_result(evaluation.packing_result, snapshot, evaluation.individual.genes)
                evaluation_count += 1
    else:
        _validate_result(run.best.packing_result, snapshot, run.best.individual.genes)
        evaluation_count = 1
    return _build_summary(run), evaluation_count


def _profile_bytes(profile: DecoderProfileRecord) -> bytes:
    return _json_bytes(profile.to_dict())


def _artifact(role: str, path: str, fmt: str, version: int | None, raw: bytes) -> dict[str, Any]:
    return {"role": role, "path": path, "format": fmt, "version": version,
            "bytes": len(raw), "sha256": _sha256(raw)}


def _safe_child(root: Path, relative: str) -> Path:
    root = root.resolve(strict=True)
    parts = Path(relative.replace("\\", "/")).parts
    if Path(relative).is_absolute() or any(part in ("..", "") for part in parts):
        raise ValueError("unsafe relative artifact path")
    target = root.joinpath(*parts)
    current = root
    for part in parts:
        current = current / part
        if current.exists() and (current.is_symlink() or getattr(current.stat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("symlink/reparse artifact path is not allowed")
    if not target.resolve(strict=False).is_relative_to(root):
        raise ValueError("artifact path escapes save root")
    return target


def _atomic_bytes(path: Path, raw: bytes, *, replace: bool = False) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix="." + path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and path.exists():
            raise FileExistsError(path)
        if replace:
            os.replace(temporary, path)
        else:
            # Hard-link publication is exclusive on both Windows and POSIX.
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _write_record_atomic(path: Path, record_data: dict[str, Any]) -> None:
    raw = _json_bytes(record_data)
    validate_research_record(record_data)
    _atomic_bytes(path, raw, replace=path.exists())


def _record_base(
    record_id: str,
    record_type: str,
    snapshot: dict[str, Any],
    capture: ResearchCapture,
    *,
    experiment_id: str | None,
    run_index: int | None,
    seed: int | None,
) -> dict[str, Any]:
    return {
        "format": "packing3d-research-record", "version": 1, "record_id": record_id,
        "record_type": record_type, "experiment_id": experiment_id, "run_index": run_index,
        "ga_run_id": None, "status": "running", "seed": seed, "save_level": "summary",
        "snapshot": snapshot, "summary": None,
        "validation": {
            "scope": "single_result" if record_type in ("BL", "PP") else "best",
            "expected_evaluations": 1, "checked_evaluations": 0,
            "geometry_support": "not_run", "source_run_match": "not_applicable", "issues": [],
        },
        "provenance": capture.provenance,
        "timing": {
            "execution_started_at": capture.execution_started_at, "execution_finished_at": capture.execution_finished_at,
            "recorded_at": utc_now(), "call_wall_seconds": capture.call_wall_seconds,
            "execution_seconds": capture.execution_seconds,
            "execution_scope": {"BL": "pack_call", "PP": "pack_pp_call", "BL-GA": "ga_call_including_resolution_and_pool",
                                "PP-GA": "ga_call_including_resolution_and_pool"}[record_type],
            "validation_seconds": None, "save_seconds": None,
            "save_scope": "artifacts_through_precommit_check",
        },
        "artifacts": [], "error": None,
    }


def _record_error(exc: BaseException, phase: str, run_index: int | None) -> dict[str, Any]:
    generation_index = getattr(exc, "generation_index", None)
    population_index = getattr(exc, "population_index", None)
    cause = getattr(exc, "cause", None)
    message = str(cause if cause is not None else exc) or type(exc).__name__
    return {"phase": phase, "type": type(exc).__name__, "message": message[:2048],
            "generation_index": generation_index, "population_index": population_index, "run_index": run_index}


def _read_trace_bytes(raw: bytes, path: Path | None = None):
    with tempfile.TemporaryDirectory(prefix="packing3d_research_trace_") as temp:
        trace = Path(temp) / "generations.jsonl"
        trace.write_bytes(raw)
        return read_generations_jsonl(trace)


def _validate_embedded_result(data: dict[str, Any]) -> PackingResult:
    if data.get("format") != "packing-result" or data.get("version") != 1:
        raise ValueError("trace contains a non-v1 packing result")
    c = data.get("container")
    if type(c) is not dict or set(c) != {"width", "height", "depth"}:
        raise ValueError("trace packing result has invalid container")
    container = Container(c["width"], c["height"], c["depth"])
    raw_placements = data.get("placements")
    if type(raw_placements) is not list:
        raise ValueError("trace packing result placements must be an array")
    placements = []
    for item in raw_placements:
        if type(item) is not dict or set(item) != {"id", "size", "position"}:
            raise ValueError("trace packing result has invalid placement")
        size = item["size"]
        if type(size) is not dict or set(size) != {"width", "height", "depth"}:
            raise ValueError("trace packing result has invalid size")
        box = Box(item["id"], size["width"], size["height"], size["depth"])
        pos_data = item["position"]
        if pos_data is None:
            point = None
        else:
            if type(pos_data) is not dict or set(pos_data) != {"x", "y", "z"}:
                raise ValueError("trace packing result has invalid position")
            point = Point(pos_data["x"], pos_data["y"], pos_data["z"])
        placements.append(Placement(box, point))
    return PackingResult(container, tuple(placements))


def _readback_and_verify(
    directory: Path,
    record: dict[str, Any],
    run: GARunResult | PPGARunResult | None,
    single_result: PackingResult | None,
) -> None:
    record_id = record["record_id"]
    metadata_data = None
    best_result = None
    trace_rows = None
    for artifact in record["artifacts"]:
        target = _safe_child(directory, artifact["path"])
        raw = target.read_bytes()
        if len(raw) != artifact["bytes"] or _sha256(raw) != artifact["sha256"]:
            raise ValueError(f"artifact hash/size mismatch: {artifact['path']}")
        if artifact["role"] == "metadata":
            metadata = RunMetadata(_loads_strict_json(raw.decode("utf-8-sig")))
            metadata_data = metadata.to_dict()
            if run is not None and metadata_data != run.run_metadata.to_dict():
                raise ValueError("saved GA metadata differs from source run")
        elif artifact["role"] == "result":
            best_result = read_result_json(target)
            expected = run.best.packing_result if run is not None else single_result
            if best_result != expected:
                raise ValueError("saved result differs from source result")
        elif artifact["role"] == "profile":
            profile = read_decoder_profile_json(target)
            if run is None or profile != run.profile:
                raise ValueError("saved profile differs from source run")
        elif artifact["role"] == "trace":
            header, rows = read_generations_jsonl(target)
            trace_rows = (header, rows)
    if single_result is not None:
        return
    assert run is not None and metadata_data is not None and best_result is not None
    validate_run_metadata(metadata_data)
    if metadata_data["run_id"] != record["ga_run_id"]:
        raise ValueError("saved metadata run_id differs from Record")
    summary_best = record["summary"]["best"]
    if best_result.container != run.container or best_result != run.best.packing_result:
        raise ValueError("best result differs from run")
    if summary_best["generation_index"] != run.best_generation_index or summary_best["population_index"] != run.best_population_index:
        raise ValueError("best indexes differ from run")
    if tuple(summary_best["genes"]) != run.best.individual.genes or summary_best["fitness"] != run.best.fitness:
        raise ValueError("best genes/fitness differ from run")
    if (summary_best["placed_count"], summary_best["failed_count"], summary_best["placed_volume"], summary_best["density"]) != (
            best_result.placed_count, best_result.failed_count, best_result.placed_volume, best_result.density):
        raise ValueError("best Summary values differ from result artifact")
    if record["save_level"] == "full":
        if trace_rows is None:
            raise ValueError("Full record omitted generations trace")
        header, rows = trace_rows
        if header["record_id"] != record_id or header["ga_run_id"] != record["ga_run_id"]:
            raise ValueError("trace identity differs from Record")
        expected_raw = _build_trace(run, record_id)
        expected_header, expected_rows = _read_trace_bytes(expected_raw)
        if header != expected_header or rows != expected_rows:
            raise ValueError("read-back Full trace differs from source GA result")
        if len(rows) != run.config.population_size * run.config.generation_count:
            raise ValueError("Full trace did not preserve every GA evaluation")
        for row in rows:
            result = _validate_embedded_result(row["packing_result"])
            _validate_result(result, record["snapshot"], row["genes"])
        _validate_trace_summary(rows, record["summary"], record["snapshot"], record["record_type"])


def _validate_trace_summary(rows: tuple[dict[str, Any], ...], summary: dict[str, Any], snapshot: dict[str, Any], record_type: str) -> None:
    population = snapshot["ga"]["population_size"]
    for gen_index, expected in enumerate(summary["generations"]):
        group = rows[gen_index * population:(gen_index + 1) * population]
        fitnesses = [row["fitness"] for row in group]
        best_index = max(range(population), key=lambda i: fitnesses[i])
        if min(fitnesses) != expected["min"] or max(fitnesses) != expected["max"]:
            raise ValueError("Full trace generation min/max differs from Summary")
        if math.fsum(fitnesses) / population != expected["mean"]:
            raise ValueError("Full trace generation mean differs from Summary")
        if best_index != expected["best_population_index"]:
            raise ValueError("Full trace generation tie/best index differs from Summary")
        if group[best_index]["fitness"] != expected["best_fitness"]:
            raise ValueError("Full trace generation best differs from Summary")
        expected_decoder = "BL" if record_type == "BL-GA" or gen_index == 0 else "PP"
        if any(row["decoder"] != expected_decoder for row in group):
            raise ValueError("Full trace decoder progression invalid")
        for row in group:
            if expected_decoder == "BL" and row["pp_input_state"] is not None:
                raise ValueError("BL trace evaluation must have null pp_input_state")
            if expected_decoder == "PP":
                if row["pp_input_state"] is None:
                    raise ValueError("PP trace evaluation requires pp_input_state")
                _validate_trace_state_bounds(row["pp_input_state"], snapshot)
    best = summary["best"]
    best_row = rows[best["generation_index"] * population + best["population_index"]]
    if best_row["genes"] != best["genes"] or best_row["fitness"] != best["fitness"]:
        raise ValueError("Full trace overall best differs from Summary")
    if best_row["packing_result"] != _result_data(_validate_embedded_result(best_row["packing_result"])):
        raise ValueError("Full trace best result is not canonical packing-result v1")


def _validate_trace_state_bounds(state: list[list[Any]], snapshot: dict[str, Any]) -> None:
    ids = [row[0] for row in snapshot["problem"]["source_boxes"]]
    if [row[0] for row in state] != sorted(ids):
        raise ValueError("trace PP state must contain all IDs in ascending order")
    sizes = {row[0]: row[1:] for row in snapshot["problem"]["source_boxes"]}
    bounds = snapshot["problem"]["container"]
    for box_id, position in state:
        if position is not None and any(position[axis] + sizes[box_id][axis] > bounds[axis] for axis in range(3)):
            raise ValueError("trace PP input state contains an out-of-bounds position")


def _build_record_data(
    *,
    run: GARunResult | PPGARunResult | None,
    result: PackingResult | None,
    record_id: str,
    capture: ResearchCapture,
    save_level: str,
    experiment_id: str | None,
    run_index: int | None,
) -> dict[str, Any]:
    snapshot = capture.snapshot
    record_type = ("BL-GA" if run.decoder_name == "BL" else "PP-GA") if run is not None else snapshot["decoder"]["name"]
    seed = run.seed if run is not None else None
    base = _record_base(record_id, record_type, snapshot, capture,
                        experiment_id=experiment_id, run_index=run_index, seed=seed)
    base["save_level"] = save_level
    if run is None:
        if result is None:
            raise ValueError("single Record requires result")
        summary = _validate_single_result(result, snapshot)
        base["summary"] = summary
        base["validation"].update(expected_evaluations=1, checked_evaluations=1,
                                  geometry_support="passed", source_run_match="not_applicable")
        base["ga_run_id"] = None
    else:
        if result is not None:
            raise ValueError("GA Record does not accept separate result")
        summary, checked = _validate_ga_run(run, snapshot, save_level == "full")
        base["summary"] = summary
        base["validation"].update(
            scope="all_evaluations" if save_level == "full" else "best",
            expected_evaluations=run.config.population_size * run.config.generation_count if save_level == "full" else 1,
            checked_evaluations=checked if save_level == "full" else 1,
            geometry_support="passed", source_run_match="passed" if save_level == "full" else "not_applicable",
        )
        base["ga_run_id"] = run.run_metadata["run_id"]
    base["status"] = "saving"
    return base


def save_research_record(
    directory: str | os.PathLike[str],
    capture: ResearchCapture,
    *,
    result: PackingResult | None = None,
    run: GARunResult | PPGARunResult | None = None,
    save_level: str = "summary",
    experiment_id: str | None = None,
    run_index: int | None = None,
    record_id: str | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> ResearchRecord:
    """Save and independently verify a new Record; never reuse a directory."""
    if type(capture) is not ResearchCapture:
        raise TypeError("capture must be a ResearchCapture from the execution wrapper")
    if (run is None) == (result is None):
        raise ValueError("provide exactly one of run or result")
    if save_level not in ("summary", "full"):
        raise ValueError("save_level must be summary or full")
    if run is not None and type(run) not in (GARunResult, PPGARunResult):
        raise TypeError("run must be GARunResult or PPGARunResult")
    if status_callback is not None and not callable(status_callback):
        raise TypeError("status_callback must be callable or None")
    if result is not None and type(result) is not PackingResult:
        raise TypeError("result must be PackingResult")
    if capture.source_bytes is not None:
        expected_source_hash = capture.snapshot["problem"]["fixture_sha256"]
        if expected_source_hash is None or _sha256(capture.source_bytes) != expected_source_hash:
            raise ValueError("captured source bytes do not match snapshot.fixture_sha256")
        _validate_source_bytes(capture.source_bytes, capture.snapshot["problem"])
    if run is None and save_level != "summary":
        raise ValueError("single BL/PP records only support summary")
    if run is not None:
        if type(run.run_metadata) is not RunMetadata:
            raise ValueError("GA Record requires completed run metadata")
        if capture.snapshot != snapshot_from_ga(run):
            raise ValueError("capture snapshot does not match GA run")
    rid = str(uuid4()) if record_id is None else record_id
    if experiment_id is not None and run_index is None:
        raise ValueError("experiment_id requires run_index")
    if experiment_id is None and run_index is not None:
        raise ValueError("run_index requires experiment_id")
    destination = Path(directory)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    base = _record_base(rid, ("BL-GA" if run.decoder_name == "BL" else "PP-GA") if run else capture.snapshot["decoder"]["name"],
                        capture.snapshot, capture, experiment_id=experiment_id, run_index=run_index,
                        seed=run.seed if run else None)
    base["save_level"] = save_level
    if run is not None:
        base["ga_run_id"] = run.run_metadata["run_id"]
        if save_level == "full":
            base["validation"].update(scope="all_evaluations", expected_evaluations=run.config.population_size * run.config.generation_count,
                                      source_run_match="not_run")
    record_path = destination / "record.json"
    phase = "validation"
    started_save = None
    validate_started = time.perf_counter()
    artifacts: list[dict[str, Any]] = []
    record: dict[str, Any] = base
    try:
        _write_record_atomic(record_path, base)
        base["status"] = "validating"
        base["timing"]["recorded_at"] = utc_now()
        _write_record_atomic(record_path, base)
        if status_callback is not None:
            status_callback("validating")
        record = _build_record_data(run=run, result=result, record_id=rid, capture=capture,
                                    save_level=save_level, experiment_id=experiment_id, run_index=run_index)
        validation_seconds = time.perf_counter() - validate_started
        record["timing"]["validation_seconds"] = validation_seconds
        record["status"] = "saving"
        _write_record_atomic(record_path, record)
        if status_callback is not None:
            status_callback("saving")
        phase = "save"
        started_save = time.perf_counter()

        if run is None:
            assert result is not None
            result_raw = _render_result_json(result).encode("utf-8")
            _store_artifact(destination, "result.json", result_raw)
            artifacts.append(_artifact("result", "result.json", "packing-result", 1, result_raw))
        else:
            meta_raw = _json_bytes(run.run_metadata.to_dict())
            validate_run_metadata(run.run_metadata.to_dict())
            _store_artifact(destination, "run.metadata.json", meta_raw)
            artifacts.append(_artifact("metadata", "run.metadata.json", "packing-ga-run-metadata", 2, meta_raw))
            result_raw = _render_result_json(run.best.packing_result).encode("utf-8")
            _store_artifact(destination, "best.result.json", result_raw)
            artifacts.append(_artifact("result", "best.result.json", "packing-result", 1, result_raw))
            if save_level == "full":
                trace = _build_trace(run, rid)
                _store_artifact(destination, "generations.jsonl", trace)
                artifacts.append(_artifact("trace", "generations.jsonl", "packing3d-ga-state-trace", 1, trace))
            if run.profile is not None:
                profile_raw = _profile_bytes(run.profile)
                _store_artifact(destination, "profile.json", profile_raw)
                artifacts.append(_artifact("profile", "profile.json", "packing3d-decoder-profile", 1, profile_raw))
        if capture.source_bytes is not None and experiment_id is None:
            expected_hash = capture.snapshot["problem"]["fixture_sha256"]
            if expected_hash is None or _sha256(capture.source_bytes) != expected_hash:
                raise ValueError("captured source bytes do not match snapshot.fixture_sha256")
            _store_artifact(destination, "source.txt", capture.source_bytes)
            artifacts.append(_artifact("source", "source.txt", "raw-bytes", None, capture.source_bytes))
        record["artifacts"] = artifacts
        _write_record_atomic(record_path, record)
        _verify_declared_artifacts(destination, record)
        _readback_and_verify(destination, record, run, result)
        record["timing"]["save_seconds"] = time.perf_counter() - started_save
        record["timing"]["recorded_at"] = utc_now()
        record["status"] = "completed"
        record["error"] = None
        validate_research_record(record)
        # Final canonical record.json is published after every artifact and check.
        _write_record_atomic(record_path, record)
        return ResearchRecord(record)
    except Exception as exc:
        try:
            failure = base
            failure["status"] = "failed"
            failure["summary"] = record.get("summary")
            failure["artifacts"] = list(artifacts)
            failure["validation"] = record.get("validation", failure["validation"])
            failure["error"] = _record_error(exc, phase, run_index)
            failure["timing"]["recorded_at"] = utc_now()
            if phase == "validation":
                failure["timing"]["validation_seconds"] = time.perf_counter() - validate_started
            elif started_save is not None:
                failure["timing"]["save_seconds"] = time.perf_counter() - started_save
            _write_record_atomic(record_path, failure)
        except Exception:
            pass
        raise


def _store_artifact(directory: Path, relative: str, raw: bytes) -> None:
    path = _safe_child(directory, relative)
    _atomic_bytes(path, raw)


def _verify_declared_artifacts(directory: Path, record: dict[str, Any]) -> None:
    for artifact in record["artifacts"]:
        target = _safe_child(directory, artifact["path"])
        raw = target.read_bytes()
        if len(raw) != artifact["bytes"] or _sha256(raw) != artifact["sha256"]:
            raise ValueError(f"artifact hash/size mismatch: {artifact['path']}")


def _verify_record_bundle(directory: Path, record: ResearchRecord) -> ResearchRecord:
    data = record.to_dict()
    if data["status"] != "completed":
        return record
    _verify_declared_artifacts(directory, data)
    roles = {artifact["role"]: artifact for artifact in data["artifacts"]}
    if "result" in roles:
        result_path = _safe_child(directory, roles["result"]["path"])
        result = read_result_json(result_path)
        _validate_result(result, data["snapshot"], data["summary"]["best"]["genes"] if data["record_type"].endswith("-GA") else None)
        if data["record_type"].endswith("-GA"):
            best = data["summary"]["best"]
        else:
            best = data["summary"]
        if (result.placed_count, result.failed_count, result.placed_volume, result.density) != (
                best["placed_count"], best["failed_count"], best["placed_volume"], best["density"]):
            raise InputFormatError("Record summary does not match its result artifact")
    if data["record_type"].endswith("-GA"):
        metadata_path = _safe_child(directory, roles["metadata"]["path"])
        metadata = RunMetadata(_loads_strict_json(metadata_path.read_text(encoding="utf-8-sig")))
        md = metadata.to_dict()
        if md["run_id"] != data["ga_run_id"]:
            raise InputFormatError("Record ga_run_id differs from metadata")
        snapshot = snapshot_from_metadata(md, data["snapshot"]["profile"])
        if snapshot != data["snapshot"] or md["ga"]["seed"] != data["seed"]:
            raise InputFormatError("Record snapshot differs from metadata")
        if data["save_level"] == "full":
            header, rows = read_generations_jsonl(_safe_child(directory, roles["trace"]["path"]))
            if header["record_id"] != data["record_id"] or header["ga_run_id"] != data["ga_run_id"]:
                raise InputFormatError("Full trace identity mismatch")
            for row in rows:
                trace_result = _validate_embedded_result(row["packing_result"])
                _validate_result(trace_result, data["snapshot"], row["genes"])
                if row["placed_count"] != trace_result.placed_count or row["failed_count"] != trace_result.failed_count or row["placed_volume"] != trace_result.placed_volume or row["density"] != trace_result.density or row["fitness"] != trace_result.density:
                    raise InputFormatError("Full trace evaluation summary differs from result")
                expected_decoder = "BL" if data["record_type"] == "BL-GA" or row["generation_index"] == 0 else "PP"
                if row["decoder"] != expected_decoder:
                    raise InputFormatError("Full trace decoder progression mismatch")
            _validate_trace_summary(rows, data["summary"], data["snapshot"], data["record_type"])
        if "profile" in roles:
            profile = read_decoder_profile_json(_safe_child(directory, roles["profile"]["path"]))
            if profile.run_id != data["ga_run_id"] or profile.decoder != data["snapshot"]["decoder"]["name"]:
                raise InputFormatError("profile does not match GA run")
    if "source" in roles:
        raw = _safe_child(directory, roles["source"]["path"]).read_bytes()
        source_hash = data["snapshot"]["problem"]["fixture_sha256"]
        if source_hash is not None and _sha256(raw) != source_hash:
            raise InputFormatError("source bytes do not match snapshot.fixture_sha256")
        _validate_source_bytes(raw, data["snapshot"]["problem"])
    return record


def snapshot_from_metadata(metadata: dict[str, Any], profile: bool) -> dict[str, Any]:
    d, execution, workers = metadata["decoder"], metadata["execution"], metadata["execution"]["workers"]
    return {
        "problem": metadata["problem"],
        "decoder": {key: d[key] for key in (
            "name", "priority", "reference_version", "fitness_version", "pp_initialization",
            "candidate_policy", "candidate_policy_version", "min_support_percent", "support_model_version")},
        "ga": {key: metadata["ga"][key] for key in ("population_size", "generation_count", "crossover_rate", "mutation_rate")},
        "execution": {"workers_request": {"mode": workers["requested_mode"], "manual_workers": workers["manual_workers"]},
                      "workers": workers["resolved_workers"], "cache": execution["cache"], "start_method": execution["start_method"]},
        "profile": profile,
        "pp_input_state": None,
    }


def read_research_record(directory: str | os.PathLike[str], *, verify_artifacts: bool = True) -> ResearchRecord:
    root = Path(directory)
    record_path = root / "record.json"
    from .research import read_research_record_json
    record = read_research_record_json(record_path)
    if not verify_artifacts:
        return record
    try:
        return _verify_record_bundle(root, record)
    except InputFormatError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise InputFormatError(f"{record_path}: invalid Record artifact set: {exc}") from exc
