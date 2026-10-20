"""Strict v1 value models and codecs for research records and experiments.

The research layer is deliberately outside the BL/PP and GA implementation.
It stores completed calls and validates their artifacts without changing how a
packing run is evaluated.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from ._io_integer import parse_public_decimal_token, require_public_io_int
from .errors import InputFormatError
from .io_result_json import _pairs_no_duplicates, _reject_constant
from .model import Box, Container, PackingResult, Placement, Point
from .io_run_metadata import (
    validate_environment,
    validate_problem,
)
from .validate import validate_result as validate_packing_result
from .workers import WorkersRequest, resolve_workers

RECORD_KEYS = (
    "format", "version", "record_id", "record_type", "experiment_id", "run_index",
    "ga_run_id", "status", "seed", "save_level", "snapshot", "summary",
    "validation", "provenance", "timing", "artifacts", "error",
)
SNAPSHOT_KEYS = ("problem", "decoder", "ga", "execution", "profile", "pp_input_state")
DECODER_KEYS = (
    "name", "priority", "reference_version", "fitness_version", "pp_initialization",
    "candidate_policy", "candidate_policy_version", "min_support_percent", "support_model_version",
)
EXECUTION_KEYS = ("workers_request", "workers", "cache", "start_method")
RECORD_TIMING_KEYS = (
    "execution_started_at", "execution_finished_at", "recorded_at", "call_wall_seconds",
    "execution_seconds", "execution_scope", "validation_seconds", "save_seconds", "save_scope",
)
PROVENANCE_KEYS = (
    "distribution_version", "git_head", "git_state", "loaded_code_verified", "python", "os",
    "cpu", "lockfile_sha256", "unavailable", "reproducibility_limits",
)
VALIDATION_KEYS = (
    "scope", "expected_evaluations", "checked_evaluations", "geometry_support",
    "source_run_match", "issues",
)
ARTIFACT_KEYS = ("role", "path", "format", "version", "bytes", "sha256")
ERROR_KEYS = ("phase", "type", "message", "generation_index", "population_index", "run_index")
SUMMARY_SINGLE_KEYS = ("placed_count", "failed_count", "placed_volume", "density")
SUMMARY_GA_KEYS = ("best", "generations", "evaluation_stats")
BEST_KEYS = (
    "generation_index", "population_index", "genes", "fitness", "placed_count",
    "failed_count", "placed_volume", "density",
)
GENERATION_SUMMARY_KEYS = (
    "generation_index", "decoder", "count", "min", "max", "mean",
    "best_population_index", "best_fitness", "cumulative_best",
)
CUMULATIVE_BEST_KEYS = ("generation_index", "population_index", "fitness")
EVALUATION_STATS_KEYS = ("requested", "computed", "cache_hits")
TRACE_HEADER_KEYS = (
    "format", "version", "record_id", "ga_run_id", "generation_count", "population_size", "box_count",
)
TRACE_EVALUATION_KEYS = (
    "generation_index", "population_index", "genes", "fitness", "decoder", "packing_result",
    "pp_input_state", "placed_count", "failed_count", "placed_volume", "density",
)
EXPERIMENT_KEYS = (
    "format", "version", "experiment_id", "status", "record_type", "save_level", "run_count",
    "start_seed", "index_width", "fixed_snapshot", "workers_selection", "provenance", "timing",
    "input_artifacts", "trials", "error",
)
WORKER_SELECTION_KEYS = ("request", "resolved_execution", "environment")
WORKERS_REQUEST_KEYS = ("mode", "manual_workers")
EXPERIMENT_TIMING_KEYS = ("started_at", "finished_at", "elapsed_seconds")
TRIAL_KEYS = ("run_index", "seed", "record_id", "record_path", "record_sha256", "status")
ARTIFACT_FORMATS = {
    "metadata": ("run.metadata.json", "packing-ga-run-metadata", 2),
    "result": (None, "packing-result", 1),
    "trace": ("generations.jsonl", "packing3d-ga-state-trace", 1),
    "profile": ("profile.json", "packing3d-decoder-profile", 1),
    "source": ("source.txt", "raw-bytes", None),
    "problem": ("input/problem.json", "packing3d-research-problem", 1),
}


def _object(value: Any, keys: tuple[str, ...], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{path}: expected object")
    missing, extra = set(keys) - set(value), set(value) - set(keys)
    if missing or extra:
        raise ValueError(f"{path}: missing={sorted(missing)}, unknown={sorted(extra)}")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{path}: expected array")
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    require_public_io_int(value, name=path)
    if value < minimum:
        raise ValueError(f"{path}: expected integer >= {minimum}")
    return value


def _number(value: Any, path: str, minimum: float = 0.0, maximum: float | None = None) -> int | float:
    if type(value) not in (int, float):
        raise TypeError(f"{path}: expected finite JSON number")
    if type(value) is int:
        require_public_io_int(value, name=path)
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{path}: number out of range")
    return value


def _string(value: Any, path: str, *, nullable: bool = False, nonempty: bool = True) -> str | None:
    if nullable and value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{path}: expected string")
    if nonempty and not value:
        raise ValueError(f"{path}: expected non-empty string")
    return value


def _enum(value: Any, values: tuple[str, ...], path: str) -> str:
    _string(value, path)
    if value not in values:
        raise ValueError(f"{path}: unsupported value {value!r}")
    return value


def _uuid(value: Any, path: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    raw = _string(value, path)
    try:
        parsed = UUID(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid UUID") from exc
    if parsed.version != 4 or str(parsed) != raw:
        raise ValueError(f"{path}: expected canonical lowercase UUID v4")
    return raw


def _hash(value: Any, path: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    raw = _string(value, path)
    if re.fullmatch(r"[0-9a-f]{64}", raw) is None:
        raise ValueError(f"{path}: expected lowercase SHA-256")
    return raw


def _time(value: Any, path: str, *, nullable: bool = False) -> str | None:
    raw = _string(value, path, nullable=nullable)
    if raw is None:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", raw) is None:
        raise ValueError(f"{path}: expected UTC timestamp with six fractional digits")
    try:
        datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ValueError(f"{path}: invalid UTC timestamp") from exc
    return raw


def _nullable_integer(value: Any, path: str, minimum: int = 0) -> int | None:
    return None if value is None else _integer(value, path, minimum)


def _string_array(value: Any, path: str) -> None:
    for index, item in enumerate(_array(value, path)):
        _string(item, f"{path}[{index}]")


def _validate_state(value: Any, ids: list[int], path: str) -> None:
    rows = _array(value, path)
    if len(rows) != len(ids):
        raise ValueError(f"{path}: state must contain every problem ID exactly once")
    seen: list[int] = []
    for index, row in enumerate(rows):
        if type(row) is not list or len(row) != 2:
            raise ValueError(f"{path}[{index}]: expected [box_id, position]")
        box_id = _integer(row[0], f"{path}[{index}][0]")
        seen.append(box_id)
        pos = row[1]
        if pos is not None:
            if type(pos) is not list or len(pos) != 3:
                raise ValueError(f"{path}[{index}][1]: expected three coordinates or null")
            for axis, coord in enumerate(pos):
                _integer(coord, f"{path}[{index}][1][{axis}]")
    if seen != ids:
        raise ValueError(f"{path}: IDs must be complete and sorted")


def _validate_snapshot(snapshot: Any, record_type: str) -> None:
    s = _object(snapshot, SNAPSHOT_KEYS, "$.snapshot")
    problem = s["problem"]
    validate_problem(problem)
    ids = [row[0] for row in problem["source_boxes"]]
    decoder = _object(s["decoder"], DECODER_KEYS, "$.snapshot.decoder")
    name = "PP" if record_type in ("PP", "PP-GA") else "BL"
    _enum(decoder["name"], ("BL", "PP"), "$.snapshot.decoder.name")
    if decoder["name"] != name:
        raise ValueError("$.snapshot.decoder.name: does not match record_type")
    _enum(decoder["priority"], ("BottomFirst", "BackFirst"), "$.snapshot.decoder.priority")
    _enum(decoder["reference_version"], (f"{name}-ref-v2",), "$.snapshot.decoder.reference_version")
    _enum(decoder["candidate_policy"], ("Cartesian", "LocalNeighbors"), "$.snapshot.decoder.candidate_policy")
    if _integer(decoder["candidate_policy_version"], "$.snapshot.decoder.candidate_policy_version", 1) != 1:
        raise ValueError("$.snapshot.decoder.candidate_policy_version: expected 1")
    percent = _integer(decoder["min_support_percent"], "$.snapshot.decoder.min_support_percent")
    if percent > 100:
        raise ValueError("$.snapshot.decoder.min_support_percent: expected 0 through 100")
    _enum(decoder["support_model_version"], ("support-union-v1",), "$.snapshot.decoder.support_model_version")
    is_ga = record_type.endswith("-GA")
    _enum(decoder["fitness_version"], ("density-v1",), "$.snapshot.decoder.fitness_version") if is_ga else None
    if not is_ga and decoder["fitness_version"] is not None:
        raise ValueError("$.snapshot.decoder.fitness_version: must be null for single runs")
    pp_initialization = "bl_first_generation" if record_type == "PP-GA" else (
        "explicit_input_state" if record_type == "PP" else None
    )
    if decoder["pp_initialization"] != pp_initialization:
        raise ValueError("$.snapshot.decoder.pp_initialization: inconsistent with record_type")

    if is_ga:
        ga = _object(s["ga"], ("population_size", "generation_count", "crossover_rate", "mutation_rate"), "$.snapshot.ga")
        _integer(ga["population_size"], "$.snapshot.ga.population_size", 1)
        _integer(ga["generation_count"], "$.snapshot.ga.generation_count", 1)
        _number(ga["crossover_rate"], "$.snapshot.ga.crossover_rate", maximum=1)
        _number(ga["mutation_rate"], "$.snapshot.ga.mutation_rate", maximum=1)
        _validate_execution(s["execution"], "$.snapshot.execution")
    else:
        if s["ga"] is not None or s["execution"] is not None:
            raise ValueError("$.snapshot.ga/execution: must be null for single runs")
    if type(s["profile"]) is not bool:
        raise TypeError("$.snapshot.profile: expected boolean")
    if not is_ga and s["profile"]:
        raise ValueError("$.snapshot.profile: single runs cannot be profiled")
    pp_state = s["pp_input_state"]
    if record_type == "PP":
        _validate_state(pp_state, ids, "$.snapshot.pp_input_state")
        dimensions = {row[0]: row[1:] for row in problem["source_boxes"]}
        bounds = problem["container"]
        for index, (box_id, pos) in enumerate(pp_state):
            if pos is None:
                continue
            size = dimensions[box_id]
            if any(pos[i] + size[i] > bounds[i] for i in range(3)):
                raise ValueError(f"$.snapshot.pp_input_state[{index}]: inherited position outside container")
    elif pp_state is not None:
        raise ValueError("$.snapshot.pp_input_state: only single PP accepts an input state")


def _validate_execution(value: Any, path: str) -> dict[str, Any]:
    ex = _object(value, EXECUTION_KEYS, path)
    request = _object(ex["workers_request"], WORKERS_REQUEST_KEYS, path + ".workers_request")
    mode = _enum(request["mode"], ("auto", "max", "manual"), path + ".workers_request.mode")
    manual = _nullable_integer(request["manual_workers"], path + ".workers_request.manual_workers", 1)
    if (mode == "manual") != (manual is not None):
        raise ValueError(path + ".workers_request: manual_workers must be set only in manual mode")
    _integer(ex["workers"], path + ".workers", 1)
    if type(ex["cache"]) is not bool:
        raise TypeError(path + ".cache: expected boolean")
    expected_method = None if ex["workers"] == 1 else "spawn"
    if ex["start_method"] != expected_method:
        raise ValueError(path + ".start_method: inconsistent with workers")
    if mode == "manual" and manual != ex["workers"]:
        raise ValueError(path + ": Manual requested workers must match resolved workers")
    return ex


def _validate_summary(value: Any, record_type: str, save_level: str, snapshot: dict[str, Any]) -> None:
    if record_type in ("BL", "PP"):
        summary = _object(value, SUMMARY_SINGLE_KEYS, "$.summary")
        for key in SUMMARY_SINGLE_KEYS[:3]:
            _integer(summary[key], "$.summary." + key)
        _number(summary["density"], "$.summary.density", maximum=1)
        ids = len(snapshot["problem"]["source_boxes"])
        if summary["placed_count"] + summary["failed_count"] != ids:
            raise ValueError("$.summary counts do not match the problem")
        if summary["placed_volume"] > math.prod(snapshot["problem"]["container"]):
            raise ValueError("$.summary.placed_volume exceeds container volume")
        if summary["density"] != summary["placed_volume"] / math.prod(snapshot["problem"]["container"]):
            raise ValueError("$.summary.density does not match placed volume")
        return
    summary = _object(value, SUMMARY_GA_KEYS, "$.summary")
    best = _object(summary["best"], BEST_KEYS, "$.summary.best")
    ga = snapshot["ga"]
    for key in ("generation_index", "population_index", "placed_count", "failed_count", "placed_volume"):
        _integer(best[key], "$.summary.best." + key)
    if best["generation_index"] >= ga["generation_count"] or best["population_index"] >= ga["population_size"]:
        raise ValueError("$.summary.best index out of range")
    genes = _array(best["genes"], "$.summary.best.genes")
    for i, gene in enumerate(genes):
        _integer(gene, f"$.summary.best.genes[{i}]")
    ids = snapshot["problem"]["source_box_order"]
    if len(genes) != len(ids) or set(genes) != set(ids):
        raise ValueError("$.summary.best.genes must be a permutation of problem IDs")
    for key in ("fitness", "density"):
        _number(best[key], "$.summary.best." + key, maximum=1)
    if best["fitness"] != best["density"]:
        raise ValueError("$.summary.best fitness and density differ")
    if best["placed_count"] + best["failed_count"] != len(ids):
        raise ValueError("$.summary.best counts do not match problem")
    if best["density"] != best["placed_volume"] / math.prod(snapshot["problem"]["container"]):
        raise ValueError("$.summary.best density does not match placed volume")
    requested_expected = ga["population_size"] * ga["generation_count"]
    stats = _object(summary["evaluation_stats"], EVALUATION_STATS_KEYS, "$.summary.evaluation_stats")
    for key in EVALUATION_STATS_KEYS:
        _integer(stats[key], "$.summary.evaluation_stats." + key)
    if stats["requested"] != requested_expected or stats["computed"] + stats["cache_hits"] != requested_expected:
        raise ValueError("$.summary.evaluation_stats inconsistent with GA dimensions")
    if not snapshot["execution"]["cache"] and stats["cache_hits"] != 0:
        raise ValueError("$.summary.evaluation_stats cache hits with cache disabled")
    generations = _array(summary["generations"], "$.summary.generations")
    if len(generations) != ga["generation_count"]:
        raise ValueError("$.summary.generations count differs from generation_count")
    cumulative: dict[str, Any] | None = None
    previous_mean_best = -1.0
    for index, raw in enumerate(generations):
        row = _object(raw, GENERATION_SUMMARY_KEYS, f"$.summary.generations[{index}]")
        if _integer(row["generation_index"], f"$.summary.generations[{index}].generation_index") != index:
            raise ValueError("$.summary.generations must be consecutive and ordered")
        expected_decoder = "BL" if record_type == "BL-GA" or index == 0 else "PP"
        if _enum(row["decoder"], ("BL", "PP"), f"$.summary.generations[{index}].decoder") != expected_decoder:
            raise ValueError("$.summary.generations decoder progression is invalid")
        if _integer(row["count"], f"$.summary.generations[{index}].count", 1) != ga["population_size"]:
            raise ValueError("$.summary.generations count differs from population_size")
        for key in ("min", "max", "mean", "best_fitness"):
            _number(row[key], f"$.summary.generations[{index}].{key}", maximum=1)
        best_population = _integer(row["best_population_index"], f"$.summary.generations[{index}].best_population_index")
        if best_population >= ga["population_size"] or row["min"] > row["max"] or not row["min"] <= row["mean"] <= row["max"]:
            raise ValueError("$.summary.generations statistics are inconsistent")
        if row["best_fitness"] != row["max"]:
            raise ValueError("$.summary.generations best_fitness must equal max")
        current = _object(row["cumulative_best"], CUMULATIVE_BEST_KEYS, f"$.summary.generations[{index}].cumulative_best")
        _integer(current["generation_index"], f"$.summary.generations[{index}].cumulative_best.generation_index")
        _integer(current["population_index"], f"$.summary.generations[{index}].cumulative_best.population_index")
        _number(current["fitness"], f"$.summary.generations[{index}].cumulative_best.fitness", maximum=1)
        if current["generation_index"] > index or current["population_index"] >= ga["population_size"]:
            raise ValueError("$.summary.generations cumulative best index out of range")
        if index == 0 and current != {"generation_index": 0, "population_index": best_population,
                                     "fitness": row["best_fitness"]}:
            raise ValueError("$.summary first cumulative best must equal the first generation best")
        if index and current["fitness"] < previous_mean_best:
            raise ValueError("$.summary.generations cumulative best regressed")
        if cumulative is not None and row["best_fitness"] > cumulative["fitness"]:
            if (current["generation_index"] != index or current["population_index"] != best_population
                    or current["fitness"] != row["best_fitness"]):
                raise ValueError("$.summary.generations cumulative best failed to update")
        elif cumulative is not None and row["best_fitness"] <= cumulative["fitness"] and current != cumulative:
            raise ValueError("$.summary.generations cumulative best changed without an improvement")
        cumulative = current
        previous_mean_best = current["fitness"]
    if cumulative != {"generation_index": best["generation_index"],
                      "population_index": best["population_index"], "fitness": best["fitness"]}:
        raise ValueError("$.summary cumulative best differs from overall best")
    if save_level not in ("summary", "full"):
        raise ValueError("$.save_level: unknown level")


def _validate_provenance(value: Any, path: str) -> None:
    p = _object(value, PROVENANCE_KEYS, path)
    _string(p["distribution_version"], path + ".distribution_version")
    git_head = _string(p["git_head"], path + ".git_head", nullable=True)
    if git_head is not None and re.fullmatch(r"[0-9a-f]{40}", git_head) is None:
        raise ValueError(path + ".git_head: expected full lowercase Git commit SHA")
    _enum(p["git_state"], ("clean", "dirty", "unavailable"), path + ".git_state")
    if type(p["loaded_code_verified"]) is not bool:
        raise TypeError(path + ".loaded_code_verified: expected boolean")
    os_obj = _object(p["os"], ("system", "release", "machine"), path + ".os")
    _string(os_obj["system"], path + ".os.system")
    for key in ("release", "machine"):
        _string(os_obj[key], path + ".os." + key, nullable=True)
    py_obj = _object(p["python"], ("implementation", "version"), path + ".python")
    _string(py_obj["implementation"], path + ".python.implementation")
    _string(py_obj["version"], path + ".python.version")
    cpu = _object(p["cpu"], ("label", "label_source", "detection"), path + ".cpu")
    _string(cpu["label"], path + ".cpu.label", nullable=True)
    if cpu["label"] is not None:
        _enum(cpu["label_source"], ("platform.processor", "proc_cpuinfo", "PROCESSOR_IDENTIFIER"), path + ".cpu.label_source")
    else:
        if cpu["label_source"] is not None:
            raise ValueError(path + ".cpu.label_source must be null when label is null")
    _hash(p["lockfile_sha256"], path + ".lockfile_sha256", nullable=True)
    _string_array(p["unavailable"], path + ".unavailable")
    _string_array(p["reproducibility_limits"], path + ".reproducibility_limits")
    if p["git_state"] == "unavailable" and p["git_head"] is not None:
        raise ValueError(path + ".git_head must be null when git is unavailable")
    if p["git_state"] != "clean" and not p["reproducibility_limits"]:
        raise ValueError(path + ": dirty/unavailable git state requires a reproducibility limit")
    if not p["loaded_code_verified"] and not p["reproducibility_limits"]:
        raise ValueError(path + ": unverified loaded code requires a reproducibility limit")


def _validate_record_timing(value: Any, record_type: str, path: str) -> None:
    t = _object(value, RECORD_TIMING_KEYS, path)
    _time(t["execution_started_at"], path + ".execution_started_at")
    _time(t["execution_finished_at"], path + ".execution_finished_at", nullable=True)
    _time(t["recorded_at"], path + ".recorded_at")
    for key in ("call_wall_seconds", "execution_seconds", "validation_seconds", "save_seconds"):
        _number(t[key], path + "." + key) if t[key] is not None else None
    scope = {"BL": "pack_call", "PP": "pack_pp_call", "BL-GA": "ga_call_including_resolution_and_pool",
             "PP-GA": "ga_call_including_resolution_and_pool"}[record_type]
    if _enum(t["execution_scope"], (scope,), path + ".execution_scope") != scope:
        raise ValueError(path + ".execution_scope inconsistent with record_type")
    if _enum(t["save_scope"], ("artifacts_through_precommit_check",), path + ".save_scope") != "artifacts_through_precommit_check":
        raise ValueError(path + ".save_scope invalid")


def _validate_validation(value: Any, record_type: str, save_level: str, snapshot: dict[str, Any], status: str) -> None:
    v = _object(value, VALIDATION_KEYS, "$.validation")
    expected_scope = "single_result" if not record_type.endswith("-GA") else ("all_evaluations" if save_level == "full" else "best")
    _enum(v["scope"], (expected_scope,), "$.validation.scope")
    expected = 1 if expected_scope != "all_evaluations" else snapshot["ga"]["population_size"] * snapshot["ga"]["generation_count"]
    if _integer(v["expected_evaluations"], "$.validation.expected_evaluations") != expected:
        raise ValueError("$.validation.expected_evaluations inconsistent with scope")
    checked = _integer(v["checked_evaluations"], "$.validation.checked_evaluations")
    if checked > expected:
        raise ValueError("$.validation.checked_evaluations exceeds expected count")
    _enum(v["geometry_support"], ("not_run", "passed", "failed"), "$.validation.geometry_support")
    match_values = ("not_run", "passed", "failed") if expected_scope == "all_evaluations" else ("not_applicable",)
    _enum(v["source_run_match"], match_values, "$.validation.source_run_match")
    _string_array(v["issues"], "$.validation.issues")
    if status == "completed":
        if checked != expected or v["geometry_support"] != "passed" or v["issues"]:
            raise ValueError("completed record must have fully passed validation")
        if expected_scope == "all_evaluations" and v["source_run_match"] != "passed":
            raise ValueError("completed Full record must match the source run")


def _validate_artifacts(value: Any, record_type: str, save_level: str, profile: bool, path: str,
                        *, in_experiment: bool) -> None:
    artifacts = _array(value, path)
    roles: set[str] = set()
    paths: set[str] = set()
    for index, raw in enumerate(artifacts):
        item = _object(raw, ARTIFACT_KEYS, f"{path}[{index}]")
        role = _enum(item["role"], tuple(ARTIFACT_FORMATS), f"{path}[{index}].role")
        if role in roles:
            raise ValueError(f"{path}: duplicate role {role}")
        roles.add(role)
        expected_path, expected_format, expected_version = ARTIFACT_FORMATS[role]
        if role == "result" and record_type.endswith("-GA"):
            expected_path = "best.result.json"
        if role == "result" and not record_type.endswith("-GA"):
            expected_path = "result.json"
        if item["path"] != expected_path:
            raise ValueError(f"{path}[{index}].path invalid for role {role}")
        _safe_relative_artifact_path(item["path"], f"{path}[{index}].path")
        if item["path"] in paths:
            raise ValueError(f"{path}: duplicate artifact path")
        paths.add(item["path"])
        _enum(item["format"], (expected_format,), f"{path}[{index}].format")
        if item["version"] != expected_version:
            raise ValueError(f"{path}[{index}].version invalid")
        _integer(item["bytes"], f"{path}[{index}].bytes")
        _hash(item["sha256"], f"{path}[{index}].sha256")
        if role == "problem":
            raise ValueError("Record cannot reference a problem artifact")
        if role == "source" and in_experiment:
            raise ValueError("Experiment child Records cannot reference source bytes")
    if "metadata" in roles and not record_type.endswith("-GA"):
        raise ValueError("single Record cannot contain GA metadata")
    if "trace" in roles and (not record_type.endswith("-GA") or save_level != "full"):
        raise ValueError("trace is only valid for Full GA Records")
    if "profile" in roles and (not record_type.endswith("-GA") or not profile):
        raise ValueError("profile artifact is only valid when profiled GA is recorded")
    if "source" in roles and record_type.endswith("-GA"):
        # Single source files are allowed; experiment children intentionally do not carry them.
        pass


def _safe_relative_artifact_path(value: Any, path: str) -> str:
    raw = _string(value, path)
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized) or ".." in normalized.split("/"):
        raise ValueError(f"{path}: unsafe artifact path")
    return raw


def validate_research_record(value: Any) -> dict[str, Any]:
    data = value.to_dict() if type(value) is ResearchRecord else value
    r = _object(data, RECORD_KEYS, "$")
    _enum(r["format"], ("packing3d-research-record",), "$.format")
    if _integer(r["version"], "$.version") != 1:
        raise ValueError("$.version: unsupported Record version")
    _uuid(r["record_id"], "$.record_id")
    record_type = _enum(r["record_type"], ("BL", "PP", "BL-GA", "PP-GA"), "$.record_type")
    experiment_id = _uuid(r["experiment_id"], "$.experiment_id", nullable=True)
    run_index = _nullable_integer(r["run_index"], "$.run_index")
    if (experiment_id is None) != (run_index is None):
        raise ValueError("$.experiment_id and $.run_index must both be null or both be set")
    ga_run_id = _uuid(r["ga_run_id"], "$.ga_run_id", nullable=True)
    status = _enum(r["status"], ("running", "validating", "saving", "completed", "failed"), "$.status")
    is_ga = record_type.endswith("-GA")
    seed = _nullable_integer(r["seed"], "$.seed")
    if is_ga != (seed is not None):
        raise ValueError("$.seed must be set exactly for GA records")
    if is_ga and status == "completed" and ga_run_id is None:
        raise ValueError("completed GA Record requires ga_run_id")
    if not is_ga and ga_run_id is not None:
        raise ValueError("single Record must not have ga_run_id")
    level = _enum(r["save_level"], ("summary", "full"), "$.save_level")
    if not is_ga and level != "summary":
        raise ValueError("single BL/PP Records only support summary")
    _validate_snapshot(r["snapshot"], record_type)
    if status in ("running", "validating") and r["summary"] is not None:
        raise ValueError("$.summary must be null before validation completes")
    if r["summary"] is not None:
        _validate_summary(r["summary"], record_type, level, r["snapshot"])
    _validate_validation(r["validation"], record_type, level, r["snapshot"], status)
    _validate_provenance(r["provenance"], "$.provenance")
    _validate_record_timing(r["timing"], record_type, "$.timing")
    _validate_artifacts(r["artifacts"], record_type, level, r["snapshot"]["profile"], "$.artifacts",
                        in_experiment=experiment_id is not None)
    error = r["error"]
    if error is not None:
        e = _object(error, ERROR_KEYS, "$.error")
        _enum(e["phase"], ("preflight", "execution", "validation", "save"), "$.error.phase")
        _string(e["type"], "$.error.type")
        _string(e["message"], "$.error.message")
        for key in ("generation_index", "population_index", "run_index"):
            _nullable_integer(e[key], "$.error." + key)
        if experiment_id is None and e["run_index"] is not None:
            raise ValueError("single Record error.run_index must be null")
        if experiment_id is not None and e["run_index"] != run_index:
            raise ValueError("Record error.run_index must match its experiment index")
    if status == "completed":
        if error is not None or r["summary"] is None:
            raise ValueError("completed Record requires summary and null error")
        expected_roles = {"result"} if not is_ga else {"metadata", "result"}
        if level == "full" and is_ga:
            expected_roles.add("trace")
        if r["snapshot"]["profile"]:
            expected_roles.add("profile")
        if not expected_roles.issubset({item["role"] for item in r["artifacts"]}):
            raise ValueError("completed Record is missing a required artifact")
        timing = r["timing"]
        for key in ("execution_finished_at", "call_wall_seconds", "execution_seconds", "validation_seconds", "save_seconds"):
            if timing[key] is None:
                raise ValueError(f"completed Record requires timing.{key}")
    elif status == "failed" and error is None:
        raise ValueError("failed Record requires error")
    return data


def _validate_worker_selection(value: Any, path: str, population_size: int) -> dict[str, Any]:
    ws = _object(value, WORKER_SELECTION_KEYS, path)
    request = _object(ws["request"], WORKERS_REQUEST_KEYS, path + ".request")
    mode = _enum(request["mode"], ("auto", "max", "manual"), path + ".request.mode")
    manual = _nullable_integer(request["manual_workers"], path + ".request.manual_workers", 1)
    if (mode == "manual") != (manual is not None):
        raise ValueError(path + ".request manual_workers inconsistent with mode")
    execution = _object(ws["resolved_execution"], ("workers", "cache", "start_method"), path + ".resolved_execution")
    if type(execution["cache"]) is not bool:
        raise TypeError(path + ".resolved_execution.cache must be boolean")
    workers = _object(execution["workers"], (
        "requested_mode", "manual_workers", "resolved_workers", "resolver_version", "auto_ratio", "auto_cap",
        "limit_reasons", "warning_reasons",
    ), path + ".resolved_execution.workers")
    resolved_mode = _enum(
        workers["requested_mode"], ("auto", "max", "manual"),
        path + ".resolved_execution.workers.requested_mode",
    )
    resolved_manual = _nullable_integer(
        workers["manual_workers"], path + ".resolved_execution.workers.manual_workers", 1,
    )
    if (resolved_mode == "manual") != (resolved_manual is not None):
        raise ValueError(path + ".resolved_execution workers manual_workers inconsistent with mode")
    if (resolved_mode, resolved_manual) != (mode, manual):
        raise ValueError(path + ".request and resolved_execution worker requests differ")
    _integer(workers["resolved_workers"], path + ".resolved_execution.workers.resolved_workers", 1)
    _enum(workers["resolver_version"], ("workers-static-v1",), path + ".resolved_execution.workers.resolver_version")
    _number(workers["auto_ratio"], path + ".resolved_execution.workers.auto_ratio")
    _integer(workers["auto_cap"], path + ".resolved_execution.workers.auto_cap", 1)
    for key in ("limit_reasons", "warning_reasons"):
        _string_array(workers[key], path + ".resolved_execution.workers." + key)
    if execution["start_method"] != (None if workers["resolved_workers"] == 1 else "spawn"):
        raise ValueError(path + ".resolved_execution.start_method inconsistent")
    _validate_provenance_environment_projection(ws["environment"], path + ".environment")
    environment = ws["environment"]
    expected_workers = resolve_workers(
        WorkersRequest(mode, manual),
        environment["cpu"]["detection"]["available_logical_estimate"],
        population_size,
        environment["executor"]["worker_limit"],
    )
    if workers != expected_workers:
        raise ValueError(path + ".resolved_execution workers inconsistent with request and environment")
    return ws


def _validate_provenance_environment_projection(value: Any, path: str) -> None:
    e = _object(value, ("os", "python", "implementation", "executor", "cpu"), path)
    validate_environment(e)


def _validate_experiment_artifacts(value: Any, path: str) -> None:
    items = _array(value, path)
    roles = set()
    for index, item_value in enumerate(items):
        item = _object(item_value, ARTIFACT_KEYS, f"{path}[{index}]")
        role = _enum(item["role"], ("problem", "source"), f"{path}[{index}].role")
        if role in roles:
            raise ValueError(f"{path}: duplicate {role} artifact")
        roles.add(role)
        filename, fmt, version = ARTIFACT_FORMATS[role]
        if role == "source":
            filename = "input/source.txt"
        _enum(item["path"], (filename,), f"{path}[{index}].path")
        _enum(item["format"], (fmt,), f"{path}[{index}].format")
        if item["version"] != version:
            raise ValueError(f"{path}[{index}].version invalid")
        _integer(item["bytes"], f"{path}[{index}].bytes")
        _hash(item["sha256"], f"{path}[{index}].sha256")


def _validate_trial(value: Any, index: int, experiment_id: str, start_seed: int, width: int, status: str) -> dict[str, Any]:
    t = _object(value, TRIAL_KEYS, f"$.trials[{index}]")
    if _integer(t["run_index"], f"$.trials[{index}].run_index") != index:
        raise ValueError("$.trials must be in exact index order")
    if _integer(t["seed"], f"$.trials[{index}].seed") != start_seed + index:
        raise ValueError("$.trials seed does not match start_seed + run_index")
    _uuid(t["record_id"], f"$.trials[{index}].record_id", nullable=True)
    record_path = t["record_path"]
    sha = _hash(t["record_sha256"], f"$.trials[{index}].record_sha256", nullable=True)
    trial_status = _enum(t["status"], ("planned", "running", "validating", "saving", "completed", "failed"), f"$.trials[{index}].status")
    if trial_status == "planned":
        if any(t[k] is not None for k in ("record_id", "record_path", "record_sha256")):
            raise ValueError(f"$.trials[{index}] planned trial must not have a Record path")
    else:
        expected_path = f"runs/{index:0{width}d}/record.json"
        if record_path != expected_path:
            raise ValueError(f"$.trials[{index}].record_path invalid")
        _safe_relative_artifact_path(record_path, f"$.trials[{index}].record_path")
        if t["record_id"] is None:
            raise ValueError(f"$.trials[{index}] started trial requires record_id")
        if trial_status == "completed" and sha is None:
            raise ValueError(f"$.trials[{index}] completed trial requires hash")
    return t


def validate_research_experiment(value: Any) -> dict[str, Any]:
    data = value.to_dict() if type(value) is ResearchExperiment else value
    e = _object(data, EXPERIMENT_KEYS, "$")
    _enum(e["format"], ("packing3d-research-experiment",), "$.format")
    if _integer(e["version"], "$.version") != 1:
        raise ValueError("$.version: unsupported Experiment version")
    experiment_id = _uuid(e["experiment_id"], "$.experiment_id")
    status = _enum(e["status"], ("planned", "running", "stopping", "completed", "failed", "stopped"), "$.status")
    record_type = _enum(e["record_type"], ("BL-GA", "PP-GA"), "$.record_type")
    _enum(e["save_level"], ("summary", "full"), "$.save_level")
    count = _integer(e["run_count"], "$.run_count", 1)
    seed = _integer(e["start_seed"], "$.start_seed")
    if seed + count - 1 > (10**1000 - 1):
        raise ValueError("$.start_seed + run_count exceeds public integer limit")
    width = max(3, len(str(count - 1)))
    if _integer(e["index_width"], "$.index_width", 3) != width:
        raise ValueError("$.index_width inconsistent with run_count")
    _validate_snapshot(e["fixed_snapshot"], record_type)
    if e["fixed_snapshot"]["pp_input_state"] is not None:
        raise ValueError("Experiment fixed_snapshot.pp_input_state must be null")
    ex = e["fixed_snapshot"]["execution"]
    if ex["workers_request"] != {"mode": "manual", "manual_workers": ex["workers"]}:
        raise ValueError("Experiment fixed snapshot must use resolved Manual worker setting")
    ws = _validate_worker_selection(
        e["workers_selection"], "$.workers_selection", e["fixed_snapshot"]["ga"]["population_size"],
    )
    if ws["resolved_execution"]["workers"]["resolved_workers"] != ex["workers"]:
        raise ValueError("$.workers_selection and fixed_snapshot worker counts differ")
    _validate_provenance(e["provenance"], "$.provenance")
    timing = _object(e["timing"], EXPERIMENT_TIMING_KEYS, "$.timing")
    _time(timing["started_at"], "$.timing.started_at")
    _time(timing["finished_at"], "$.timing.finished_at", nullable=True)
    if timing["elapsed_seconds"] is not None:
        _number(timing["elapsed_seconds"], "$.timing.elapsed_seconds")
    _validate_experiment_artifacts(e["input_artifacts"], "$.input_artifacts")
    input_roles = {a["role"] for a in e["input_artifacts"]}
    if status in ("running", "stopping", "completed", "stopped") and "problem" not in input_roles:
        raise ValueError("planned Experiment must persist problem.json before running")
    trials = _array(e["trials"], "$.trials")
    if len(trials) != count:
        raise ValueError("$.trials must contain run_count entries")
    ids: set[str] = set()
    seen_uncompleted = False
    seen_planned = False
    nonterminal_index: int | None = None
    for index, raw in enumerate(trials):
        trial = _validate_trial(raw, index, experiment_id, seed, width, status)
        if trial["record_id"] is not None:
            if trial["record_id"] in ids:
                raise ValueError("duplicate Record ID in Experiment")
            ids.add(trial["record_id"])
        if trial["status"] != "completed":
            seen_uncompleted = True
        elif seen_uncompleted:
            raise ValueError("completed Experiment trials must form a prefix")
        if trial["status"] not in ("completed", "planned"):
            if seen_planned:
                raise ValueError("in-flight or failed Trial cannot follow a planned Trial")
            if nonterminal_index is not None:
                raise ValueError("Experiment may contain at most one in-flight or failed Trial")
            nonterminal_index = index
            if any(later["status"] != "planned" for later in trials[index + 1:]):
                raise ValueError("in-flight or failed Trial must be followed only by planned Trials")
        elif trial["status"] == "planned":
            seen_planned = True
    error = e["error"]
    if error is not None:
        err = _object(error, ERROR_KEYS, "$.error")
        _enum(err["phase"], ("preflight", "execution", "validation", "save"), "$.error.phase")
        _string(err["type"], "$.error.type")
        _string(err["message"], "$.error.message")
        for key in ("generation_index", "population_index", "run_index"):
            _nullable_integer(err[key], "$.error." + key)
    trial_states = [trial["status"] for trial in trials]
    if status == "planned" and any(state != "planned" for state in trial_states):
        raise ValueError("planned Experiment may contain only planned Trials")
    if status in ("running", "stopping") and any(state == "failed" for state in trial_states):
        raise ValueError(f"{status} Experiment cannot contain a failed Trial")
    if any(state in ("running", "validating", "saving") for state in trial_states) and status not in (
        "running", "stopping", "failed"
    ):
        raise ValueError("in-flight Trial requires a running, stopping, or failed Experiment")
    if any(state == "failed" for state in trial_states) and status != "failed":
        raise ValueError("failed Trial requires a failed Experiment")
    if status == "completed":
        if any(t["status"] != "completed" for t in trials) or error is not None:
            raise ValueError("completed Experiment requires all trials completed and null error")
    elif status == "stopped":
        if any(t["status"] not in ("completed", "planned") for t in trials) or error is not None:
            raise ValueError("stopped Experiment cannot contain in-flight/failed trials")
    elif status == "failed" and error is None:
        raise ValueError("failed Experiment requires error")
    if status in ("completed", "failed", "stopped"):
        if timing["finished_at"] is None or timing["elapsed_seconds"] is None:
            raise ValueError("terminal Experiment requires finished time and elapsed duration")
    return data


@dataclass(frozen=True, slots=True, init=False)
class _FrozenJSON(Mapping[str, Any]):
    _text: str
    _keys: tuple[str, ...]

    def __init__(self, data: Any, validator, keys: tuple[str, ...]):
        validator(data)
        text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "_keys", keys)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._text, parse_int=parse_public_decimal_token)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


class ResearchRecord(_FrozenJSON):
    """Immutable strict v1 Research Record value."""

    def __init__(self, data: Any):
        super().__init__(data, validate_research_record, RECORD_KEYS)


class ResearchExperiment(_FrozenJSON):
    """Immutable strict v1 Research Experiment value."""

    def __init__(self, data: Any):
        super().__init__(data, validate_research_experiment, EXPERIMENT_KEYS)


def _read_json(path: str | os.PathLike[str]) -> Any:
    p = Path(path)
    try:
        raw = p.read_bytes()
        text = raw.decode("utf-8-sig")
        return json.loads(text, object_pairs_hook=_pairs_no_duplicates, parse_constant=_reject_constant,
                          parse_int=parse_public_decimal_token)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, InputFormatError):
            raise
        raise InputFormatError(f"{p}: {exc}") from exc


def _loads_strict_json(text: str) -> Any:
    """Decode embedded Research JSON with the same duplicate/number rules as roots."""
    if type(text) is not str:
        raise TypeError("JSON text must be a string")
    return json.loads(text, object_pairs_hook=_pairs_no_duplicates, parse_constant=_reject_constant,
                      parse_int=parse_public_decimal_token)


def read_research_record_json(path: str | os.PathLike[str]) -> ResearchRecord:
    try:
        return ResearchRecord(_read_json(path))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, InputFormatError):
            raise
        raise InputFormatError(f"{path}: {exc}") from exc


def read_research_experiment_json(path: str | os.PathLike[str]) -> ResearchExperiment:
    try:
        return ResearchExperiment(_read_json(path))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, InputFormatError):
            raise
        raise InputFormatError(f"{path}: {exc}") from exc


def _write_value(path: str | os.PathLike[str], value: _FrozenJSON, validator) -> None:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be str or os.PathLike")
    if not isinstance(value, _FrozenJSON):
        raise TypeError("value must be a validated research record/experiment")
    data = value.to_dict()
    validator(data)
    Path(path).write_text(value._text, encoding="utf-8", newline="\n")


def write_research_record_json(path: str | os.PathLike[str], record: ResearchRecord) -> None:
    """Write a standalone codec value; lifecycle storage uses exclusive atomic publication."""
    _write_value(path, record, validate_research_record)


def write_research_experiment_json(path: str | os.PathLike[str], experiment: ResearchExperiment) -> None:
    _write_value(path, experiment, validate_research_experiment)


def read_generations_jsonl(path: str | os.PathLike[str]) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    p = Path(path)
    try:
        raw = p.read_bytes()
        text = raw.decode("utf-8-sig")
        if not text.endswith("\n") or "\r" in text:
            raise ValueError("JSONL must use LF and a final newline")
        lines = text[:-1].split("\n")
        if any(not line for line in lines):
            raise ValueError("JSONL contains an empty line")
        values = [json.loads(line, object_pairs_hook=_pairs_no_duplicates, parse_constant=_reject_constant,
                             parse_int=parse_public_decimal_token) for line in lines]
        if not values:
            raise ValueError("JSONL is empty")
        header = _object(values[0], TRACE_HEADER_KEYS, "$.header")
        _enum(header["format"], ("packing3d-ga-state-trace",), "$.header.format")
        if _integer(header["version"], "$.header.version") != 1:
            raise ValueError("$.header.version must be 1")
        _uuid(header["record_id"], "$.header.record_id")
        _uuid(header["ga_run_id"], "$.header.ga_run_id")
        generations = _integer(header["generation_count"], "$.header.generation_count", 1)
        population = _integer(header["population_size"], "$.header.population_size", 1)
        boxes = _integer(header["box_count"], "$.header.box_count")
        if len(values) != 1 + generations * population:
            raise ValueError("JSONL evaluation row count does not match header")
        evaluations = []
        ids = None
        for offset, raw_item in enumerate(values[1:]):
            path_name = f"$.evaluations[{offset}]"
            item = _object(raw_item, TRACE_EVALUATION_KEYS, path_name)
            g_index, p_index = divmod(offset, population)
            if _integer(item["generation_index"], path_name + ".generation_index") != g_index or _integer(item["population_index"], path_name + ".population_index") != p_index:
                raise ValueError("JSONL indices must be complete and ordered")
            genes = _array(item["genes"], path_name + ".genes")
            for i, gene in enumerate(genes):
                _integer(gene, f"{path_name}.genes[{i}]")
            if len(genes) != boxes or len(set(genes)) != boxes:
                raise ValueError(path_name + ".genes must be a permutation of box IDs")
            if ids is None:
                ids = set(genes)
            elif set(genes) != ids:
                raise ValueError(path_name + ".genes differs from problem IDs")
            packing_result = _validate_trace_packing_result(
                item["packing_result"], genes, boxes, path_name + ".packing_result",
            )
            _number(item["fitness"], path_name + ".fitness", maximum=1)
            _enum(item["decoder"], ("BL", "PP"), path_name + ".decoder")
            if item["pp_input_state"] is not None:
                _validate_state(item["pp_input_state"], sorted(ids), path_name + ".pp_input_state")
            for key in ("placed_count", "failed_count", "placed_volume"):
                _integer(item[key], path_name + "." + key)
            _number(item["density"], path_name + ".density", maximum=1)
            if item["fitness"] != item["density"]:
                raise ValueError(path_name + " fitness and density differ")
            if (item["placed_count"], item["failed_count"], item["placed_volume"]) != (
                packing_result.placed_count, boxes - packing_result.placed_count, packing_result.placed_volume
            ):
                raise ValueError(path_name + " counts/volume differ from packing_result")
            expected_density = packing_result.placed_volume / (
                packing_result.container.W * packing_result.container.H * packing_result.container.D
            )
            if item["density"] != expected_density:
                raise ValueError(path_name + " density differs from packing_result")
            evaluations.append(item)
        return header, tuple(evaluations)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, InputFormatError):
            raise
        raise InputFormatError(f"{p}: {exc}") from exc


def _validate_trace_packing_result(
    value: Any, genes: list[Any], box_count: int, path: str,
) -> PackingResult:
    result = _object(value, ("format", "version", "container", "placements"), path)
    _enum(result["format"], ("packing-result",), path + ".format")
    if _integer(result["version"], path + ".version") != 1:
        raise ValueError(path + ".version must be 1")
    container_data = _object(result["container"], ("width", "height", "depth"), path + ".container")
    container = Container(
        _integer(container_data["width"], path + ".container.width", 1),
        _integer(container_data["height"], path + ".container.height", 1),
        _integer(container_data["depth"], path + ".container.depth", 1),
    )
    raw_placements = _array(result["placements"], path + ".placements")
    if len(raw_placements) != box_count:
        raise ValueError(path + ".placements count differs from box_count")
    placements = []
    seen: set[int] = set()
    for index, raw in enumerate(raw_placements):
        item_path = f"{path}.placements[{index}]"
        item = _object(raw, ("id", "size", "position"), item_path)
        box_id = _integer(item["id"], item_path + ".id")
        if box_id in seen or box_id != genes[index]:
            raise ValueError(path + ".placements IDs must be unique and follow genes order")
        seen.add(box_id)
        size = _object(item["size"], ("width", "height", "depth"), item_path + ".size")
        box = Box(
            box_id,
            _integer(size["width"], item_path + ".size.width", 1),
            _integer(size["height"], item_path + ".size.height", 1),
            _integer(size["depth"], item_path + ".size.depth", 1),
        )
        position_data = item["position"]
        if position_data is None:
            position = None
        else:
            position_data = _object(position_data, ("x", "y", "z"), item_path + ".position")
            position = Point(
                _integer(position_data["x"], item_path + ".position.x"),
                _integer(position_data["y"], item_path + ".position.y"),
                _integer(position_data["z"], item_path + ".position.z"),
            )
        placements.append(Placement(box, position))
    parsed = PackingResult(container, tuple(placements))
    errors = validate_packing_result(parsed)
    if errors:
        raise ValueError(path + " contains an illegal packing result: " + "; ".join(errors))
    return parsed
