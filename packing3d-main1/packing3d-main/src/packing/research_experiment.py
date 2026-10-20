"""Sequential, GUI-independent Research Experiment lifecycle."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from ._decoder_config import DecoderConfig
from .bl import BLPriority
from .execution import GAExecutionConfig
from .ga import GAConfig, run_bl_ga
from .model import Box, Container
from .pp_ga import run_pp_ga
from .research import (
    ResearchExperiment,
    _loads_strict_json,
    _uuid,
    read_research_experiment_json,
    read_research_record_json,
    validate_research_experiment,
)
from .research_records import (
    ResearchCapture,
    _artifact,
    _atomic_bytes,
    _json_bytes,
    _safe_child,
    _sha256,
    capture_ga_run,
    make_provenance,
    read_research_record,
    save_research_record,
    snapshot_from_ga,
    validate_research_output_root,
    _validate_source_bytes,
)
from .run_context import RunOptions
from .workers import WorkersRequest, detect_cpu, observe_environment, resolve_workers


CSV_COLUMNS = (
    "run_index", "seed", "status", "record_id", "run_id", "best_fitness", "density",
    "placed_count", "failed_count", "best_generation", "best_population_index", "elapsed_time",
    "artifact_directory",
)


@dataclass(frozen=True, slots=True, init=False)
class ResearchExperimentPlan:
    """Preflighted immutable inputs, with Auto/Max resolved exactly once."""

    container: Container
    boxes: tuple[Box, ...]
    record_type: str
    config: GAConfig
    priority: BLPriority
    decoder_config: DecoderConfig
    start_seed: int
    run_count: int
    save_level: str
    profile: bool
    cache: bool
    _workers_selection_json: str
    _fixed_snapshot_json: str
    run_options: RunOptions
    source_bytes: bytes | None
    _provenance_json: str
    preflight_started_at: str
    preflight_started_clock: float

    def __init__(self, container, boxes, record_type, config, priority, decoder_config, start_seed,
                 run_count, save_level, profile, cache, workers_selection, fixed_snapshot,
                 run_options, source_bytes, provenance, preflight_started_at, preflight_started_clock):
        object.__setattr__(self, "container", container)
        object.__setattr__(self, "boxes", tuple(boxes))
        object.__setattr__(self, "record_type", record_type)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "decoder_config", decoder_config)
        object.__setattr__(self, "start_seed", start_seed)
        object.__setattr__(self, "run_count", run_count)
        object.__setattr__(self, "save_level", save_level)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "cache", cache)
        object.__setattr__(self, "_workers_selection_json", json.dumps(workers_selection, ensure_ascii=False, allow_nan=False))
        object.__setattr__(self, "_fixed_snapshot_json", json.dumps(fixed_snapshot, ensure_ascii=False, allow_nan=False))
        object.__setattr__(self, "run_options", run_options)
        object.__setattr__(self, "source_bytes", source_bytes)
        object.__setattr__(self, "_provenance_json", json.dumps(provenance, ensure_ascii=False, allow_nan=False))
        object.__setattr__(self, "preflight_started_at", preflight_started_at)
        object.__setattr__(self, "preflight_started_clock", preflight_started_clock)

    @property
    def workers_selection(self) -> dict[str, Any]:
        return json.loads(self._workers_selection_json)

    @property
    def fixed_snapshot(self) -> dict[str, Any]:
        return json.loads(self._fixed_snapshot_json)

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(self._provenance_json)

    @property
    def workers(self) -> int:
        return self.fixed_snapshot["execution"]["workers"]


def prepare_research_experiment(
    container: Container,
    boxes: tuple[Box, ...],
    *,
    record_type: str,
    config: GAConfig,
    priority: BLPriority = BLPriority.BOTTOM_FIRST,
    decoder_config: DecoderConfig | None = None,
    start_seed: int,
    run_count: int,
    workers_request: WorkersRequest = WorkersRequest("manual", 1),
    cache: bool = False,
    save_level: str = "summary",
    profile: bool = False,
    run_options: RunOptions | None = None,
    source_bytes: bytes | None = None,
    cpu_reader: Callable = detect_cpu,
) -> ResearchExperimentPlan:
    """Validate all fixed inputs and resolve a single Manual worker count."""
    from ._decoder_config import resolve_decoder_config
    from ._io_integer import require_public_io_int
    from .workers import preflight_workers

    preflight_started_at = _utc_now()
    preflight_started_clock = time.perf_counter()
    if type(container) is not Container or type(boxes) is not tuple or any(type(box) is not Box for box in boxes):
        raise TypeError("container and boxes must be resolved core values")
    if record_type not in ("BL-GA", "PP-GA"):
        raise ValueError("Research Experiment supports BL-GA or PP-GA")
    if type(config) is not GAConfig or type(priority) is not BLPriority:
        raise TypeError("config and priority must be resolved core values")
    if type(workers_request) is not WorkersRequest:
        raise TypeError("workers_request must be WorkersRequest")
    if type(cache) is not bool or type(profile) is not bool:
        raise TypeError("cache and profile must be bool")
    if save_level not in ("summary", "full"):
        raise ValueError("save_level must be summary or full")
    require_public_io_int(start_seed, name="start_seed")
    require_public_io_int(run_count, name="run_count")
    if run_count < 1:
        raise ValueError("run_count must be >= 1")
    require_public_io_int(start_seed + run_count - 1, name="last seed")
    options = RunOptions() if run_options is None else run_options
    if type(options) is not RunOptions:
        raise TypeError("run_options must be RunOptions")
    if source_bytes is not None and type(source_bytes) is not bytes:
        raise TypeError("source_bytes must be bytes or None")
    decoder = resolve_decoder_config(decoder_config)

    environment = observe_environment(implementation_version=options.implementation_version,
                                      revision=options.revision, cpu_reader=cpu_reader)
    available = environment["cpu"]["detection"]["available_logical_estimate"]
    workers = resolve_workers(workers_request, available, config.population_size,
                              environment["executor"]["worker_limit"])
    resolved_count = workers["resolved_workers"]
    preflight_workers(resolved_count)
    resolved_execution = {
        "workers": workers,
        "cache": cache,
        "start_method": None if resolved_count == 1 else "spawn",
    }
    source_boxes = boxes if options.source_boxes is None else options.source_boxes
    if type(source_boxes) is not tuple or {b.id: b for b in source_boxes} != {b.id: b for b in boxes}:
        raise ValueError("run_options.source_boxes must describe the same boxes")
    if options.order_mode not in ("source", "depth_desc_stable"):
        raise ValueError("unknown order mode")
    problem = {
        "container": [container.W, container.H, container.D],
        "source_boxes": [[b.id, b.w, b.h, b.d] for b in source_boxes],
        "source_box_order": [b.id for b in source_boxes],
        "effective_box_order": [b.id for b in boxes],
        "order_mode": options.order_mode,
        "source": options.source,
        "fixture_sha256": options.fixture_sha256,
    }
    if source_bytes is not None and (
        options.fixture_sha256 is None or hashlib.sha256(source_bytes).hexdigest() != options.fixture_sha256
    ):
        raise ValueError("captured source bytes do not match the problem fixture hash")
    if source_bytes is not None:
        _validate_source_bytes(source_bytes, problem)
    snapshot = {
        "problem": problem,
        "decoder": {
            "name": "BL" if record_type == "BL-GA" else "PP",
            "priority": priority.value,
            "reference_version": "BL-ref-v2" if record_type == "BL-GA" else "PP-ref-v2",
            "fitness_version": "density-v1",
            "pp_initialization": None if record_type == "BL-GA" else "bl_first_generation",
            "candidate_policy": decoder.candidate_policy,
            "candidate_policy_version": decoder.candidate_policy_version,
            "min_support_percent": decoder.min_support_percent,
            "support_model_version": decoder.support_model_version,
        },
        "ga": {
            "population_size": config.population_size,
            "generation_count": config.generation_count,
            "crossover_rate": config.crossover_rate,
            "mutation_rate": config.mutation_rate,
        },
        "execution": {
            "workers_request": {"mode": "manual", "manual_workers": resolved_count},
            "workers": resolved_count,
            "cache": cache,
            "start_method": resolved_execution["start_method"],
        },
        "profile": profile,
        "pp_input_state": None,
    }
    fixed_options = RunOptions(
        request=WorkersRequest("manual", resolved_count),
        origin=options.origin,
        replay_of=None,
        source_boxes=source_boxes,
        order_mode=options.order_mode,
        source=options.source,
        fixture_sha256=options.fixture_sha256,
        implementation_version=options.implementation_version,
        revision=options.revision,
        cpu_reader=cpu_reader,
    )
    return ResearchExperimentPlan(
        container, boxes, record_type, config, priority, decoder, start_seed, run_count,
        save_level, profile, cache,
        {"request": {"mode": workers_request.mode, "manual_workers": workers_request.manual_workers},
         "resolved_execution": resolved_execution, "environment": environment},
        snapshot, fixed_options, source_bytes, make_provenance(environment),
        preflight_started_at, preflight_started_clock,
    )


def _experiment_base(plan: ResearchExperimentPlan, experiment_id: str) -> dict[str, Any]:
    width = max(3, len(str(plan.run_count - 1)))
    return {
        "format": "packing3d-research-experiment", "version": 1,
        "experiment_id": experiment_id, "status": "planned", "record_type": plan.record_type,
        "save_level": plan.save_level, "run_count": plan.run_count, "start_seed": plan.start_seed,
        "index_width": width, "fixed_snapshot": plan.fixed_snapshot,
        "workers_selection": plan.workers_selection, "provenance": plan.provenance,
        "timing": {"started_at": plan.preflight_started_at, "finished_at": None, "elapsed_seconds": None},
        "input_artifacts": [],
        "trials": [
            {"run_index": index, "seed": plan.start_seed + index, "record_id": None,
             "record_path": None, "record_sha256": None, "status": "planned"}
            for index in range(plan.run_count)
        ],
        "error": None,
    }


def _utc_now() -> str:
    from .research_records import utc_now
    return utc_now()


def _write_experiment(path: Path, data: dict[str, Any], *, terminal: bool = False) -> None:
    if path.exists():
        current = read_research_experiment_json(path)
        if current["status"] in ("completed", "failed", "stopped"):
            raise ValueError("terminal Experiment is immutable")
    validate_research_experiment(data)
    _atomic_bytes(path, _json_bytes(data), replace=path.exists())


def _csv_bytes(data: dict[str, Any]) -> bytes:
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for trial in data["trials"]:
        record_dir = f"runs/{trial['run_index']:0{data['index_width']}d}"
        row = {key: "" for key in CSV_COLUMNS}
        row.update(run_index=trial["run_index"], seed=trial["seed"],
                   status={"completed": "SUCCESS", "failed": "FAILED", "planned": "NOT_RUN"}.get(
                       trial["status"], "IN_PROGRESS"),
                   record_id=trial["record_id"], artifact_directory=record_dir)
        if trial["record_path"] is not None:
            path = Path(trial["record_path"]).parent / "record.json"
            # The relative path is fixed by the schema; load the authoritative child below.
            record_file = data.get("_base_path")
            if record_file is not None:
                child_path = Path(record_file).parent / path
                try:
                    record = read_research_record_json(child_path)
                    meta = record.to_dict()
                    if meta["ga_run_id"] is not None:
                        row["run_id"] = meta["ga_run_id"]
                    if meta["timing"]["execution_seconds"] is not None:
                        row["elapsed_time"] = meta["timing"]["execution_seconds"]
                    summary = meta["summary"]
                    if trial["status"] == "completed" and summary is not None:
                        best = summary.get("best", summary)
                        row.update(best_fitness=best.get("fitness"),
                                   density=best.get("density"), placed_count=best.get("placed_count"),
                                   failed_count=best.get("failed_count"),
                                   best_generation=best.get("generation_index"),
                                   best_population_index=best.get("population_index"))
                except (OSError, TypeError, ValueError):
                    pass
        writer.writerow(row)
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def _update_csv(directory: Path, data: dict[str, Any]) -> bool:
    data["_base_path"] = str(directory / "experiment.json")
    try:
        raw = _csv_bytes(data)
        target = directory / "summary.csv"
        _atomic_bytes(target, raw, replace=target.exists())
    except (OSError, csv.Error, TypeError, ValueError):
        # CSV is derived; a warning does not invalidate JSON research evidence.
        return False
    finally:
        data.pop("_base_path", None)
    return True


def _problem_bytes(problem: dict[str, Any]) -> bytes:
    return _json_bytes({"format": "packing3d-research-problem", "version": 1, "problem": problem})


def _write_failed_record(
    destination: Path,
    plan: ResearchExperimentPlan,
    experiment_id: str,
    index: int,
    record_id: str,
    capture: ResearchCapture,
    exc: BaseException,
    phase: str,
) -> str | None:
    from .research_records import _record_base, _record_error, _write_record_atomic

    destination.mkdir(parents=True, exist_ok=False)
    record = _record_base(record_id, plan.record_type, plan.fixed_snapshot, capture,
                          experiment_id=experiment_id, run_index=index, seed=plan.start_seed + index)
    record["save_level"] = plan.save_level
    record["status"] = "failed"
    record["error"] = _record_error(exc, phase, index)
    record["timing"]["execution_finished_at"] = capture.execution_finished_at
    try:
        _write_record_atomic(destination / "record.json", record)
        return _sha256((destination / "record.json").read_bytes())
    except (OSError, TypeError, ValueError):
        return None


def _update_trial_phase(phase: str, trial: dict[str, Any], data: dict[str, Any], experiment_path: Path) -> None:
    if phase not in ("validating", "saving"):
        raise ValueError("unsupported Record phase")
    trial["status"] = phase
    _write_experiment(experiment_path, data)


def run_research_experiment(
    plan: ResearchExperimentPlan,
    output_root: str | os.PathLike[str],
    *,
    stop_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    experiment_id: str | None = None,
) -> ResearchExperiment:
    """Run planned trials sequentially and publish each child before continuing."""
    if type(plan) is not ResearchExperimentPlan:
        raise TypeError("plan must be created by prepare_research_experiment")
    if stop_event is not None and not isinstance(stop_event, threading.Event):
        raise TypeError("stop_event must be threading.Event or None")
    experiment_id = str(uuid4()) if experiment_id is None else _uuid(experiment_id, "experiment_id")
    root = validate_research_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / experiment_id
    directory.mkdir(exist_ok=False)
    (directory / "input").mkdir()
    data = _experiment_base(plan, experiment_id)
    started_clock = plan.preflight_started_clock
    warning = lambda exc: progress_callback and progress_callback(
        {"kind": "warning", "message": str(exc) or type(exc).__name__})
    problem_raw = _problem_bytes(plan.fixed_snapshot["problem"])
    problem_path = directory / "input" / "problem.json"
    experiment_path = directory / "experiment.json"
    current_index: int | None = None
    try:
        _atomic_bytes(problem_path, problem_raw)
        data["input_artifacts"].append(_artifact(
            "problem", "input/problem.json", "packing3d-research-problem", 1, problem_raw))
        if plan.source_bytes is not None:
            expected = plan.fixed_snapshot["problem"]["fixture_sha256"]
            if expected is None or _sha256(plan.source_bytes) != expected:
                raise ValueError("captured source bytes do not match fixed snapshot hash")
            _atomic_bytes(directory / "input" / "source.txt", plan.source_bytes)
            data["input_artifacts"].append(_artifact("source", "input/source.txt", "raw-bytes", None,
                                                     plan.source_bytes))
        _write_experiment(experiment_path, data)
        if stop_event is not None and stop_event.is_set():
            data["status"] = "stopped"
            data["timing"].update(finished_at=_utc_now(), elapsed_seconds=time.perf_counter() - started_clock)
            _write_experiment(experiment_path, data, terminal=True)
            if not _update_csv(directory, data):
                warning("CSV書込みエラー")
            if progress_callback is not None:
                progress_callback({"kind": "stopped", "directory": str(directory)})
            return ResearchExperiment(data)
        data["status"] = "running"
        _write_experiment(experiment_path, data)
        if not _update_csv(directory, data):
            warning("CSV書込みエラー")

        stopping_written = False
        for index, trial in enumerate(data["trials"]):
            current_index = index
            if stop_event is not None and stop_event.is_set():
                data["status"] = "stopping"
                _write_experiment(experiment_path, data)
                stopping_written = True
                break
            record_id = str(uuid4())
            relative_dir = Path("runs") / f"{index:0{data['index_width']}d}"
            trial.update(record_id=record_id, record_path=(relative_dir / "record.json").as_posix(), status="running")
            _write_experiment(experiment_path, data)
            trial_directory = directory / relative_dir
            execution_started_at = _utc_now()
            execution_provenance = plan.provenance
            call_started = None
            capture = None
            try:
                execution_provenance = make_provenance()
                runner = run_bl_ga if plan.record_type == "BL-GA" else run_pp_ga

                def on_evaluation(progress) -> None:
                    nonlocal stopping_written
                    if progress_callback is not None:
                        if stop_event is not None and stop_event.is_set() and not stopping_written:
                            data["status"] = "stopping"
                            _write_experiment(experiment_path, data)
                            stopping_written = True
                        progress_callback({
                            "kind": "progress", "run_index": index,
                            "generation_index": progress.generation_index,
                            "generation_count": progress.generation_count,
                            "population_index": progress.population_index,
                            "population_size": progress.population_size,
                            "decoder": progress.decoder_name,
                        })

                call_started = time.perf_counter()
                run = runner(
                    plan.container, plan.boxes, priority=plan.priority, config=plan.config,
                    seed=trial["seed"], execution=GAExecutionConfig(plan.workers, plan.cache),
                    decoder_config=plan.decoder_config, run_options=plan.run_options, profile=plan.profile,
                    progress_callback=on_evaluation if progress_callback is not None else None,
                )
                if snapshot_from_ga(run) != plan.fixed_snapshot:
                    raise ValueError("executed run differs from the fixed Experiment snapshot")
                capture = capture_ga_run(
                    run, execution_started_at=execution_started_at, execution_finished_at=_utc_now(),
                    call_wall_seconds=time.perf_counter() - call_started, source_bytes=plan.source_bytes,
                    execution_provenance=execution_provenance,
                )
                record = save_research_record(
                    trial_directory, capture, run=run, save_level=plan.save_level,
                    experiment_id=experiment_id, run_index=index, record_id=record_id,
                    status_callback=lambda phase: _update_trial_phase(
                        phase, trial, data, experiment_path,
                    ),
                )
                trial.update(status="completed", record_sha256=_sha256((trial_directory / "record.json").read_bytes()))
            except Exception as exc:
                child_error = None
                try:
                    child_path = trial_directory / "record.json"
                    if child_path.exists():
                        failed_record = read_research_record_json(child_path)
                        trial["status"] = failed_record["status"]
                        trial["record_sha256"] = _sha256(child_path.read_bytes()) if failed_record["status"] in ("completed", "failed") else None
                        child_error = failed_record["error"]
                    else:
                        if capture is None:
                            elapsed_call = 0.0 if call_started is None else time.perf_counter() - call_started
                            capture = ResearchCapture(
                                plan.fixed_snapshot, execution_provenance, execution_started_at, _utc_now(),
                                elapsed_call, None, plan.source_bytes)
                        trial["record_sha256"] = _write_failed_record(
                            trial_directory, plan, experiment_id, index, record_id, capture, exc,
                            "execution")
                        trial["status"] = "failed"
                except Exception:
                    trial["status"] = "failed"
                    trial["record_sha256"] = None
                data["status"] = "failed"
                data["error"] = {"phase": child_error["phase"] if child_error is not None else (
                                      "execution" if trial["record_sha256"] else "save"),
                                  "type": type(exc).__name__, "message": (str(exc) or type(exc).__name__)[:2048],
                                  "generation_index": getattr(exc, "generation_index", None),
                                  "population_index": getattr(exc, "population_index", None), "run_index": index}
                data["timing"].update(finished_at=_utc_now(), elapsed_seconds=time.perf_counter() - started_clock)
                try:
                    _write_experiment(experiment_path, data, terminal=True)
                finally:
                    if not _update_csv(directory, data):
                        warning("CSV書込みエラー")
                if progress_callback is not None:
                    progress_callback({"kind": "failed", "run_index": index, "error": str(exc)})
                return ResearchExperiment(data)

            _write_experiment(experiment_path, data)
            if not _update_csv(directory, data):
                warning("CSV書込みエラー")
            if progress_callback is not None:
                summary = record["summary"]["best"]
                progress_callback({"kind": "trial_completed", "run_index": index,
                                   "seed": trial["seed"], "record_id": record_id,
                                   "run_id": record["ga_run_id"], "fitness": summary["fitness"]})

        all_done = all(trial["status"] == "completed" for trial in data["trials"])
        if all_done:
            final_status = "completed"
        elif stop_event is not None and stop_event.is_set():
            data["status"] = "stopping"
            _write_experiment(experiment_path, data)
            final_status = "stopped"
        else:
            # An unexpected loop exit is a durable failure, never a false stop.
            final_status = "failed"
            data["error"] = {"phase": "execution", "type": "RuntimeError",
                             "message": "experiment ended before all trials completed", "generation_index": None,
                             "population_index": None, "run_index": None}
        data["status"] = final_status
        data["timing"].update(finished_at=_utc_now(), elapsed_seconds=time.perf_counter() - started_clock)
        _write_experiment(experiment_path, data, terminal=True)
        if not _update_csv(directory, data):
            warning("CSV書込みエラー")
        if progress_callback is not None:
            progress_callback({"kind": final_status, "directory": str(directory)})
        return ResearchExperiment(data)
    except Exception as exc:
        # A parent-manifest failure never rolls back already-published child Records.
        data["status"] = "failed"
        data["error"] = {"phase": "save", "type": type(exc).__name__,
                         "message": (str(exc) or type(exc).__name__)[:2048],
                         "generation_index": getattr(exc, "generation_index", None),
                         "population_index": getattr(exc, "population_index", None),
                         "run_index": current_index}
        data["timing"].update(finished_at=_utc_now(), elapsed_seconds=time.perf_counter() - started_clock)
        try:
            _write_experiment(experiment_path, data)
        except Exception:
            pass
        raise


def read_research_experiment(directory: str | os.PathLike[str], *, verify_artifacts: bool = True) -> ResearchExperiment:
    root = Path(directory)
    experiment_path = root / "experiment.json"
    experiment = read_research_experiment_json(experiment_path)
    if not verify_artifacts:
        return experiment
    data = experiment.to_dict()
    try:
        for artifact in data["input_artifacts"]:
            target = _safe_child(root, artifact["path"])
            raw = target.read_bytes()
            if len(raw) != artifact["bytes"] or _sha256(raw) != artifact["sha256"]:
                raise ValueError(f"Experiment artifact hash/size mismatch: {artifact['path']}")
            if artifact["role"] == "problem":
                payload = _loads_strict_json(raw.decode("utf-8-sig"))
                if payload != {"format": "packing3d-research-problem", "version": 1,
                               "problem": data["fixed_snapshot"]["problem"]}:
                    raise ValueError("Experiment problem artifact differs from fixed snapshot")
            elif artifact["role"] == "source":
                expected = data["fixed_snapshot"]["problem"]["fixture_sha256"]
                if expected is None or _sha256(raw) != expected:
                    raise ValueError("Experiment source bytes differ from fixed snapshot hash")
                _validate_source_bytes(raw, data["fixed_snapshot"]["problem"])
        ids: set[str] = set()
        ga_run_ids: set[str] = set()
        for trial in data["trials"]:
            if trial["record_id"] is None:
                continue
            child_path = _safe_child(root, trial["record_path"])
            if not child_path.exists():
                if data["status"] == "completed" or trial["status"] == "completed":
                    raise ValueError(f"Experiment is missing Record at {trial['record_path']}")
                continue
            raw = child_path.read_bytes()
            record = read_research_record(child_path.parent)
            if record["record_id"] in ids or record["record_id"] != trial["record_id"]:
                raise ValueError("Experiment contains a duplicate/mismatched Record ID")
            ids.add(record["record_id"])
            if record["experiment_id"] != data["experiment_id"] or record["run_index"] != trial["run_index"]:
                raise ValueError("Record membership differs from Experiment plan")
            if record["record_type"] != data["record_type"] or record["save_level"] != data["save_level"]:
                raise ValueError("Record type/save level differs from Experiment plan")
            if record["seed"] != trial["seed"] or record["snapshot"] != data["fixed_snapshot"]:
                raise ValueError("Record seed/snapshot differs from Experiment plan")
            if trial["record_sha256"] is not None and _sha256(raw) != trial["record_sha256"]:
                raise ValueError("Record hash differs from Experiment trial reference")
            if record["status"] != trial["status"]:
                raise ValueError("Record terminal status differs from Experiment trial")
            if record["ga_run_id"] is not None:
                if record["ga_run_id"] in ga_run_ids:
                    raise ValueError("Experiment reuses a GA run ID")
                ga_run_ids.add(record["ga_run_id"])
        return experiment
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        from .errors import InputFormatError
        if isinstance(exc, InputFormatError):
            raise
        raise InputFormatError(f"{experiment_path}: invalid Experiment bundle: {exc}") from exc


def regenerate_research_experiment_csv(directory: str | os.PathLike[str]) -> Path:
    """Rebuild the derived CSV from the immutable Experiment/Record JSON files."""
    root = Path(directory)
    experiment = read_research_experiment(root)
    data = experiment.to_dict()
    if not _update_csv(root, data):
        raise OSError(f"could not regenerate derived CSV at {root / 'summary.csv'}")
    return root / "summary.csv"
