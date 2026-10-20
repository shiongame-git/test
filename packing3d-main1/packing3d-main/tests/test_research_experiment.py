from __future__ import annotations

import csv
import hashlib
import json
import threading
from pathlib import Path
from uuid import uuid4

import pytest

from packing.errors import InputFormatError
from packing.research import ResearchExperiment, write_research_experiment_json
from packing import (
    BLPriority,
    Box,
    Container,
    DecoderConfig,
    GAConfig,
    GAExecutionConfig,
    RunOptions,
    WorkersRequest,
    parse_box_sizes_bytes,
    prepare_research_experiment,
    read_research_experiment,
    read_research_experiment_json,
    read_research_record,
    regenerate_research_experiment_csv,
    run_bl_ga,
    run_pp_ga,
    run_research_experiment,
)


def _problem():
    return Container(4, 3, 3), (Box(1, 1, 1, 1), Box(2, 2, 1, 3), Box(3, 3, 2, 2), Box(4, 3, 1, 3))


@pytest.mark.parametrize(
    "experiment_id",
    ("../escaped", "..", "not-a-uuid", "", "abcdefab-cdef-4abc-8abc-abcdefabcdef".upper(), 123),
)
def test_experiment_id_is_validated_before_creating_directories(tmp_path: Path, experiment_id):
    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1),
        start_seed=0, run_count=1,
    )
    root = tmp_path / "output"
    with pytest.raises((TypeError, ValueError), match="experiment_id"):
        run_research_experiment(plan, root, experiment_id=experiment_id)
    assert not root.exists()
    assert not (tmp_path / "escaped").exists()


def test_experiment_accepts_explicit_valid_uuid(tmp_path: Path):
    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1),
        start_seed=0, run_count=1,
    )
    stop = threading.Event()
    stop.set()
    identifier = str(uuid4())
    experiment = run_research_experiment(
        plan, tmp_path / "output", experiment_id=identifier, stop_event=stop,
    )
    assert experiment["experiment_id"] == identifier
    assert (tmp_path / "output" / identifier / "experiment.json").is_file()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("record_type", "policy"),
    (("BL-GA", "Cartesian"), ("PP-GA", "LocalNeighbors")),
)
def test_experiment_runs_100_fixed_seed_trials_and_direct_best_matches(
    tmp_path: Path, record_type: str, policy: str
):
    container, boxes = _problem()
    config = GAConfig(population_size=4, generation_count=3, crossover_rate=0.8, mutation_rate=0.1)
    decoder = DecoderConfig(candidate_policy=policy, min_support_percent=55)
    plan = prepare_research_experiment(
        container, boxes, record_type=record_type, config=config,
        priority=BLPriority.BOTTOM_FIRST, decoder_config=decoder, start_seed=17, run_count=100,
        workers_request=WorkersRequest("manual", 1), cache=False, save_level="summary", profile=False,
    )
    plan.fixed_snapshot["problem"]["container"][0] = 999
    plan.workers_selection["request"]["manual_workers"] = 999
    assert plan.fixed_snapshot["problem"]["container"][0] == 4
    assert plan.workers_selection["request"]["manual_workers"] == 1
    root = tmp_path / "out"
    experiment = run_research_experiment(plan, root)
    assert experiment["status"] == "completed"
    assert len(experiment["trials"]) == 100
    assert [trial["seed"] for trial in experiment["trials"]] == list(range(17, 117))
    assert len({trial["record_id"] for trial in experiment["trials"]}) == 100
    assert len({trial["record_sha256"] for trial in experiment["trials"]}) == 100
    directory = root / experiment["experiment_id"]
    assert read_research_experiment(directory) == experiment
    with (directory / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 100
    assert [int(row["run_index"]) for row in rows] == list(range(100))
    assert {row["status"] for row in rows} == {"SUCCESS"}

    runner = run_bl_ga if record_type == "BL-GA" else run_pp_ga
    for trial in experiment["trials"]:
        record_dir = directory / Path(trial["record_path"]).parent
        record = read_research_record(record_dir)
        direct = runner(
            container, boxes, priority=BLPriority.BOTTOM_FIRST, config=config, seed=trial["seed"],
            execution=GAExecutionConfig(workers=1, cache=False), decoder_config=decoder,
        )
        best = record["summary"]["best"]
        assert best["genes"] == list(direct.best.individual.genes)
        assert best["fitness"] == direct.best.fitness
        assert best["generation_index"] == direct.best_generation_index
        assert best["population_index"] == direct.best_population_index


def test_stop_request_finishes_current_record_and_preserves_remaining_plan(tmp_path: Path):
    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(4, 3), start_seed=41, run_count=5,
        workers_request=WorkersRequest("manual", 1),
    )
    stop = threading.Event()
    def progress(message):
        if message["kind"] == "trial_completed":
            stop.set()

    result = run_research_experiment(plan, tmp_path / "stopped", stop_event=stop, progress_callback=progress)
    assert result["status"] == "stopped"
    assert [trial["status"] for trial in result["trials"]] == ["completed", "planned", "planned", "planned", "planned"]
    directory = tmp_path / "stopped" / result["experiment_id"]
    assert read_research_experiment(directory) == result
    assert (directory / "runs" / "000" / "record.json").exists()
    assert not (directory / "runs" / "001").exists()
    with (directory / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["status"] for row in rows] == ["SUCCESS", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN"]
    csv_path = directory / "summary.csv"
    original_csv = csv_path.read_bytes()
    csv_path.unlink()
    assert regenerate_research_experiment_csv(directory) == csv_path
    assert csv_path.read_bytes() == original_csv


def test_auto_worker_selection_is_saved_with_original_request_and_resolved_execution(tmp_path: Path):
    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1), start_seed=51, run_count=1,
        workers_request=WorkersRequest("auto", None),
    )
    selection = plan.workers_selection
    assert selection["request"] == {"mode": "auto", "manual_workers": None}
    assert selection["resolved_execution"]["workers"]["requested_mode"] == "auto"
    assert selection["resolved_execution"]["workers"]["manual_workers"] is None
    assert selection["resolved_execution"]["workers"]["resolved_workers"] == plan.workers
    assert plan.fixed_snapshot["execution"]["workers_request"] == {
        "mode": "manual", "manual_workers": plan.workers,
    }

    stop = threading.Event()
    stop.set()
    result = run_research_experiment(plan, tmp_path / "auto", stop_event=stop)
    assert result["status"] == "stopped"
    directory = tmp_path / "auto" / result["experiment_id"]
    saved = read_research_experiment(directory)
    assert saved["workers_selection"] == selection
    assert saved["fixed_snapshot"]["execution"]["workers_request"] == {
        "mode": "manual", "manual_workers": plan.workers,
    }


def test_stop_before_first_trial_publishes_stopped_plan_without_creating_records(tmp_path: Path):
    raw = b"{4,3,3}{1,1,1}{2,1,3}{3,2,2}{3,1,3}"
    container, boxes = parse_box_sizes_bytes(raw, source="fixed.txt")
    options = RunOptions(origin="api", source="fixed.txt", fixture_sha256=hashlib.sha256(raw).hexdigest())
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1), start_seed=7, run_count=2,
        run_options=options, source_bytes=raw,
    )
    stop = threading.Event()
    stop.set()
    result = run_research_experiment(plan, tmp_path / "pre-stopped", stop_event=stop)
    assert result["status"] == "stopped"
    assert all(trial["status"] == "planned" for trial in result["trials"])
    directory = tmp_path / "pre-stopped" / result["experiment_id"]
    assert read_research_experiment(directory) == result
    source_artifact = next(item for item in result["input_artifacts"] if item["role"] == "source")
    assert source_artifact["path"] == "input/source.txt"
    assert (directory / source_artifact["path"]).read_bytes() == raw
    assert not (directory / "runs").exists()
    malformed = result.to_dict()
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        ResearchExperiment(malformed)
    malformed = result.to_dict()
    del malformed["workers_selection"]
    with pytest.raises(ValueError, match="missing"):
        ResearchExperiment(malformed)
    duplicate = tmp_path / "duplicate-experiment.json"
    duplicate.write_text(
        '{"format":"packing3d-research-experiment","format":"other"}', encoding="utf-8")
    with pytest.raises(InputFormatError, match="duplicate"):
        read_research_experiment_json(duplicate)


def test_experiment_reader_rejects_duplicate_keys_inside_problem_artifact(tmp_path: Path):
    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1), start_seed=11, run_count=1,
    )
    stop = threading.Event()
    stop.set()
    result = run_research_experiment(plan, tmp_path / "duplicate-problem", stop_event=stop)
    directory = tmp_path / "duplicate-problem" / result["experiment_id"]
    data = result.to_dict()
    descriptor = next(item for item in data["input_artifacts"] if item["role"] == "problem")
    artifact_path = directory / descriptor["path"]
    problem = json.loads(artifact_path.read_text(encoding="utf-8"))
    duplicate_text = '{"format":"other",' + json.dumps(problem, separators=(",", ":"))[1:]
    raw = duplicate_text.encode("utf-8")
    artifact_path.write_bytes(raw)
    descriptor["bytes"] = len(raw)
    descriptor["sha256"] = hashlib.sha256(raw).hexdigest()
    write_research_experiment_json(directory / "experiment.json", ResearchExperiment(data))

    with pytest.raises(InputFormatError, match="duplicate"):
        read_research_experiment(directory)


@pytest.mark.parametrize(
    ("experiment_status", "trial_statuses"),
    (("running", ("running", "validating", "planned")),
     ("failed", ("failed", "failed", "planned")),
     ("failed", ("planned", "failed", "planned"))),
)
def test_experiment_schema_rejects_multiple_nonterminal_trials(
    tmp_path: Path, experiment_status: str, trial_statuses: tuple[str, ...],
):
    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1), start_seed=17, run_count=3,
    )
    stop = threading.Event()
    stop.set()
    experiment = run_research_experiment(plan, tmp_path / "schema", stop_event=stop)
    data = experiment.to_dict()
    data["status"] = experiment_status
    if experiment_status == "failed":
        data["error"] = {"phase": "execution", "type": "RuntimeError", "message": "injected",
                          "generation_index": None, "population_index": None, "run_index": 1}
    else:
        data["timing"].update(finished_at=None, elapsed_seconds=None)
    for index, state in enumerate(trial_statuses):
        trial = data["trials"][index]
        trial["status"] = state
        if state == "planned":
            trial.update(record_id=None, record_path=None, record_sha256=None)
        else:
            trial.update(record_id=str(uuid4()), record_path=f"runs/{index:03d}/record.json")
    with pytest.raises(ValueError, match="in-flight or failed Trial"):
        ResearchExperiment(data)


def test_csv_failure_warns_but_does_not_stop_experiment(tmp_path: Path, monkeypatch):
    from packing import research_experiment as module

    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1), start_seed=3, run_count=2,
    )
    original = module._atomic_bytes
    def fail_csv(path, raw, *, replace=False):
        if Path(path).name == "summary.csv":
            raise OSError("injected CSV failure")
        return original(path, raw, replace=replace)
    monkeypatch.setattr(module, "_atomic_bytes", fail_csv)
    messages = []
    result = run_research_experiment(plan, tmp_path / "csv-fail", progress_callback=messages.append)
    assert result["status"] == "completed"
    assert all(trial["status"] == "completed" for trial in result["trials"])
    assert any(message["kind"] == "warning" for message in messages)


def test_ga_failure_persists_failed_child_and_does_not_start_next_trial(tmp_path: Path, monkeypatch):
    from packing import research_experiment as module

    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1), start_seed=100, run_count=3,
    )
    def fail_run(*_args, **_kwargs):
        raise RuntimeError("injected GA failure")
    monkeypatch.setattr(module, "run_bl_ga", fail_run)
    result = run_research_experiment(plan, tmp_path / "failed")
    assert result["status"] == "failed"
    assert [trial["status"] for trial in result["trials"]] == ["failed", "planned", "planned"]
    directory = tmp_path / "failed" / result["experiment_id"]
    failed_record = read_research_record(directory / "runs" / "000")
    assert failed_record["status"] == "failed"
    assert failed_record["error"]["phase"] == "execution"
    assert read_research_experiment(directory) == result
    assert not (directory / "runs" / "001").exists()


def test_later_preflight_failure_does_not_reuse_previous_trial_capture(tmp_path: Path, monkeypatch):
    import packing.research_experiment as module

    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1), start_seed=77, run_count=2,
    )
    original = module.make_provenance
    calls = 0

    def fail_second_provenance():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-trial preflight failure")
        return original()

    monkeypatch.setattr(module, "make_provenance", fail_second_provenance)
    experiment = run_research_experiment(plan, tmp_path / "capture-reset")

    assert experiment["status"] == "failed"
    assert [trial["status"] for trial in experiment["trials"]] == ["completed", "failed"]
    directory = tmp_path / "capture-reset" / experiment["experiment_id"]
    previous = read_research_record(directory / "runs" / "000")
    failed = read_research_record(directory / "runs" / "001")
    assert previous["ga_run_id"] is not None
    assert failed["ga_run_id"] is None
    assert failed["summary"] is None
    assert failed["timing"]["execution_seconds"] is None
    assert read_research_experiment(directory) == experiment


@pytest.mark.parametrize("failure", ("validation", "write", "replace", "trace"))
def test_trial_failure_preserves_plan_and_leaves_failed_csv_best_fields_empty(
    tmp_path: Path, monkeypatch, failure: str,
):
    import packing.research_records as records_module

    from packing.research import read_research_record_json

    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1), start_seed=200,
        run_count=2, save_level="full" if failure == "trace" else "summary",
    )
    if failure == "validation":
        monkeypatch.setattr(
            records_module, "_validate_ga_run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("injected validation failure")),
        )
    elif failure == "write":
        original = records_module._store_artifact

        def fail_artifact(directory, relative, raw):
            if relative == "best.result.json":
                raise OSError("injected artifact write failure")
            return original(directory, relative, raw)

        monkeypatch.setattr(records_module, "_store_artifact", fail_artifact)
    elif failure == "replace":
        original_replace = records_module.os.replace
        failed_once = False

        def fail_record_replace(source, destination):
            nonlocal failed_once
            if not failed_once and Path(destination).name == "record.json":
                failed_once = True
                raise OSError("injected atomic replace failure")
            return original_replace(source, destination)

        monkeypatch.setattr(records_module.os, "replace", fail_record_replace)
    else:
        original_trace = records_module._build_trace

        def corrupt_trace(run, record_id):
            lines = original_trace(run, record_id).decode("utf-8").splitlines()
            evaluation = json.loads(lines[1])
            evaluation["packing_result"]["container"]["width"] = True
            lines[1] = json.dumps(evaluation)
            return ("\n".join(lines) + "\n").encode("utf-8")

        monkeypatch.setattr(records_module, "_build_trace", corrupt_trace)

    experiment = run_research_experiment(plan, tmp_path / f"failed-{failure}")
    assert experiment["status"] == "failed"
    assert [trial["status"] for trial in experiment["trials"]] == ["failed", "planned"]
    directory = tmp_path / f"failed-{failure}" / experiment["experiment_id"]
    failed_record = read_research_record_json(directory / "runs" / "000" / "record.json")
    assert failed_record["status"] == "failed"
    assert failed_record["ga_run_id"] is not None
    assert not (directory / "runs" / "001").exists()
    with (directory / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "FAILED"
    for field in ("best_fitness", "density", "placed_count", "failed_count",
                  "best_generation", "best_population_index"):
        assert row[field] == ""
    if failure != "trace":
        assert read_research_experiment(directory) == experiment


def test_parent_reference_update_failure_keeps_child_and_stops_next_trial(tmp_path: Path, monkeypatch):
    from packing import research_experiment as module

    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1), start_seed=12, run_count=2,
    )
    original = module._write_experiment
    failed_once = False
    def fail_first_child_update(path, data, *, terminal=False):
        nonlocal failed_once
        if not failed_once and data["trials"][0]["status"] == "validating":
            failed_once = True
            raise OSError("injected parent reference failure")
        return original(path, data, terminal=terminal)
    monkeypatch.setattr(module, "_write_experiment", fail_first_child_update)
    result = run_research_experiment(plan, tmp_path / "parent-failure")
    assert result["status"] == "failed"
    assert result["trials"][0]["status"] == "failed"
    assert result["trials"][1]["status"] == "planned"
    directory = tmp_path / "parent-failure" / result["experiment_id"]
    assert (directory / "runs" / "000" / "record.json").exists()
    assert read_research_experiment(directory) == result


@pytest.mark.parametrize("stop_phase", ("execution", "validating", "saving"))
def test_stop_during_current_trial_finishes_save_then_stops(
    tmp_path: Path, monkeypatch, stop_phase: str,
):
    import packing.research_experiment as module

    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1), start_seed=301, run_count=2,
    )
    stop = threading.Event()
    if stop_phase == "execution":
        def on_progress(message):
            if message["kind"] == "progress":
                stop.set()
    else:
        original_save = module.save_research_record

        def save_with_stop(*args, **kwargs):
            original_callback = kwargs["status_callback"]

            def on_phase(phase):
                if phase == stop_phase:
                    stop.set()
                original_callback(phase)

            kwargs["status_callback"] = on_phase
            return original_save(*args, **kwargs)

        monkeypatch.setattr(module, "save_research_record", save_with_stop)
        on_progress = None
    result = run_research_experiment(
        plan, tmp_path / f"stop-{stop_phase}", stop_event=stop, progress_callback=on_progress,
    )
    assert result["status"] == "stopped"
    assert [trial["status"] for trial in result["trials"]] == ["completed", "planned"]


def test_stop_requested_during_final_trial_prefers_completed(tmp_path: Path):
    container, boxes = _problem()
    plan = prepare_research_experiment(
        container, boxes, record_type="BL-GA", config=GAConfig(2, 1), start_seed=401, run_count=1,
    )
    stop = threading.Event()
    def on_progress(message):
        if message["kind"] == "trial_completed":
            stop.set()
    result = run_research_experiment(
        plan, tmp_path / "final-stop", stop_event=stop, progress_callback=on_progress,
    )
    assert result["status"] == "completed"
    assert result["trials"][0]["status"] == "completed"
