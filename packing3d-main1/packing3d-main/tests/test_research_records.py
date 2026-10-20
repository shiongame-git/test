from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from uuid import uuid4

import pytest

from packing import (
    BLPriority,
    Box,
    Container,
    DecoderConfig,
    GAConfig,
    GAExecutionConfig,
    PPState,
    Point,
    pack_pp,
    pack,
    read_generations_jsonl,
    read_research_record,
    run_bl_ga,
    run_pp_ga,
)
from packing.errors import InputFormatError
from packing.model import PackingResult, Placement
from packing.research import ResearchRecord, read_research_record_json, write_research_record_json
from packing.research_records import (
    capture_ga_run,
    capture_single_run,
    save_research_record,
    snapshot_for_single,
    utc_now,
)


def _boxes() -> tuple[Box, ...]:
    return (Box(12, 1, 1, 1), Box(4, 1, 1, 1), Box(21, 1, 1, 1))


def _ga_record(tmp_path: Path, decoder: str, level: str = "full"):
    container = Container(3, 3, 3)
    boxes = _boxes()
    config = GAConfig(population_size=3, generation_count=3, crossover_rate=0.8, mutation_rate=0.1)
    decoder_config = DecoderConfig(candidate_policy="LocalNeighbors", min_support_percent=0)
    started = utc_now()
    clock = time.perf_counter()
    runner = run_bl_ga if decoder == "BL" else run_pp_ga
    run = runner(
        container,
        boxes,
        priority=BLPriority.BACK_FIRST,
        config=config,
        seed=17,
        execution=GAExecutionConfig(workers=1, cache=False),
        decoder_config=decoder_config,
    )
    capture = capture_ga_run(
        run,
        execution_started_at=started,
        execution_finished_at=utc_now(),
        call_wall_seconds=time.perf_counter() - clock,
    )
    record_id = str(uuid4())
    directory = tmp_path / "records" / record_id
    record = save_research_record(directory, capture, run=run, save_level=level, record_id=record_id)
    return run, directory, record


@pytest.mark.parametrize("decoder", ["BL", "PP"])
@pytest.mark.parametrize("level", ["summary", "full"])
def test_ga_record_round_trip_and_source_run_match(tmp_path: Path, decoder: str, level: str):
    run, directory, record = _ga_record(tmp_path, decoder, level)

    assert record["status"] == "completed"
    assert record["record_type"] == f"{decoder}-GA"
    assert record["summary"]["best"]["genes"] == list(run.best.individual.genes)
    assert record["validation"]["scope"] == ("all_evaluations" if level == "full" else "best")
    assert record["validation"]["source_run_match"] == ("passed" if level == "full" else "not_applicable")

    loaded = read_research_record(directory)
    assert loaded == record
    assert loaded["summary"] == record["summary"]
    if level == "full":
        trace = next(item for item in record["artifacts"] if item["role"] == "trace")
        header, rows = read_generations_jsonl(directory / trace["path"])
        assert header["record_id"] == record["record_id"]
        assert len(rows) == run.config.population_size * run.config.generation_count
        assert [row["decoder"] for row in rows[:3]] == ["BL"] * 3
        if decoder == "PP":
            assert [row["decoder"] for row in rows[3:6]] == ["PP"] * 3
            assert all(row["pp_input_state"] is not None for row in rows[3:])
        else:
            assert all(row["pp_input_state"] is None for row in rows)
        # Every evaluation, including repeated individuals, remains a distinct row.
        assert [(row["generation_index"], row["population_index"]) for row in rows] == [
            (g, p) for g in range(3) for p in range(3)
        ]


def test_single_pp_record_persists_explicit_state_and_result(tmp_path: Path):
    container = Container(3, 3, 3)
    boxes = (Box(3, 1, 1, 1), Box(9, 1, 1, 1))
    state = PPState({3: Point(0, 0, 0), 9: None})
    settings = DecoderConfig(min_support_percent=0)
    snapshot = snapshot_for_single(
        container,
        boxes,
        record_type="PP",
        priority=BLPriority.BOTTOM_FIRST,
        decoder_config=settings,
        pp_input_state=state,
    )
    started = utc_now()
    clock = time.perf_counter()
    result = pack_pp(container, boxes, state, decoder_config=settings)
    elapsed = time.perf_counter() - clock
    capture = capture_single_run(
        snapshot=snapshot,
        execution_started_at=started,
        execution_finished_at=utc_now(),
        call_wall_seconds=elapsed,
    )
    capture.snapshot["problem"]["container"][0] = 999
    assert capture.snapshot["problem"]["container"][0] == 3
    record_id = str(uuid4())
    directory = tmp_path / "records" / record_id
    record = save_research_record(directory, capture, result=result, record_id=record_id)

    assert record["snapshot"]["pp_input_state"] == [[3, [0, 0, 0]], [9, None]]
    assert record["snapshot"]["ga"] is None
    assert record["summary"]["placed_count"] == result.placed_count
    assert read_research_record(directory)["record_id"] == record_id


def test_single_snapshot_preserves_source_and_effective_box_order():
    source_boxes = (Box(1, 1, 1, 1), Box(2, 1, 1, 3))
    effective_boxes = (source_boxes[1], source_boxes[0])
    snapshot = snapshot_for_single(
        Container(3, 2, 3), effective_boxes, record_type="BL",
        priority=BLPriority.BOTTOM_FIRST, decoder_config=DecoderConfig(),
        order_mode="depth_desc_stable", source_boxes=source_boxes,
    )
    assert snapshot["problem"]["source_box_order"] == [1, 2]
    assert snapshot["problem"]["effective_box_order"] == [2, 1]


def test_independent_support_validation_rejects_incomplete_support(tmp_path: Path):
    container = Container(2, 2, 1)
    boxes = (Box(1, 1, 1, 1), Box(2, 2, 1, 1))
    settings = DecoderConfig(min_support_percent=55)
    snapshot = snapshot_for_single(
        container,
        boxes,
        record_type="BL",
        priority=BLPriority.BOTTOM_FIRST,
        decoder_config=settings,
    )
    result = PackingResult(
        container,
        (
            Placement(boxes[0], Point(0, 0, 0)),
            Placement(boxes[1], Point(0, 1, 0)),
        ),
    )
    started = utc_now()
    capture = capture_single_run(
        snapshot=snapshot,
        execution_started_at=started,
        execution_finished_at=utc_now(),
        call_wall_seconds=0.0,
    )
    record_id = str(uuid4())
    directory = tmp_path / "records" / record_id
    with pytest.raises(ValueError, match="support validation"):
        save_research_record(directory, capture, result=result, record_id=record_id)
    failed = read_research_record_json(directory / "record.json")
    assert failed["status"] == "failed"
    assert failed["error"]["phase"] == "validation"
    assert failed["summary"] is None


def test_independent_support_validator_detects_injected_decoder_support_fault(tmp_path: Path, monkeypatch):
    from packing import bl as decoder_module

    container = Container(2, 2, 1)
    boxes = (Box(1, 1, 1, 1), Box(2, 2, 1, 1))
    settings = DecoderConfig(min_support_percent=100)
    monkeypatch.setattr(decoder_module, "has_sufficient_support", lambda *_args, **_kwargs: True)
    result = pack(container, boxes, decoder_config=settings)
    assert result.placements[1].position == Point(0, 1, 0)
    snapshot = snapshot_for_single(
        container, boxes, record_type="BL", priority=BLPriority.BOTTOM_FIRST,
        decoder_config=settings,
    )
    capture = capture_single_run(
        snapshot=snapshot, execution_started_at=utc_now(), execution_finished_at=utc_now(),
        call_wall_seconds=0.0,
    )
    record_id = str(uuid4())
    with pytest.raises(ValueError, match="independent support validation"):
        save_research_record(tmp_path / "decoder-fault" / record_id, capture,
                             result=result, record_id=record_id)
    failed = read_research_record_json(tmp_path / "decoder-fault" / record_id / "record.json")
    assert failed["status"] == "failed"


def test_record_artifact_write_failure_does_not_publish_completed(tmp_path: Path, monkeypatch):
    import packing.research_records as records_module

    container = Container(2, 2, 2)
    run = run_bl_ga(container, (Box(1, 1, 1, 1),), config=GAConfig(1, 1))
    capture = capture_ga_run(run, execution_started_at=utc_now(), execution_finished_at=utc_now(),
                            call_wall_seconds=0.01)
    original = records_module._atomic_bytes
    def fail_best(path, raw, *, replace=False):
        if Path(path).name == "best.result.json":
            raise OSError("injected write failure")
        return original(path, raw, replace=replace)
    monkeypatch.setattr(records_module, "_atomic_bytes", fail_best)
    directory = tmp_path / "write-fault" / str(uuid4())
    with pytest.raises(OSError, match="injected write failure"):
        save_research_record(directory, capture, run=run, record_id=directory.name)
    failed = read_research_record_json(directory / "record.json")
    assert failed["status"] == "failed"
    assert failed["error"]["phase"] == "save"


def test_record_directory_is_exclusive_and_completed_artifact_tampering_is_detected(tmp_path: Path):
    run, directory, record = _ga_record(tmp_path, "BL", "summary")
    with pytest.raises(FileExistsError):
        save_research_record(directory, capture_ga_run(
            run,
            execution_started_at=utc_now(),
            execution_finished_at=utc_now(),
            call_wall_seconds=0.1,
        ), run=run, record_id=record["record_id"])

    result_artifact = next(item for item in record["artifacts"] if item["role"] == "result")
    artifact_path = directory / result_artifact["path"]
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
    with pytest.raises(InputFormatError, match="artifact hash/size mismatch"):
        read_research_record(directory)


def test_record_schema_rejects_unknown_missing_duplicate_and_invalid_nullable_fields(tmp_path: Path):
    _run, directory, record = _ga_record(tmp_path, "PP", "summary")
    data = record.to_dict()
    data["surprise"] = True
    with pytest.raises(ValueError, match="unknown"):
        ResearchRecord(data)

    data = record.to_dict()
    del data["snapshot"]["decoder"]["min_support_percent"]
    with pytest.raises(ValueError, match="missing"):
        ResearchRecord(data)

    path = tmp_path / "duplicate.json"
    path.write_text('{"format":"packing3d-research-record","format":"x"}', encoding="utf-8")
    with pytest.raises(InputFormatError, match="duplicate"):
        read_research_record_json(path)

    data = record.to_dict()
    data["seed"] = None
    with pytest.raises(ValueError, match="seed"):
        ResearchRecord(data)


def test_record_reader_rejects_duplicate_keys_inside_metadata_artifact(tmp_path: Path):
    _run, directory, record = _ga_record(tmp_path, "BL", "summary")
    data = record.to_dict()
    descriptor = next(item for item in data["artifacts"] if item["role"] == "metadata")
    artifact_path = directory / descriptor["path"]
    metadata = json.loads(artifact_path.read_text(encoding="utf-8"))
    duplicate_text = '{"format":"other",' + json.dumps(metadata, separators=(",", ":"))[1:]
    raw = duplicate_text.encode("utf-8")
    artifact_path.write_bytes(raw)
    descriptor["bytes"] = len(raw)
    descriptor["sha256"] = hashlib.sha256(raw).hexdigest()
    write_research_record_json(directory / "record.json", ResearchRecord(data))

    with pytest.raises(InputFormatError, match="duplicate"):
        read_research_record(directory)


def test_trace_reader_rejects_missing_or_duplicate_evaluation_indexes(tmp_path: Path):
    _run, directory, record = _ga_record(tmp_path, "BL", "full")
    artifact = next(item for item in record["artifacts"] if item["role"] == "trace")
    original = (directory / artifact["path"]).read_text(encoding="utf-8")
    lines = original.splitlines()
    first = json.loads(lines[1])
    first["population_index"] = 1
    lines[1] = json.dumps(first)
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(InputFormatError, match="indices"):
        read_generations_jsonl(invalid)


def test_trace_reader_rejects_invalid_nested_packing_result(tmp_path: Path):
    _run, directory, record = _ga_record(tmp_path, "BL", "full")
    artifact = next(item for item in record["artifacts"] if item["role"] == "trace")
    lines = (directory / artifact["path"]).read_text(encoding="utf-8").splitlines()
    evaluation = json.loads(lines[1])
    evaluation["packing_result"]["container"]["width"] = True
    lines[1] = json.dumps(evaluation)
    invalid = tmp_path / "invalid-result.jsonl"
    invalid.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(InputFormatError, match="packing_result.*Python int"):
        read_generations_jsonl(invalid)
