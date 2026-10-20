from __future__ import annotations

import json

import pytest

from packing import (
    BLPriority,
    Box,
    Container,
    DecoderConfig,
    GAConfig,
    GAExecutionConfig,
    Individual,
    Placement,
    Point,
    PPState,
    find_bl_point,
    pack,
    pack_pp,
    read_decoder_profile_json,
    run_bl_ga,
    run_pp_ga,
    write_decoder_profile_json,
)
from packing.execution import EvaluationRuntime
from packing.ga import EvaluationStats, evaluate_bl_individual
from packing.profile import DecoderProfileCollector


@pytest.mark.parametrize(
    ("candidate_policy", "raw", "bounds_eligible", "visited"),
    (("Cartesian", 9, 3, 3), ("LocalNeighbors", 5, 3, 5)),
)
def test_candidate_metrics_match_raw_bounds_and_actual_visits(
    candidate_policy: str, raw: int, bounds_eligible: int, visited: int
) -> None:
    profile = DecoderProfileCollector()
    pack(
        Container(2, 2, 2),
        (Box(0, 2, 1, 2), Box(1, 2, 1, 2)),
        decoder_config=DecoderConfig(candidate_policy=candidate_policy, min_support_percent=55),
        _profile=profile,
    )

    metrics = profile.snapshot()
    assert metrics.raw_candidates == raw
    assert metrics.bounds_eligible_candidates == bounds_eligible
    assert metrics.visited_candidates == visited
    assert metrics.support_searches == 1
    assert metrics.support_intersection_rectangles == 1


@pytest.mark.parametrize(
    ("candidate_policy", "successful", "expected_raw"),
    (
        ("Cartesian", (), 1),
        (
            "Cartesian",
            (Placement(Box(1, 2, 2, 2), Point(0, 0, 0)),),
            8,
        ),
        ("LocalNeighbors", (), 1),
        (
            "LocalNeighbors",
            (Placement(Box(1, 2, 2, 2), Point(0, 0, 0)),),
            4,
        ),
    ),
)
def test_oversized_box_profiles_raw_candidates_before_bounds_failure(
    candidate_policy: str,
    successful: tuple[Placement, ...],
    expected_raw: int,
) -> None:
    profile = DecoderProfileCollector()
    point = find_bl_point(
        Container(4, 4, 4),
        Box(9, 5, 1, 1),
        successful,
        decoder_config=DecoderConfig(candidate_policy=candidate_policy),
        _profile=profile,
    )

    assert point is None
    metrics = profile.snapshot()
    assert metrics.raw_candidates == expected_raw
    assert metrics.bounds_eligible_candidates == 0
    assert metrics.visited_candidates == 0


@pytest.mark.parametrize("candidate_policy", ("Cartesian", "LocalNeighbors"))
@pytest.mark.parametrize("runner", (run_bl_ga, run_pp_ga), ids=("BL-GA", "PP-GA"))
def test_profile_toggle_preserves_every_generation(
    candidate_policy: str, runner
) -> None:
    container = Container(2, 2, 2)
    boxes = (Box(0, 2, 1, 2), Box(1, 2, 1, 2), Box(2, 2, 1, 2))
    config = GAConfig(population_size=4, generation_count=3, crossover_rate=0.8, mutation_rate=0.1)
    settings = DecoderConfig(candidate_policy=candidate_policy, min_support_percent=55)

    plain = runner(
        container,
        boxes,
        priority=BLPriority.BOTTOM_FIRST,
        config=config,
        seed=7,
        decoder_config=settings,
    )
    measured = runner(
        container,
        boxes,
        priority=BLPriority.BOTTOM_FIRST,
        config=config,
        seed=7,
        decoder_config=settings,
        profile=True,
    )

    assert measured == plain
    assert plain.profile is None
    assert measured.profile is not None
    assert measured.profile.run_id == measured.run_metadata["run_id"]
    assert measured.profile.decoder_config == settings
    assert measured.profile.requested_evaluations == 12
    assert measured.profile.computed_evaluations == 12
    assert measured.profile.cache_hits == 0
    metrics = measured.profile.metrics
    assert metrics.raw_candidates >= metrics.bounds_eligible_candidates
    assert metrics.visited_candidates <= metrics.raw_candidates
    assert metrics.visited_candidates > 0
    assert metrics.aabb_checks == metrics.broad_phase_collision_candidates
    assert metrics.support_searches > 0
    assert metrics.support_intersection_rectangles > 0
    assert metrics.candidate_construction_seconds >= 0.0
    assert metrics.common_legality_seconds >= 0.0
    assert metrics.support_union_seconds >= 0.0

    if runner is run_pp_ga:
        assert metrics.pp_inherited_accepted + metrics.pp_deferred_overlap + metrics.pp_deferred_support > 0
        assert metrics.pp_repack_successes + metrics.pp_repack_failures > 0
    else:
        assert metrics.pp_inherited_accepted == 0
        assert metrics.pp_deferred_overlap == 0
        assert metrics.pp_deferred_support == 0
        assert metrics.pp_repack_successes == 0
        assert metrics.pp_repack_failures == 0


@pytest.mark.parametrize("workers", (1, 2))
def test_profile_counts_only_unique_computed_cached_evaluations(workers: int) -> None:
    container = Container(2, 2, 2)
    boxes = (Box(0, 2, 1, 2), Box(1, 2, 1, 2))
    individual = Individual((0, 1))
    population = (individual,) * 4
    settings = DecoderConfig(min_support_percent=55)
    runtime = EvaluationRuntime(
        GAExecutionConfig(workers=workers, cache=True), profile_enabled=True
    )

    with runtime:
        results = runtime.evaluate_bl(
            container,
            boxes,
            population,
            priority=BLPriority.BOTTOM_FIRST,
            generation_index=0,
            decoder_config=settings,
        )
    assert results[0] == results[1] == results[2] == results[3]
    assert runtime.stats == EvaluationStats(requested=4, computed=1, cache_hits=3)

    expected = DecoderProfileCollector()
    evaluate_bl_individual(
        container,
        boxes,
        individual,
        priority=BLPriority.BOTTOM_FIRST,
        decoder_config=settings,
        _profile=expected,
    )
    assert runtime.profile_metrics is not None
    measured_values = runtime.profile_metrics.to_dict()
    expected_values = expected.snapshot().to_dict()
    for key in measured_values.keys() - {
        "support_union_seconds",
        "candidate_construction_seconds",
        "common_legality_seconds",
    }:
        assert measured_values[key] == expected_values[key]


def test_pp_profile_counts_distinct_inherited_deferral_reasons() -> None:
    container = Container(6, 3, 2)
    boxes = (Box(0, 2, 1, 2), Box(1, 4, 1, 2))
    state = PPState(((0, Point(0, 0, 0)), (1, Point(0, 1, 0))))
    profile = DecoderProfileCollector()

    result = pack_pp(
        container,
        boxes,
        state,
        decoder_config=DecoderConfig(min_support_percent=55),
        _profile=profile,
    )

    assert result.placements[0].position == state[0]
    assert result.placements[1].position is not None
    metrics = profile.snapshot()
    assert metrics.pp_inherited_accepted == 1
    assert metrics.pp_deferred_overlap == 0
    assert metrics.pp_deferred_support == 1
    assert metrics.pp_repack_successes == 1
    assert metrics.pp_repack_failures == 0


def test_pp_profile_does_not_overlap_count_none_or_support_deferrals() -> None:
    container = Container(2, 2, 2)
    boxes = (Box(0, 2, 1, 2), Box(1, 2, 1, 2))
    state = PPState(((0, Point(0, 0, 0)), (1, None)))
    profile = DecoderProfileCollector()

    pack_pp(container, boxes, state, _profile=profile)

    metrics = profile.snapshot()
    assert metrics.pp_inherited_accepted == 1
    assert metrics.pp_deferred_overlap == 0
    assert metrics.pp_deferred_support == 0
    assert metrics.pp_repack_successes == 1
    assert metrics.pp_repack_failures == 0


def test_profile_json_round_trip_is_separate_from_run_metadata(tmp_path) -> None:
    run = run_bl_ga(
        Container(2, 2, 2),
        (Box(0, 2, 1, 2), Box(1, 2, 1, 2)),
        config=GAConfig(population_size=2, generation_count=2, crossover_rate=0.8, mutation_rate=0.1),
        profile=True,
    )
    assert run.profile is not None
    metadata = run.run_metadata.to_dict()
    assert "profile" not in metadata
    assert "metrics" not in metadata

    path = tmp_path / "profile.json"
    write_decoder_profile_json(path, run.profile)
    loaded = read_decoder_profile_json(path)
    assert loaded == run.profile
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "packing3d-decoder-profile"
    assert data["run_id"] == metadata["run_id"]
    assert data["evaluations"] == {"requested": 4, "computed": 4, "cache_hits": 0}


def test_profile_reader_rejects_unknown_keys_and_bad_metrics(tmp_path) -> None:
    run = run_bl_ga(
        Container(2, 2, 2),
        (Box(0, 2, 1, 2),),
        config=GAConfig(population_size=1, generation_count=1),
        profile=True,
    )
    assert run.profile is not None
    data = run.profile.to_dict()

    unknown = dict(data, extra=True)
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        read_decoder_profile_json(path)

    bad_metric = json.loads(json.dumps(data))
    bad_metric["metrics"]["aabb_checks"] = -1
    path.write_text(json.dumps(bad_metric), encoding="utf-8")
    with pytest.raises(ValueError, match="aabb_checks"):
        read_decoder_profile_json(path)
