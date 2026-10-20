from __future__ import annotations

import random

import pytest

import packing.bl as bl_module
from packing import (
    BLPriority,
    Box,
    Container,
    DecoderConfig,
    GAConfig,
    GAExecutionConfig,
    Placement,
    Point,
    PPState,
    find_bl_point,
    pack,
    pack_pp,
    run_bl_ga,
    run_manual_replay,
)
from oracle_local_neighbors import (
    find_local_neighbor_point_oracle,
    local_neighbor_candidates,
)


LOCAL_ZERO = DecoderConfig(
    candidate_policy="LocalNeighbors",
    candidate_policy_version=1,
    min_support_percent=0,
    support_model_version="support-union-v1",
)


def _overlaps(box: Box, point: Point, placement: Placement) -> bool:
    other_point = placement.position
    if other_point is None:
        return False
    other_box = placement.box
    return (
        point.x < other_point.x + other_box.w
        and other_point.x < point.x + box.w
        and point.y < other_point.y + other_box.h
        and other_point.y < point.y + box.h
        and point.z < other_point.z + other_box.d
        and other_point.z < point.z + box.d
    )


def _ground_history(seed: int) -> tuple[Container, tuple[Placement, ...], Box]:
    rng = random.Random(seed)
    container = Container(rng.randint(2, 6), 1, rng.randint(2, 6))
    history: list[Placement] = []
    for box_id in range(rng.randint(0, 5)):
        for _ in range(30):
            width = rng.randint(1, min(3, container.W))
            depth = rng.randint(1, min(3, container.D))
            point = Point(
                rng.randint(0, container.W - width),
                0,
                rng.randint(0, container.D - depth),
            )
            box = Box(box_id + 1, width, 1, depth)
            if not any(_overlaps(box, point, prior) for prior in history):
                history.append(Placement(box, point))
                break
    box = Box(
        100,
        rng.randint(1, container.W),
        1,
        rng.randint(1, container.D),
    )
    return container, tuple(history), box


@pytest.mark.parametrize("priority", tuple(BLPriority))
@pytest.mark.parametrize("percent", (0, 55, 100))
def test_local_neighbors_matches_independent_oracle_on_small_ground_histories(
    priority: BLPriority, percent: int
):
    for seed in range(120):
        container, history, box = _ground_history(seed)
        settings = DecoderConfig(
            candidate_policy="LocalNeighbors", min_support_percent=percent
        )
        actual = find_bl_point(
            container,
            box,
            history,
            priority=priority,
            decoder_config=settings,
        )
        expected = find_local_neighbor_point_oracle(
            container,
            box,
            history,
            priority=priority,
            min_support_percent=percent,
        )
        assert actual == expected


@pytest.mark.parametrize("priority", tuple(BLPriority))
def test_local_neighbors_policy_difference_is_a_valid_result(priority: BLPriority):
    container = Container(3, 1, 3)
    history = (
        Placement(Box(10, 1, 1, 2), Point(0, 0, 0)),
        Placement(Box(11, 1, 1, 1), Point(2, 0, 0)),
    )
    current = Box(99, 2, 1, 2)

    local = find_bl_point(
        container,
        current,
        history,
        priority=priority,
        decoder_config=DecoderConfig(
            candidate_policy="LocalNeighbors", min_support_percent=55
        ),
    )
    cartesian = find_bl_point(
        container,
        current,
        history,
        priority=priority,
        decoder_config=DecoderConfig(min_support_percent=55),
    )
    expected = find_local_neighbor_point_oracle(
        container,
        current,
        history,
        priority=priority,
        min_support_percent=55,
    )

    assert cartesian == Point(1, 0, 1)
    assert local is None
    assert local == expected


def test_local_neighbors_raw_set_deduplicates_and_ignores_failed_placements(
    monkeypatch,
):
    container = Container(5, 3, 5)
    history = (
        Placement(Box(1, 1, 1, 1), Point(0, 1, 0)),
        Placement(Box(2, 1, 1, 1), Point(1, 0, 0)),
        Placement(Box(3, 2, 2, 2), None),
    )
    expected = local_neighbor_candidates(history, BLPriority.BOTTOM_FIRST)
    assert len(expected) == 6  # origin plus five unique neighbors; one is shared
    assert len(expected) <= 3 * 2 + 1

    visited: list[Point] = []
    monkeypatch.setattr(bl_module, "is_feasible_position", lambda *_args: True)

    def reject_for_support(_box, point, _history, _settings):
        visited.append(point)
        return False

    monkeypatch.setattr(bl_module, "has_sufficient_support", reject_for_support)
    assert (
        find_bl_point(
            container,
            Box(99, 1, 1, 1),
            history,
            decoder_config=LOCAL_ZERO,
        )
        is None
    )
    assert tuple(visited) == expected


def test_local_neighbors_reconsiders_raw_point_for_smaller_later_box():
    container = Container(3, 1, 1)
    result = pack(
        container,
        (Box(1, 2, 1, 1), Box(2, 2, 1, 1), Box(3, 1, 1, 1)),
        decoder_config=LOCAL_ZERO,
    )
    assert tuple(item.position for item in result.placements) == (
        Point(0, 0, 0),
        None,
        Point(2, 0, 0),
    )


def test_local_neighbors_checks_support_with_shared_legality_rule():
    container = Container(2, 2, 1)
    history = (Placement(Box(1, 2, 1, 1), Point(0, 0, 0)),)
    box = Box(2, 2, 1, 1)
    for percent in (0, 55, 100):
        settings = DecoderConfig(
            candidate_policy="LocalNeighbors", min_support_percent=percent
        )
        assert find_bl_point(
            container, box, history, decoder_config=settings
        ) == find_local_neighbor_point_oracle(
            container,
            box,
            history,
            priority=BLPriority.BOTTOM_FIRST,
            min_support_percent=percent,
        ) == Point(0, 1, 0)


def test_pp_accepts_policy_external_inherited_position_and_uses_policy_for_repack():
    container = Container(3, 1, 3)
    boxes = (
        Box(10, 1, 1, 2),
        Box(11, 1, 1, 1),
        Box(99, 2, 1, 2),
    )
    state = PPState(
        {10: Point(0, 0, 0), 11: Point(2, 0, 0), 99: None}
    )
    local_result = pack_pp(
        container,
        boxes,
        state,
        decoder_config=DecoderConfig(
            candidate_policy="LocalNeighbors", min_support_percent=55
        ),
    )
    cartesian_result = pack_pp(
        container,
        boxes,
        state,
        decoder_config=DecoderConfig(min_support_percent=55),
    )
    assert tuple(item.position for item in local_result.placements) == (
        Point(0, 0, 0),
        Point(2, 0, 0),
        None,
    )
    assert tuple(item.position for item in cartesian_result.placements) == (
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(1, 0, 1),
    )

    # Policy membership does not constrain inherited positions.
    external_state = PPState({7: Point(2, 0, 0)})
    assert pack_pp(
        Container(3, 1, 1),
        (Box(7, 1, 1, 1),),
        external_state,
        decoder_config=LOCAL_ZERO,
    ).placements[0].position == Point(2, 0, 0)


def test_local_neighbors_ga_metadata_replay_restores_policy():
    settings = DecoderConfig(
        candidate_policy="LocalNeighbors", min_support_percent=55
    )
    run = run_bl_ga(
        Container(3, 2, 2),
        (Box(1, 1, 1, 1), Box(2, 1, 1, 2), Box(3, 1, 2, 1)),
        config=GAConfig(2, 2, 0.8, 0.1),
        seed=29,
        execution=GAExecutionConfig(workers=1, cache=False),
        decoder_config=settings,
    )
    replay = run_manual_replay(run.run_metadata)

    assert replay.decoder_config == settings
    assert replay.run_metadata["decoder"]["candidate_policy"] == "LocalNeighbors"
    assert replay.run_metadata["decoder"]["min_support_percent"] == 55
    assert replay.run_metadata["replay_of"] == run.run_metadata["run_id"]
    assert replay.generations == run.generations
