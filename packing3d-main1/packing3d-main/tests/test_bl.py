from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path

import pytest

from packing import (
    BLPriority,
    Box,
    Container,
    DecoderConfig,
    PackingResult,
    Placement,
    Point,
    find_bl_point,
    pack,
    validate_result,
    validate_support,
)
from packing._support import has_sufficient_support, support_union_area
from oracle_grid import find_bl_point_grid, pack_grid
from oracle_cartesian import find_cartesian_point_oracle
from oracle_support import support_area_oracle


SUPPORT_AUDIT_FIXTURE = Path(__file__).parent / "fixtures" / "legacy_support_difference.json"
LEGACY_CARTESIAN_ZERO = DecoderConfig(
    candidate_policy="Cartesian",
    candidate_policy_version=1,
    min_support_percent=0,
    support_model_version="support-union-v1",
)


def positions(result):
    return tuple(p.position for p in result.placements)


def _cartesian_candidates(container, box, placed, priority):
    """Enumerate the fixed Cartesian candidate set independently for the audit case."""
    max_x = container.W - box.w
    max_y = container.H - box.h
    max_z = container.D - box.d
    axes = [{0}, {0}, {0}]
    maxima = (max_x, max_y, max_z)
    for placement in placed:
        if placement.position is None:
            continue
        point = placement.position
        prior_box = placement.box
        ends = (
            point.x + prior_box.w,
            point.y + prior_box.h,
            point.z + prior_box.d,
        )
        for axis, end, maximum in zip(axes, ends, maxima):
            if end <= maximum:
                axis.add(end)
    candidates = [Point(x, y, z) for x, y, z in product(*map(sorted, axes))]
    key = (lambda point: (point.y, point.z, point.x)) if priority is BLPriority.BOTTOM_FIRST else (
        lambda point: (point.z, point.y, point.x)
    )
    return tuple(sorted(candidates, key=key))


def _overlaps(box, point, placement):
    prior_point = placement.position
    if prior_point is None:
        return False
    prior_box = placement.box
    return (
        point.x < prior_point.x + prior_box.w
        and prior_point.x < point.x + box.w
        and point.y < prior_point.y + prior_box.h
        and prior_point.y < point.y + box.h
        and point.z < prior_point.z + prior_box.d
        and prior_point.z < point.z + box.d
    )


def _point_value(point):
    return None if point is None else [point.x, point.y, point.z]


def test_empty_input():
    for priority in BLPriority:
        result = pack(Container(2, 2, 2), [], priority=priority)
        assert result.placements == ()
        assert result.placed_count == 0
        assert result.failed_count == 0
        assert result.placed_volume == 0
        assert result.density == 0
        assert result.failed_boxes == ()


def test_exact_fit():
    b = Box(1, 2, 3, 4)
    for priority in BLPriority:
        result = pack(Container(2, 3, 4), [b], priority=priority)
        assert positions(result) == (Point(0, 0, 0),)
        assert result.density == 1.0
        assert result.placements[0].box is b


def test_unit_cube_orders_distinguish_priorities_and_default_is_bottom_first():
    boxes = tuple(Box(i, 1, 1, 1) for i in range(8))
    container = Container(2, 2, 2)

    bottom_expected = (
        Point(0, 0, 0), Point(1, 0, 0), Point(0, 0, 1), Point(1, 0, 1),
        Point(0, 1, 0), Point(1, 1, 0), Point(0, 1, 1), Point(1, 1, 1),
    )
    back_expected = (
        Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0), Point(1, 1, 0),
        Point(0, 0, 1), Point(1, 0, 1), Point(0, 1, 1), Point(1, 1, 1),
    )

    assert positions(pack(container, boxes)) == bottom_expected
    assert positions(pack(container, boxes, priority=BLPriority.BOTTOM_FIRST)) == bottom_expected
    assert positions(pack(container, boxes, priority=BLPriority.BACK_FIRST)) == back_expected


def test_failure_after_full_and_continue_after_failure():
    boxes = tuple(Box(i, 1, 1, 1) for i in range(9))
    for priority in BLPriority:
        result = pack(Container(2, 2, 2), boxes, priority=priority)
        assert result.placements[-1].position is None

        result = pack(
            Container(2, 2, 2),
            [Box(1, 3, 1, 1), Box(2, 1, 1, 1)],
            priority=priority,
        )
        assert positions(result) == (None, Point(0, 0, 0))


def test_insufficient_space_then_continue():
    boxes = [Box(1, 1, 2, 1), Box(2, 2, 1, 1), Box(3, 1, 1, 1)]
    for priority in BLPriority:
        assert positions(pack(Container(2, 2, 1), boxes, priority=priority)) == (
            Point(0, 0, 0), None, Point(1, 0, 0)
        )


def test_input_order_and_duplicate_ids():
    boxes = [Box(42, 1, 1, 1), Box(7, 1, 1, 1)]
    for priority in BLPriority:
        result = pack(Container(2, 1, 1), boxes, priority=priority)
        assert [p.box.id for p in result.placements] == [42, 7]
    with pytest.raises(ValueError, match="duplicate"):
        pack(Container(2, 2, 2), [Box(1, 1, 1, 1), Box(1, 1, 1, 1)])


def test_list_tuple_generator_same_result():
    container = Container(2, 2, 1)
    for priority in BLPriority:
        source = (Box(i, 1, 1, 1) for i in range(4))
        expected = pack(container, source, priority=priority)
        boxes = tuple(p.box for p in expected.placements)
        assert pack(container, list(boxes), priority=priority) == expected
        assert pack(container, boxes, priority=priority) == expected
        assert pack(container, (b for b in boxes), priority=priority) == expected


def test_model_contract_and_immutability():
    with pytest.raises(TypeError):
        Container(True, 1, 1)
    with pytest.raises(TypeError):
        Box(0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        Box(-1, 1, 1, 1)
    with pytest.raises(ValueError):
        Point(0, -1, 0)
    b = Box(1, 1, 1, 1)
    with pytest.raises(Exception):
        b.w = 2  # type: ignore[misc]


def test_priority_requires_public_enum():
    container = Container(2, 2, 2)
    box = Box(1, 1, 1, 1)
    with pytest.raises(TypeError, match="BLPriority"):
        pack(container, [box], priority="BottomFirst")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BLPriority"):
        find_bl_point(container, box, [], priority="BackFirst")  # type: ignore[arg-type]


@pytest.mark.parametrize("priority", tuple(BLPriority))
def test_candidate_search_matches_grid_oracle_random_1000_cases(priority: BLPriority):
    # Retained v1.1 Cartesian/0% full-grid regression. This oracle is not a
    # LocalNeighbors oracle and must keep its original seed and case count.
    rng = random.Random(20260918)
    for case_index in range(1000):
        container = Container(rng.randint(1, 6), rng.randint(1, 6), rng.randint(1, 6))
        boxes = tuple(
            Box(i, rng.randint(1, 7), rng.randint(1, 7), rng.randint(1, 7))
            for i in range(rng.randint(0, 10))
        )
        actual = positions(pack(
            container,
            boxes,
            priority=priority,
            decoder_config=LEGACY_CARTESIAN_ZERO,
        ))
        expected = pack_grid(container, boxes, priority=priority)
        assert actual == expected, (
            f"case={case_index} seed=20260918 priority={priority.value} "
            f"container={container} boxes={boxes} expected={expected} actual={actual}"
        )


@pytest.mark.parametrize("priority", tuple(BLPriority))
def test_candidate_search_on_arbitrary_legal_obstacles_matches_oracle(priority: BLPriority):
    c = Container(5, 5, 5)
    obstacles = [
        Placement(Box(10, 1, 3, 2), Point(0, 0, 0)),
        Placement(Box(11, 2, 1, 2), Point(1, 0, 0)),
        Placement(Box(12, 1, 1, 3), Point(0, 3, 0)),
    ]
    box = Box(99, 2, 2, 2)
    assert validate_result(PackingResult(c, tuple(obstacles))) == ()
    assert find_bl_point(
        c, box, obstacles, priority=priority, decoder_config=LEGACY_CARTESIAN_ZERO
    ) == find_bl_point_grid(
        c, box, obstacles, priority=priority
    )
    for percent in (0, 55):
        settings = DecoderConfig(min_support_percent=percent)
        assert find_bl_point(
            c, box, obstacles, priority=priority, decoder_config=settings
        ) == find_cartesian_point_oracle(
            c,
            box,
            obstacles,
            priority=priority,
            min_support_percent=percent,
        )


@pytest.mark.parametrize("priority", tuple(BLPriority))
@pytest.mark.parametrize("percent", (0, 55, 100))
def test_cartesian_support_candidates_match_independent_minimum_oracle(priority, percent):
    case = json.loads(SUPPORT_AUDIT_FIXTURE.read_text(encoding="utf-8"))
    container = Container(*case["container"])
    boxes = tuple(Box(*values) for values in case["boxes"])
    history = tuple(
        Placement(box, Point(*position))
        for box, position in zip(boxes[:-1], case["legacy_positions"][:-1])
    )
    settings = DecoderConfig(min_support_percent=percent)

    actual = find_bl_point(
        container,
        boxes[-1],
        history,
        priority=priority,
        decoder_config=settings,
    )
    expected = find_cartesian_point_oracle(
        container,
        boxes[-1],
        history,
        priority=priority,
        min_support_percent=percent,
    )
    assert actual == expected
    assert actual == (Point(1, 1, 0) if percent == 0 else None)


@pytest.mark.parametrize("priority", tuple(BLPriority))
def test_policy_external_lattice_point_is_legal_but_not_cartesian_candidate(priority):
    container = Container(5, 2, 1)
    history = (
        Placement(Box(10, 1, 2, 1), Point(0, 0, 0)),
        Placement(Box(11, 2, 1, 1), Point(3, 0, 0)),
    )
    box = Box(99, 3, 1, 1)
    point = Point(2, 1, 0)
    settings = DecoderConfig(min_support_percent=55)

    assert point not in _cartesian_candidates(container, box, history, priority)
    assert find_bl_point(
        container, box, history, priority=priority, decoder_config=settings
    ) is None
    assert find_cartesian_point_oracle(
        container,
        box,
        history,
        priority=priority,
        min_support_percent=55,
    ) is None
    assert support_area_oracle(box, point, history) == 2

    result = PackingResult(container, history + (Placement(box, point),))
    assert validate_result(result) == ()
    assert validate_support(result, settings=settings) == ()


@pytest.mark.parametrize("priority", tuple(BLPriority))
def test_cartesian_hundred_percent_accepts_fully_supported_candidate(priority):
    container = Container(5, 2, 4)
    successful = (Placement(Box(1, 5, 1, 4), Point(0, 0, 0)),)
    box = Box(2, 1, 1, 1)
    settings = DecoderConfig(min_support_percent=100)

    expected = find_cartesian_point_oracle(
        container,
        box,
        successful,
        priority=priority,
        min_support_percent=100,
    )
    assert expected == Point(0, 1, 0)
    assert find_bl_point(
        container,
        box,
        successful,
        priority=priority,
        decoder_config=settings,
    ) == expected


@pytest.mark.parametrize("priority", tuple(BLPriority))
def test_candidate_combines_end_faces_from_different_obstacles(priority: BLPriority):
    # The candidate x=1 comes from obstacle A's +X face, while y=1 comes
    # from obstacle B's +Y face. No single obstacle supplies both coordinates.
    c = Container(5, 5, 1)
    obstacles = [
        Placement(Box(10, 1, 1, 1), Point(0, 1, 0)),
        Placement(Box(11, 1, 1, 1), Point(2, 0, 0)),
    ]
    candidate = Box(99, 3, 1, 1)
    assert find_bl_point(
        c, candidate, obstacles, priority=priority, decoder_config=LEGACY_CARTESIAN_ZERO
    ) == Point(1, 1, 0)
    assert find_bl_point(
        c, candidate, obstacles, priority=priority, decoder_config=LEGACY_CARTESIAN_ZERO
    ) == find_bl_point_grid(
        c, candidate, obstacles, priority=priority
    )


def test_cartesian_support_audit_keeps_legacy_zero_and_shows_55_percent_difference():
    case = json.loads(SUPPORT_AUDIT_FIXTURE.read_text(encoding="utf-8"))
    container = Container(*case["container"])
    boxes = tuple(Box(*values) for values in case["boxes"])
    priority = next(item for item in BLPriority if item.value == case["priority"])
    legacy_settings = DecoderConfig(**case["legacy_settings"])
    support_55_settings = DecoderConfig(**case["support_55_settings"])

    # The fixed legacy case uses explicit Cartesian/0% settings. An omitted
    # setting now uses the standard 55% support requirement.
    legacy_result = pack(container, boxes, priority=priority, decoder_config=legacy_settings)
    assert [_point_value(point) for point in positions(legacy_result)] == case["legacy_positions"]
    default_result = pack(container, boxes, priority=priority)
    assert [_point_value(point) for point in positions(default_result)] == case[
        "reference_positions_at_55"
    ]
    assert find_bl_point(
        container,
        boxes[-1],
        legacy_result.placements[:-1],
        priority=priority,
    ) is None
    assert [_point_value(point) for point in positions(pack(
        container,
        boxes,
        priority=priority,
        decoder_config=support_55_settings,
    ))] == case["reference_positions_at_55"]

    box = boxes[-1]
    accepted_history = legacy_result.placements[:-1]
    candidates = _cartesian_candidates(container, box, accepted_history, priority)
    geometric_candidates = tuple(
        point
        for point in candidates
        if not any(_overlaps(box, point, placement) for placement in accepted_history)
    )
    assert [_point_value(point) for point in geometric_candidates] == case[
        "geometry_feasible_cartesian_candidates"
    ]

    accepted_by_percent = {}
    for label, settings in (("0", legacy_settings), ("55", support_55_settings)):
        accepted = []
        for point in geometric_candidates:
            area = support_union_area(box, point, accepted_history)
            assert area == support_area_oracle(box, point, accepted_history)
            if has_sufficient_support(box, point, accepted_history, settings):
                accepted.append(_point_value(point))
        accepted_by_percent[label] = accepted
    assert accepted_by_percent == case["accepted_cartesian_candidates_by_support_percent"]
    assert accepted_by_percent["0"] == [_point_value(legacy_result.placements[-1].position)]

    support_55_prefix = []
    support_55_history = []
    for placement in legacy_result.placements[:-1]:
        assert placement.position is not None
        assert has_sufficient_support(
            placement.box, placement.position, support_55_history, support_55_settings
        )
        support_55_prefix.append(_point_value(placement.position))
        support_55_history.append(placement)
    support_55_last = accepted_by_percent["55"][0] if accepted_by_percent["55"] else None
    assert support_55_prefix + [support_55_last] == case["reference_positions_at_55"]
