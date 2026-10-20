from __future__ import annotations

import pytest

from packing import (
    BLPriority,
    Box,
    Container,
    DecoderConfig,
    PackingResult,
    PPState,
    Placement,
    Point,
    pack,
    pack_pp,
    validate_result,
)
from packing.validate import validate_support
from oracle_support import support_area_oracle


def positions(result):
    return tuple(p.position for p in result.placements)


def test_all_none_pp_matches_bl_for_each_priority():
    c = Container(2, 2, 2)
    boxes = tuple(Box(i, 1, 1, 1) for i in range(1, 5))
    state = PPState({box.id: None for box in boxes})
    legacy = DecoderConfig(
        candidate_policy="Cartesian",
        candidate_policy_version=1,
        min_support_percent=0,
        support_model_version="support-union-v1",
    )

    for priority in BLPriority:
        pp_result = pack_pp(c, boxes, state, priority=priority, decoder_config=legacy)
        bl_result = pack(c, boxes, priority=priority, decoder_config=legacy)
        assert pp_result == bl_result
        assert validate_result(pp_result, expected_container=c, ordered_boxes=boxes) == ()
        for percent in (0, 55):
            settings = DecoderConfig(min_support_percent=percent)
            assert pack_pp(
                c, boxes, state, priority=priority, decoder_config=settings
            ) == pack(c, boxes, priority=priority, decoder_config=settings)


def test_legacy_pp_inherited_half_support_needs_explicit_55_percent_check():
    c = Container(3, 2, 1)
    boxes = (Box(1, 1, 1, 1), Box(2, 2, 1, 1))
    state = PPState({1: Point(0, 0, 0), 2: Point(0, 1, 0)})

    legacy = DecoderConfig(
        candidate_policy="Cartesian",
        candidate_policy_version=1,
        min_support_percent=0,
        support_model_version="support-union-v1",
    )
    result = pack_pp(c, boxes, state, decoder_config=legacy)
    assert result == pack_pp(c, boxes, state, decoder_config=DecoderConfig(min_support_percent=0))

    assert positions(result) == (Point(0, 0, 0), Point(0, 1, 0))
    assert validate_result(result, expected_container=c, ordered_boxes=boxes) == ()
    assert validate_support(result, settings=DecoderConfig(min_support_percent=0)) == ()
    assert validate_support(result, settings=DecoderConfig(min_support_percent=55)) == (
        "placement[1] does not meet the configured support requirement",
    )

    # With explicit 55%, the inherited half-supported location is deferred
    # and the box gets one supported Cartesian re-placement.
    supported = pack_pp(
        c,
        boxes,
        state,
        decoder_config=DecoderConfig(min_support_percent=55),
    )
    assert pack_pp(c, boxes, state) == supported
    assert positions(supported) == (Point(0, 0, 0), Point(1, 0, 0))
    assert validate_support(supported, settings=DecoderConfig(min_support_percent=55)) == ()


def test_pp_accepts_legal_inherited_position_outside_cartesian_candidates():
    container = Container(4, 1, 1)
    boxes = (Box(7, 1, 1, 1),)
    state = PPState({7: Point(2, 0, 0)})

    result = pack_pp(
        container,
        boxes,
        state,
        decoder_config=DecoderConfig(min_support_percent=55),
    )
    assert positions(result) == (Point(2, 0, 0),)


def test_pp_deferred_support_failures_get_one_cartesian_pass_without_retry():
    container = Container(5, 2, 1)
    boxes = (
        Box(10, 1, 2, 1),
        Box(11, 2, 1, 1),
        Box(99, 3, 1, 1),
        Box(100, 1, 1, 1),
    )
    state = PPState({
        10: Point(0, 0, 0),
        11: Point(3, 0, 0),
        99: None,
        100: None,
    })
    settings = DecoderConfig(min_support_percent=55)

    result = pack_pp(container, boxes, state, decoder_config=settings)

    assert positions(result) == (
        Point(0, 0, 0),
        Point(3, 0, 0),
        None,
        Point(1, 0, 0),
    )
    assert validate_result(result, expected_container=container, ordered_boxes=boxes) == ()
    assert validate_support(result, settings=settings) == ()

    later_supported_position = Point(1, 1, 0)
    assert support_area_oracle(
        boxes[2], later_supported_position, result.placements
    ) == 2
    after_later_box = PackingResult(
        container,
        result.placements[:2]
        + (Placement(boxes[2], later_supported_position), result.placements[3]),
    )
    assert validate_result(
        after_later_box, expected_container=container, ordered_boxes=boxes
    ) == ()
    assert validate_support(after_later_box, settings=settings) == ()


def test_decoder_config_type_errors_are_rejected_before_work():
    container = Container(2, 1, 1)
    with pytest.raises(TypeError, match="decoder_config"):
        pack(container, (), decoder_config="Cartesian")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="decoder_config"):
        pack_pp(container, (), PPState({}), decoder_config=object())  # type: ignore[arg-type]


def test_repacking_uses_selected_bl_priority():
    c = Container(2, 2, 2)
    boxes = tuple(Box(i, 1, 1, 1) for i in range(1, 4))
    state = PPState({box.id: None for box in boxes})

    bottom = pack_pp(c, boxes, state, priority=BLPriority.BOTTOM_FIRST)
    back = pack_pp(c, boxes, state, priority=BLPriority.BACK_FIRST)

    assert positions(bottom) == (Point(0, 0, 0), Point(1, 0, 0), Point(0, 0, 1))
    assert positions(back) == (Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0))


def test_default_repacking_priority_is_bottom_first():
    c = Container(2, 2, 2)
    boxes = tuple(Box(i, 1, 1, 1) for i in range(1, 4))
    state = PPState({box.id: None for box in boxes})
    assert pack_pp(c, boxes, state) == pack_pp(
        c, boxes, state, priority=BLPriority.BOTTOM_FIRST
    )


def test_conflicting_inherited_positions_use_ordered_box_priority_then_repack():
    c = Container(3, 1, 1)
    b1 = Box(1, 1, 1, 1)
    b2 = Box(2, 1, 1, 1)
    b3 = Box(3, 1, 1, 1)
    state = PPState({1: Point(0, 0, 0), 2: Point(0, 0, 0), 3: Point(2, 0, 0)})

    first_order = pack_pp(c, (b1, b2, b3), state)
    assert positions(first_order) == (Point(0, 0, 0), Point(1, 0, 0), Point(2, 0, 0))

    second_order = pack_pp(c, (b2, b1, b3), state)
    assert tuple((p.box.id, p.position) for p in second_order.placements) == (
        (2, Point(0, 0, 0)),
        (1, Point(1, 0, 0)),
        (3, Point(2, 0, 0)),
    )


def test_inherited_successes_remain_obstacles_during_repacking():
    c = Container(2, 1, 1)
    boxes = (Box(1, 1, 1, 1), Box(2, 1, 1, 1), Box(3, 1, 1, 1))
    state = PPState({1: Point(1, 0, 0), 2: None, 3: None})

    result = pack_pp(c, boxes, state)

    assert positions(result) == (Point(1, 0, 0), Point(0, 0, 0), None)
    assert result.placed_count == 2
    assert result.failed_count == 1
    assert validate_result(result, expected_container=c, ordered_boxes=boxes) == ()


def test_result_records_stay_in_input_order_even_when_boxes_are_deferred():
    c = Container(3, 1, 1)
    boxes = (Box(20, 1, 1, 1), Box(3, 1, 1, 1), Box(11, 1, 1, 1))
    state = PPState({20: None, 3: Point(0, 0, 0), 11: Point(2, 0, 0)})

    result = pack_pp(c, boxes, state)

    assert tuple(p.box.id for p in result.placements) == (20, 3, 11)
    assert positions(result) == (Point(1, 0, 0), Point(0, 0, 0), Point(2, 0, 0))


def test_pp_requires_exact_problem_id_set_at_decoder_boundary():
    c = Container(2, 1, 1)
    boxes = (Box(1, 1, 1, 1), Box(2, 1, 1, 1))

    with pytest.raises(ValueError, match=r"missing=\[2\].*unknown=\[99\]"):
        pack_pp(c, boxes, PPState({1: None, 99: None}))


def test_pp_rejects_out_of_container_inherited_position():
    c = Container(2, 2, 2)
    boxes = (Box(1, 2, 1, 1),)
    state = PPState({1: Point(1, 0, 0)})

    with pytest.raises(ValueError, match="outside the container"):
        pack_pp(c, boxes, state)


def test_pp_state_itself_remains_problem_agnostic_after_phase4():
    # PPState is still a reusable value type; problem-specific validation happens
    # when the PP decoder is called.
    state = PPState({1: Point(99, 99, 99), 999: None})
    assert state[1] == Point(99, 99, 99)
    assert state[999] is None


def test_pp_input_contract_types():
    c = Container(2, 2, 2)
    boxes = (Box(1, 1, 1, 1),)
    state = PPState({1: None})

    with pytest.raises(TypeError, match="container"):
        pack_pp(object(), boxes, state)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PPState"):
        pack_pp(c, boxes, {1: None})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BLPriority"):
        pack_pp(c, boxes, state, priority="BottomFirst")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"ordered_boxes\[0\]"):
        pack_pp(c, (object(),), PPState({}))  # type: ignore[arg-type]


def test_same_complete_pp_input_is_deterministic():
    c = Container(3, 2, 2)
    boxes = (
        Box(10, 1, 2, 1),
        Box(30, 1, 1, 2),
        Box(20, 2, 1, 1),
    )
    state = PPState({10: Point(0, 0, 0), 30: Point(0, 0, 0), 20: None})

    first = pack_pp(c, boxes, state, priority=BLPriority.BACK_FIRST)
    second = pack_pp(c, boxes, state, priority=BLPriority.BACK_FIRST)
    assert first == second


def test_seeded_small_random_pp_results_are_legal_and_deterministic():
    import random

    rng = random.Random(20261001)
    for _ in range(200):
        c = Container(rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 5))
        boxes = tuple(
            Box(i + 1, rng.randint(1, 6), rng.randint(1, 6), rng.randint(1, 6))
            for i in range(rng.randint(0, 8))
        )
        entries = {}
        for box in boxes:
            if box.w <= c.W and box.h <= c.H and box.d <= c.D and rng.random() < 0.7:
                entries[box.id] = Point(
                    rng.randint(0, c.W - box.w),
                    rng.randint(0, c.H - box.h),
                    rng.randint(0, c.D - box.d),
                )
            else:
                entries[box.id] = None
        state = PPState(entries)
        priority = rng.choice(tuple(BLPriority))

        first = pack_pp(c, boxes, state, priority=priority)
        second = pack_pp(c, boxes, state, priority=priority)
        assert first == second
        assert validate_result(first, expected_container=c, ordered_boxes=boxes) == ()
