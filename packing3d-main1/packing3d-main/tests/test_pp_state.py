from __future__ import annotations

import pytest

from packing import Box, Container, DecoderInput, PPState, Point


def test_pp_state_is_id_keyed_and_independent_of_construction_order():
    first = PPState([(20, Point(2, 0, 0)), (3, None), (11, Point(0, 1, 4))])
    second = PPState({11: Point(0, 1, 4), 20: Point(2, 0, 0), 3: None})

    assert first == second
    assert tuple(first) == (3, 11, 20)
    assert first[20] == Point(2, 0, 0)
    assert 3 in first
    assert first[3] is None
    assert first.items_tuple() == (
        (3, None),
        (11, Point(0, 1, 4)),
        (20, Point(2, 0, 0)),
    )


def test_pp_state_none_is_distinct_from_missing_id():
    state = PPState({1: None})
    assert 1 in state
    assert state[1] is None
    assert 2 not in state
    with pytest.raises(KeyError):
        _ = state[2]


def test_pp_state_allows_conflicting_and_problem_unchecked_positions():
    # Phase 2 deliberately does not reject inherited positions merely because
    # they collide with each other or would be out of range for some problem.
    shared = Point(99, 99, 99)
    state = PPState({1: shared, 2: shared, 999: None})
    assert state[1] == shared
    assert state[2] == shared
    assert state[999] is None


def test_pp_state_freezes_source_and_is_immutable():
    source = {1: Point(1, 2, 3)}
    state = PPState(source)
    source[1] = None
    source[2] = Point(0, 0, 0)

    assert state.items_tuple() == ((1, Point(1, 2, 3)),)
    with pytest.raises(AttributeError):
        state._items = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        state[1] = None  # type: ignore[index]
    with pytest.raises(TypeError):
        state._lookup[1] = None  # type: ignore[index]


def test_pp_state_validates_only_its_local_value_contract():
    with pytest.raises(TypeError, match="Python int"):
        PPState({True: None})
    with pytest.raises(ValueError, match=">= 0"):
        PPState([(-1, None)])
    with pytest.raises(ValueError, match="duplicate"):
        PPState([(1, None), (1, Point(0, 0, 0))])
    with pytest.raises(TypeError, match="Point or None"):
        PPState([(1, (0, 0, 0))])  # type: ignore[list-item]


def test_decoder_input_keeps_order_and_pp_state_as_separate_fields():
    boxes = (Box(20, 1, 1, 1), Box(3, 1, 1, 1))
    state = PPState({3: Point(7, 0, 0), 20: None})
    request = DecoderInput(Container(10, 10, 10), boxes, state)

    assert tuple(box.id for box in request.ordered_boxes) == (20, 3)
    assert request.decoder_state is state
    # Looking up inherited positions uses IDs, not the order's array indices.
    assert tuple(request.decoder_state[box.id] for box in request.ordered_boxes) == (
        None,
        Point(7, 0, 0),
    )


def test_decoder_input_is_generic_and_does_not_require_pp_state():
    boxes = (Box(1, 1, 1, 1),)
    bl_input = DecoderInput(Container(2, 2, 2), boxes, None)
    marker_state = object()
    other_decoder_input = DecoderInput(Container(2, 2, 2), boxes, marker_state)

    assert bl_input.decoder_state is None
    assert other_decoder_input.decoder_state is marker_state


def test_decoder_input_validates_order_without_interpreting_decoder_state():
    c = Container(2, 2, 2)
    with pytest.raises(TypeError, match="tuple"):
        DecoderInput(c, [Box(1, 1, 1, 1)], None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate"):
        DecoderInput(c, (Box(1, 1, 1, 1), Box(1, 1, 1, 1)), None)
