"""Position-preserving (PP) decoder."""

from __future__ import annotations

from collections.abc import Iterable

from ._decoder_config import DecoderConfig, resolve_decoder_config
from .bl import BLPriority, find_bl_point
from ._validation import freeze_decoder_boxes
from .geometry import is_feasible_position
from .model import Box, Container, PackingResult, Placement, Point
from .pp_state import PPState
from ._support import has_sufficient_support
from .profile import DecoderProfileCollector


def _validate_state_for_problem(
    container: Container,
    boxes: tuple[Box, ...],
    state: PPState,
) -> None:
    problem_ids = {box.id for box in boxes}
    state_ids = set(state)
    if state_ids != problem_ids:
        missing = sorted(problem_ids - state_ids)
        unknown = sorted(state_ids - problem_ids)
        raise ValueError(
            "PPState IDs must exactly match the problem box IDs: "
            f"missing={missing}, unknown={unknown}"
        )

    by_id = {box.id: box for box in boxes}
    for box_id in sorted(problem_ids):
        position = state[box_id]
        if position is None:
            continue
        box = by_id[box_id]
        if (
            position.x + box.w > container.W
            or position.y + box.h > container.H
            or position.z + box.d > container.D
        ):
            raise ValueError(
                f"PPState position for box id {box_id} is outside the container: "
                f"position=({position.x},{position.y},{position.z}) "
                f"size=({box.w},{box.h},{box.d})"
            )


def pack_pp(
    container: Container,
    ordered_boxes: Iterable[Box],
    state: PPState,
    *,
    priority: BLPriority = BLPriority.BOTTOM_FIRST,
    decoder_config: DecoderConfig | None = None,
    _profile: DecoderProfileCollector | None = None,
) -> PackingResult:
    """Decode inherited positions, then BL-repack deferred boxes once.

    Inherited positions are attempted in ``ordered_boxes`` order.  An inherited
    position that conflicts with an earlier accepted position is deferred.  A
    ``None`` inherited position is also deferred.  Deferred boxes are processed
    once, in the same order, with the selected BL candidate priority while all
    accepted and newly placed boxes remain as obstacles.

    The final ``PackingResult`` records remain in the original ``ordered_boxes``
    order.  PP does not rewrite the supplied box order.
    """
    if type(container) is not Container:
        raise TypeError("container must be a Container")
    if type(state) is not PPState:
        raise TypeError("state must be a PPState")
    if type(priority) is not BLPriority:
        raise TypeError("priority must be a BLPriority")
    decoder_config = resolve_decoder_config(decoder_config)

    boxes = freeze_decoder_boxes(ordered_boxes)
    _validate_state_for_problem(container, boxes, state)

    accepted: list[Placement] = []
    deferred: list[Box] = []
    final_positions: dict[int, Point | None] = {}

    # Inherited-position pass.  Earlier boxes in evaluation order win conflicts.
    for box in boxes:
        inherited = state[box.id]
        if _profile is None:
            accepted_inherited = (
                inherited is not None
                and is_feasible_position(container, box, inherited, accepted)
                and has_sufficient_support(box, inherited, accepted, decoder_config)
            )
        elif inherited is None:
            accepted_inherited = False
        elif not is_feasible_position(
            container, box, inherited, accepted, _profile=_profile
        ):
            _profile.pp_deferred_overlap += 1
            accepted_inherited = False
        elif not has_sufficient_support(
            box, inherited, accepted, decoder_config, _profile=_profile
        ):
            _profile.pp_deferred_support += 1
            accepted_inherited = False
        else:
            _profile.pp_inherited_accepted += 1
            accepted_inherited = True

        if accepted_inherited:
            placement = Placement(box, inherited)
            accepted.append(placement)
            final_positions[box.id] = inherited
        else:
            deferred.append(box)

    # One BL re-placement pass using the caller-selected priority.  Boxes that
    # remain unplaced cannot become easier to place later because obstacles only
    # accumulate during this pass.
    for box in deferred:
        if _profile is None:
            point = find_bl_point(
                container,
                box,
                accepted,
                priority=priority,
                decoder_config=decoder_config,
            )
        else:
            point = find_bl_point(
                container,
                box,
                accepted,
                priority=priority,
                decoder_config=decoder_config,
                _profile=_profile,
            )
        final_positions[box.id] = point
        if point is not None:
            accepted.append(Placement(box, point))
            if _profile is not None:
                _profile.pp_repack_successes += 1
        elif _profile is not None:
            _profile.pp_repack_failures += 1

    placements = tuple(Placement(box, final_positions[box.id]) for box in boxes)
    return PackingResult(container, placements)
