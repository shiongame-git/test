"""Reference three-dimensional Bottom-Left packing implementation."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from time import perf_counter

from ._decoder_config import DecoderConfig, resolve_decoder_config
from ._support import has_sufficient_support
from ._validation import freeze_decoder_boxes
from .geometry import is_feasible_position
from .model import Box, Container, PackingResult, Placement, Point
from .profile import DecoderProfileCollector


class BLPriority(Enum):
    """Candidate-point comparison order used by the reference BL decoder."""

    BOTTOM_FIRST = "BottomFirst"
    BACK_FIRST = "BackFirst"


def _require_priority(priority: object) -> BLPriority:
    if type(priority) is not BLPriority:
        raise TypeError("priority must be a BLPriority")
    return priority


def _candidate_points(
    xs: list[int], ys: list[int], zs: list[int], priority: BLPriority
) -> Iterable[Point]:
    if priority is BLPriority.BOTTOM_FIRST:
        # Comparison key: (y, z, x)
        for y in ys:
            for z in zs:
                for x in xs:
                    yield Point(x, y, z)
        return

    # Comparison key: (z, y, x)
    for z in zs:
        for y in ys:
            for x in xs:
                yield Point(x, y, z)


def _local_neighbor_points(placed: tuple[Placement, ...]) -> set[Point]:
    """Return LocalNeighbors v1 raw points from successful placements only."""
    points = {Point(0, 0, 0)}
    for placement in placed:
        if placement.position is None:
            continue
        position = placement.position
        box = placement.box
        points.update(
            (
                Point(position.x + box.w, position.y, position.z),
                Point(position.x, position.y + box.h, position.z),
                Point(position.x, position.y, position.z + box.d),
            )
        )
    return points


def _priority_key(point: Point, priority: BLPriority) -> tuple[int, int, int]:
    if priority is BLPriority.BOTTOM_FIRST:
        return (point.y, point.z, point.x)
    return (point.z, point.y, point.x)


def find_bl_point(
    container: Container,
    box: Box,
    placed: Iterable[Placement],
    *,
    priority: BLPriority = BLPriority.BOTTOM_FIRST,
    decoder_config: DecoderConfig | None = None,
    _profile: DecoderProfileCollector | None = None,
) -> Point | None:
    """Find the minimum feasible candidate point for the selected BL priority.

    ``BottomFirst`` compares candidates by ``(y, z, x)`` and is the default.
    ``BackFirst`` compares candidates by ``(z, y, x)`` and preserves the v0.3.0
    reference search order.

    The function expects a legal existing placement history; :func:`pack` is the
    normal public entry point that guarantees this precondition.
    """
    if type(container) is not Container:
        raise TypeError("container must be a Container")
    if type(box) is not Box:
        raise TypeError("box must be a Box")
    priority = _require_priority(priority)
    decoder_config = resolve_decoder_config(decoder_config)

    placed_tuple = tuple(placed)
    for index, placement in enumerate(placed_tuple):
        if type(placement) is not Placement:
            raise TypeError(f"placed[{index}] must be a Placement")

    if box.w > container.W or box.h > container.H or box.d > container.D:
        if _profile is not None:
            build_started = perf_counter()
            if decoder_config.candidate_policy == "Cartesian":
                raw_xs = {0}
                raw_ys = {0}
                raw_zs = {0}
                for placement in placed_tuple:
                    if placement.position is None:
                        continue
                    position = placement.position
                    placed_box = placement.box
                    raw_xs.add(position.x + placed_box.w)
                    raw_ys.add(position.y + placed_box.h)
                    raw_zs.add(position.z + placed_box.d)
                _profile.raw_candidates += len(raw_xs) * len(raw_ys) * len(raw_zs)
            else:
                _profile.raw_candidates += len(_local_neighbor_points(placed_tuple))
            _profile.candidate_construction_seconds += perf_counter() - build_started
        return None

    placed_list = list(placed_tuple)
    if decoder_config.candidate_policy == "Cartesian":
        max_x = container.W - box.w
        max_y = container.H - box.h
        max_z = container.D - box.d

        cx = {0}
        cy = {0}
        cz = {0}
        if _profile is None:
            for placement in placed_tuple:
                if placement.position is None:
                    continue
                p = placement.position
                b = placement.box
                end_x = p.x + b.w
                end_y = p.y + b.h
                end_z = p.z + b.d
                if end_x <= max_x:
                    cx.add(end_x)
                if end_y <= max_y:
                    cy.add(end_y)
                if end_z <= max_z:
                    cz.add(end_z)
        else:
            build_started = perf_counter()
            raw_cx = {0}
            raw_cy = {0}
            raw_cz = {0}
            for placement in placed_tuple:
                if placement.position is None:
                    continue
                p = placement.position
                b = placement.box
                end_x = p.x + b.w
                end_y = p.y + b.h
                end_z = p.z + b.d
                raw_cx.add(end_x)
                raw_cy.add(end_y)
                raw_cz.add(end_z)
                if end_x <= max_x:
                    cx.add(end_x)
                if end_y <= max_y:
                    cy.add(end_y)
                if end_z <= max_z:
                    cz.add(end_z)

        candidates = _candidate_points(
            sorted(cx), sorted(cy), sorted(cz), priority
        )
        if _profile is not None:
            _profile.raw_candidates += len(raw_cx) * len(raw_cy) * len(raw_cz)
            _profile.bounds_eligible_candidates += len(cx) * len(cy) * len(cz)
            _profile.candidate_construction_seconds += perf_counter() - build_started
    else:
        build_started = perf_counter() if _profile is not None else 0.0
        raw_points = _local_neighbor_points(placed_tuple)
        candidates = iter(
            sorted(raw_points, key=lambda point: _priority_key(point, priority))
        )
        if _profile is not None:
            _profile.raw_candidates += len(raw_points)
            _profile.bounds_eligible_candidates += sum(
                point.x + box.w <= container.W
                and point.y + box.h <= container.H
                and point.z + box.d <= container.D
                for point in raw_points
            )
            _profile.candidate_construction_seconds += perf_counter() - build_started

    if _profile is None:
        for point in candidates:
            if is_feasible_position(container, box, point, placed_list) and has_sufficient_support(
                box, point, placed_tuple, decoder_config
            ):
                return point
        return None

    candidate_iter = iter(candidates)
    while True:
        next_started = perf_counter()
        try:
            point = next(candidate_iter)
        except StopIteration:
            _profile.candidate_construction_seconds += perf_counter() - next_started
            break
        _profile.candidate_construction_seconds += perf_counter() - next_started
        _profile.visited_candidates += 1
        if is_feasible_position(
            container, box, point, placed_list, _profile=_profile
        ) and has_sufficient_support(
            box, point, placed_tuple, decoder_config, _profile=_profile
        ):
            return point
    return None


def pack(
    container: Container,
    ordered_boxes: Iterable[Box],
    *,
    priority: BLPriority = BLPriority.BOTTOM_FIRST,
    decoder_config: DecoderConfig | None = None,
    _profile: DecoderProfileCollector | None = None,
) -> PackingResult:
    """Pack boxes in the supplied order with the selected reference 3D BL rule."""
    if type(container) is not Container:
        raise TypeError("container must be a Container")
    priority = _require_priority(priority)
    decoder_config = resolve_decoder_config(decoder_config)
    boxes = freeze_decoder_boxes(ordered_boxes)

    placements: list[Placement] = []
    successful: list[Placement] = []
    for box in boxes:
        if _profile is None:
            point = find_bl_point(
                container,
                box,
                successful,
                priority=priority,
                decoder_config=decoder_config,
            )
        else:
            point = find_bl_point(
                container,
                box,
                successful,
                priority=priority,
                decoder_config=decoder_config,
                _profile=_profile,
            )
        placement = Placement(box, point)
        placements.append(placement)
        if point is not None:
            successful.append(placement)

    return PackingResult(container, tuple(placements))
