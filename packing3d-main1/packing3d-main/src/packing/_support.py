"""Reference support-union calculations for decoder placements."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

from ._decoder_config import DecoderConfig
from .model import Box, Placement, Point
from .profile import DecoderProfileCollector


_Rectangle = tuple[int, int, int, int]


def _rectangle_union_area(rectangles: Iterable[_Rectangle]) -> int:
    """Return the exact union area of XZ rectangles ``(x0, x1, z0, z1)``."""
    materialized: list[_Rectangle] = []
    for index, rectangle in enumerate(rectangles):
        if type(rectangle) is not tuple or len(rectangle) != 4:
            raise TypeError(f"rectangles[{index}] must be a four-int tuple")
        if any(type(value) is not int for value in rectangle):
            raise TypeError(f"rectangles[{index}] coordinates must be Python ints")
        x0, x1, z0, z1 = rectangle
        if x1 < x0 or z1 < z0:
            raise ValueError(f"rectangles[{index}] end coordinates must not precede starts")
        if x0 < x1 and z0 < z1:
            materialized.append(rectangle)

    if not materialized:
        return 0

    x_edges = sorted({edge for x0, x1, _, _ in materialized for edge in (x0, x1)})
    area = 0
    for left, right in zip(x_edges, x_edges[1:]):
        if left == right:
            continue
        intervals = sorted(
            (z0, z1)
            for x0, x1, z0, z1 in materialized
            if x0 < right and x1 > left
        )
        covered_z = 0
        interval_start: int | None = None
        interval_end: int | None = None
        for start, end in intervals:
            if interval_start is None:
                interval_start, interval_end = start, end
            elif start > interval_end:
                covered_z += interval_end - interval_start
                interval_start, interval_end = start, end
            elif end > interval_end:
                interval_end = end
        if interval_start is not None:
            covered_z += interval_end - interval_start
        area += (right - left) * covered_z
    return area


def support_union_area(
    box: Box,
    position: Point,
    successful_placements: Iterable[Placement],
    *,
    _profile: DecoderProfileCollector | None = None,
) -> int:
    """Calculate bottom-face support from floor and successful top faces.

    Container bounds and collision legality are separate checks. Failed
    placements are ignored, and only top faces exactly at ``position.y`` count.
    """
    if type(box) is not Box:
        raise TypeError("box must be a Box")
    if type(position) is not Point:
        raise TypeError("position must be a Point")

    base_area = box.w * box.d
    if position.y == 0:
        return base_area

    rectangles: list[_Rectangle] = []
    for index, placement in enumerate(tuple(successful_placements)):
        if type(placement) is not Placement:
            raise TypeError(f"successful_placements[{index}] must be a Placement")
        support_position = placement.position
        if support_position is None:
            continue
        support_box = placement.box
        if support_position.y + support_box.h != position.y:
            continue

        x0 = max(position.x, support_position.x)
        x1 = min(position.x + box.w, support_position.x + support_box.w)
        z0 = max(position.z, support_position.z)
        z1 = min(position.z + box.d, support_position.z + support_box.d)
        if x0 < x1 and z0 < z1:
            rectangles.append((x0, x1, z0, z1))
            if _profile is not None:
                _profile.support_intersection_rectangles += 1
    return _rectangle_union_area(rectangles)


def has_sufficient_support(
    box: Box,
    position: Point,
    successful_placements: Iterable[Placement],
    settings: DecoderConfig,
    *,
    _profile: DecoderProfileCollector | None = None,
) -> bool:
    """Apply the exact integer support threshold to one candidate placement."""
    if type(settings) is not DecoderConfig:
        raise TypeError("settings must be a DecoderConfig")
    if type(box) is not Box:
        raise TypeError("box must be a Box")
    if type(position) is not Point:
        raise TypeError("position must be a Point")
    if settings.min_support_percent == 0 or position.y == 0:
        return True

    if _profile is None:
        area = support_union_area(box, position, successful_placements)
    else:
        _profile.support_searches += 1
        started = perf_counter()
        try:
            area = support_union_area(
                box, position, successful_placements, _profile=_profile
            )
        finally:
            _profile.support_union_seconds += perf_counter() - started
    return area * 100 >= box.w * box.d * settings.min_support_percent
