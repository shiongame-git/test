"""Independent Cartesian candidate and support oracle for decoder tests."""

from __future__ import annotations

from itertools import product
from typing import Iterable

from packing import BLPriority
from packing.model import Box, Container, Placement, Point
from oracle_support import support_area_oracle


def cartesian_candidates(
    container: Container,
    box: Box,
    successful: Iterable[Placement],
    priority: BLPriority,
) -> tuple[Point, ...]:
    """Build policy endpoints directly, then order by the specified priority."""
    limits = (container.W - box.w, container.H - box.h, container.D - box.d)
    axes = [{0}, {0}, {0}]
    for placement in successful:
        position = placement.position
        if position is None:
            continue
        ends = (
            position.x + placement.box.w,
            position.y + placement.box.h,
            position.z + placement.box.d,
        )
        for axis, end, limit in zip(axes, ends, limits):
            if end <= limit:
                axis.add(end)

    points = [Point(x, y, z) for x, y, z in product(*axes)]
    if priority is BLPriority.BOTTOM_FIRST:
        key = lambda point: (point.y, point.z, point.x)
    else:
        key = lambda point: (point.z, point.y, point.x)
    return tuple(sorted(points, key=key))


def _inside(container: Container, box: Box, point: Point) -> bool:
    return (
        point.x + box.w <= container.W
        and point.y + box.h <= container.H
        and point.z + box.d <= container.D
    )


def _overlap(box: Box, point: Point, other: Placement) -> bool:
    other_point = other.position
    if other_point is None:
        return False
    other_box = other.box
    return (
        point.x < other_point.x + other_box.w
        and other_point.x < point.x + box.w
        and point.y < other_point.y + other_box.h
        and other_point.y < point.y + box.h
        and point.z < other_point.z + other_box.d
        and other_point.z < point.z + box.d
    )


def find_cartesian_point_oracle(
    container: Container,
    box: Box,
    successful: Iterable[Placement],
    *,
    priority: BLPriority,
    min_support_percent: int,
) -> Point | None:
    """Return the first independent candidate meeting bounds, geometry, support."""
    history = tuple(successful)
    for point in cartesian_candidates(container, box, history, priority):
        if not _inside(container, box, point):
            continue
        if any(_overlap(box, point, placement) for placement in history):
            continue
        support_area = support_area_oracle(box, point, history)
        if support_area * 100 >= box.w * box.d * min_support_percent:
            return point
    return None
