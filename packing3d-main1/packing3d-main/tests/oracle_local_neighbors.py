"""Independent LocalNeighbors v1 candidate and support oracle."""

from __future__ import annotations

from collections.abc import Iterable

from packing import BLPriority
from packing.model import Box, Container, Placement, Point
from oracle_support import support_area_oracle


def local_neighbor_candidates(
    successful: Iterable[Placement], priority: BLPriority
) -> tuple[Point, ...]:
    """Build the raw origin-plus-three-successor set independently."""
    points = {Point(0, 0, 0)}
    for placement in successful:
        if placement.position is None:
            continue
        position = placement.position
        box = placement.box
        points.add(Point(position.x + box.w, position.y, position.z))
        points.add(Point(position.x, position.y + box.h, position.z))
        points.add(Point(position.x, position.y, position.z + box.d))

    if priority is BLPriority.BOTTOM_FIRST:
        key = lambda point: (point.y, point.z, point.x)
    else:
        key = lambda point: (point.z, point.y, point.x)
    return tuple(sorted(points, key=key))


def _fits(container: Container, box: Box, point: Point) -> bool:
    return (
        point.x >= 0
        and point.y >= 0
        and point.z >= 0
        and point.x + box.w <= container.W
        and point.y + box.h <= container.H
        and point.z + box.d <= container.D
    )


def _overlaps(box: Box, point: Point, other: Placement) -> bool:
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


def find_local_neighbor_point_oracle(
    container: Container,
    box: Box,
    successful: Iterable[Placement],
    *,
    priority: BLPriority,
    min_support_percent: int,
) -> Point | None:
    """Find the first raw LocalNeighbors point meeting independent checks."""
    history = tuple(successful)
    for point in local_neighbor_candidates(history, priority):
        if not _fits(container, box, point):
            continue
        if any(_overlaps(box, point, placement) for placement in history):
            continue
        supported_area = support_area_oracle(box, point, history)
        if supported_area * 100 >= box.w * box.d * min_support_percent:
            return point
    return None
