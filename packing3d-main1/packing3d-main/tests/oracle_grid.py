"""Independent small-integer-grid BL oracle used only by tests."""

from __future__ import annotations

from collections.abc import Iterable

from packing import BLPriority
from packing.model import Box, Container, Placement, Point


def _cells(box: Box, point: Point) -> set[tuple[int, int, int]]:
    return {
        (x, y, z)
        for z in range(point.z, point.z + box.d)
        for y in range(point.y, point.y + box.h)
        for x in range(point.x, point.x + box.w)
    }


def _all_points(container: Container, box: Box, priority: BLPriority) -> Iterable[Point]:
    if box.w > container.W or box.h > container.H or box.d > container.D:
        return

    if priority is BLPriority.BOTTOM_FIRST:
        for y in range(container.H - box.h + 1):
            for z in range(container.D - box.d + 1):
                for x in range(container.W - box.w + 1):
                    yield Point(x, y, z)
        return

    for z in range(container.D - box.d + 1):
        for y in range(container.H - box.h + 1):
            for x in range(container.W - box.w + 1):
                yield Point(x, y, z)


def find_bl_point_grid(
    container: Container,
    box: Box,
    placed: list[Placement] | tuple[Placement, ...],
    *,
    priority: BLPriority = BLPriority.BOTTOM_FIRST,
) -> Point | None:
    occupied: set[tuple[int, int, int]] = set()
    for placement in placed:
        if placement.position is not None:
            occupied.update(_cells(placement.box, placement.position))

    for point in _all_points(container, box, priority):
        if _cells(box, point).isdisjoint(occupied):
            return point
    return None


def pack_grid(
    container: Container,
    boxes: tuple[Box, ...],
    *,
    priority: BLPriority = BLPriority.BOTTOM_FIRST,
) -> tuple[Point | None, ...]:
    successful: list[Placement] = []
    positions: list[Point | None] = []
    for box in boxes:
        point = find_bl_point_grid(container, box, successful, priority=priority)
        positions.append(point)
        if point is not None:
            successful.append(Placement(box, point))
    return tuple(positions)
