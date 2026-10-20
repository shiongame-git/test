"""Independent coordinate-cell oracle for support union areas."""

from __future__ import annotations

from collections.abc import Iterable

from packing.model import Box, Placement, Point


Rectangle = tuple[int, int, int, int]


def rectangle_union_area_oracle(rectangles: Iterable[Rectangle]) -> int:
    """Sum coordinate-compressed cells covered by at least one rectangle."""
    valid = tuple(rectangle for rectangle in rectangles if rectangle[0] < rectangle[1] and rectangle[2] < rectangle[3])
    if not valid:
        return 0
    x_edges = sorted({edge for x0, x1, _, _ in valid for edge in (x0, x1)})
    z_edges = sorted({edge for _, _, z0, z1 in valid for edge in (z0, z1)})
    area = 0
    for x_index in range(len(x_edges) - 1):
        left, right = x_edges[x_index], x_edges[x_index + 1]
        for z_index in range(len(z_edges) - 1):
            near, far = z_edges[z_index], z_edges[z_index + 1]
            if any(x0 <= left and right <= x1 and z0 <= near and far <= z1 for x0, x1, z0, z1 in valid):
                area += (right - left) * (far - near)
    return area


def support_area_oracle(
    box: Box,
    position: Point,
    placements: Iterable[Placement],
) -> int:
    """Calculate clipped support with a 2D coordinate-cell partition."""
    if position.y == 0:
        return box.w * box.d

    base_x0, base_x1 = position.x, position.x + box.w
    base_z0, base_z1 = position.z, position.z + box.d
    rectangles: list[Rectangle] = []
    for placement in placements:
        support_position = placement.position
        if support_position is None or support_position.y + placement.box.h != position.y:
            continue
        clipped = (
            max(base_x0, support_position.x),
            min(base_x1, support_position.x + placement.box.w),
            max(base_z0, support_position.z),
            min(base_z1, support_position.z + placement.box.d),
        )
        if clipped[0] < clipped[1] and clipped[2] < clipped[3]:
            rectangles.append(clipped)
    return rectangle_union_area_oracle(rectangles)


def support_requirement_oracle(area: int, box: Box, percent: int) -> bool:
    return percent == 0 or area * 100 >= box.w * box.d * percent
