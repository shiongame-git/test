"""Shared axis-aligned packing geometry used by decoders."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

from .model import Box, Container, Placement, Point
from .profile import DecoderProfileCollector


def overlaps_at(box: Box, point: Point, other: Placement) -> bool:
    """Return True when ``box`` at ``point`` overlaps ``other`` by positive volume."""
    assert other.position is not None
    p = other.position
    b = other.box
    return not (
        point.x + box.w <= p.x
        or p.x + b.w <= point.x
        or point.y + box.h <= p.y
        or p.y + b.h <= point.y
        or point.z + box.d <= p.z
        or p.z + b.d <= point.z
    )


def is_feasible_position(
    container: Container,
    box: Box,
    point: Point,
    placed: Iterable[Placement],
    *,
    _profile: DecoderProfileCollector | None = None,
) -> bool:
    """Check container bounds and positive-volume overlap against successful placements."""
    started = perf_counter() if _profile is not None else 0.0
    # Candidate bounds are constant for this feasibility query. Calculate them
    # once so the same values serve both the container and obstacle checks.
    right_x = point.x + box.w
    upper_y = point.y + box.h
    back_z = point.z + box.d
    if (
        right_x > container.W
        or upper_y > container.H
        or back_z > container.D
    ):
        if _profile is not None:
            _profile.common_legality_seconds += perf_counter() - started
        return False

    for other in placed:
        other_position = other.position
        if other_position is None:
            continue
        if _profile is not None:
            # The reference decoder has no separate broad-phase index: each
            # successful AABB is both emitted as a collision candidate and checked.
            _profile.broad_phase_collision_candidates += 1
            _profile.aabb_checks += 1
        other_box = other.box
        if not (
            right_x <= other_position.x
            or other_position.x + other_box.w <= point.x
            or upper_y <= other_position.y
            or other_position.y + other_box.h <= point.y
            or back_z <= other_position.z
            or other_position.z + other_box.d <= point.z
        ):
            if _profile is not None:
                _profile.common_legality_seconds += perf_counter() - started
            return False

    if _profile is not None:
        _profile.common_legality_seconds += perf_counter() - started
    return True
