"""Independent legality and input-consistency validation for packing results."""

from __future__ import annotations

from collections.abc import Iterable

from ._decoder_config import DecoderConfig
from .model import Box, Container, PackingResult, Placement, Point


def _safe_text(value: object) -> str:
    """Render validation diagnostics without leaking int string-limit errors."""
    try:
        return str(value) if type(value) is int else repr(value)
    except Exception:
        return f"<{type(value).__name__} value unavailable>"


def _validate_expected_boxes(ordered_boxes: Iterable[Box]) -> tuple[Box, ...]:
    boxes = tuple(ordered_boxes)
    seen: set[int] = set()
    for index, box in enumerate(boxes):
        if type(box) is not Box:
            raise TypeError(f"ordered_boxes[{index}] must be a Box")
        # Box construction normally guarantees these primitive contracts. Check
        # again here because validate_result explicitly owns input validation for
        # its comparison arguments.
        for name, value, minimum in (
            ("id", box.id, 0), ("w", box.w, 1), ("h", box.h, 1), ("d", box.d, 1)
        ):
            if type(value) is not int:
                raise TypeError(f"ordered_boxes[{index}].{name} must be a Python int")
            if value < minimum:
                raise ValueError(f"ordered_boxes[{index}].{name} must be >= {minimum}")
        if box.id in seen:
            raise ValueError(f"duplicate box id in ordered_boxes: {_safe_text(box.id)}")
        seen.add(box.id)
    return boxes


def _intersection_length(a0: int, a1: int, b0: int, b1: int) -> int:
    return min(a1, b1) - max(a0, b0)


def _validation_rectangle_union_area(
    rectangles: Iterable[tuple[int, int, int, int]],
) -> int:
    """Independently measure an XZ rectangle union by sweeping along Z."""
    materialized = tuple(
        rectangle
        for rectangle in rectangles
        if rectangle[0] < rectangle[1] and rectangle[2] < rectangle[3]
    )
    if not materialized:
        return 0

    z_edges = sorted({edge for _, _, z0, z1 in materialized for edge in (z0, z1)})
    area = 0
    for near, far in zip(z_edges, z_edges[1:]):
        x_intervals = sorted(
            (x0, x1)
            for x0, x1, z0, z1 in materialized
            if z0 < far and z1 > near
        )
        if not x_intervals:
            continue

        covered_x = 0
        interval_start, interval_end = x_intervals[0]
        for start, end in x_intervals[1:]:
            if start > interval_end:
                covered_x += interval_end - interval_start
                interval_start, interval_end = start, end
            elif end > interval_end:
                interval_end = end
        covered_x += interval_end - interval_start
        area += (far - near) * covered_x
    return area


def _independent_support_area(
    box: Box,
    position: Point,
    placements: Iterable[Placement],
) -> int:
    """Recompute support faces and clipping without decoder support code."""
    if position.y == 0:
        return box.w * box.d

    base_right = position.x + box.w
    base_back = position.z + box.d
    clipped_faces: list[tuple[int, int, int, int]] = []
    for placement in placements:
        support_position = placement.position
        if support_position is None:
            continue
        support_box = placement.box
        if support_position.y + support_box.h != position.y:
            continue

        left = max(position.x, support_position.x)
        right = min(base_right, support_position.x + support_box.w)
        near = max(position.z, support_position.z)
        far = min(base_back, support_position.z + support_box.d)
        if left < right and near < far:
            clipped_faces.append((left, right, near, far))

    return _validation_rectangle_union_area(clipped_faces)


def _int_field_error(value: object, minimum: int) -> str | None:
    if type(value) is not int:
        return "must be a Python int"
    if value < minimum:
        return f"must be >= {minimum}"
    return None


def validate_result(
    result: PackingResult,
    *,
    expected_container: Container | None = None,
    ordered_boxes: Iterable[Box] | None = None,
) -> tuple[str, ...]:
    """Return all safely detectable result errors without using BL collision code."""
    if type(result) is not PackingResult:
        raise TypeError("result must be a PackingResult")
    if expected_container is not None:
        if type(expected_container) is not Container:
            raise TypeError("expected_container must be a Container or None")
        for name, value in (("W", expected_container.W), ("H", expected_container.H), ("D", expected_container.D)):
            if type(value) is not int:
                raise TypeError(f"expected_container.{name} must be a Python int")
            if value < 1:
                raise ValueError(f"expected_container.{name} must be >= 1")
    expected_boxes = _validate_expected_boxes(ordered_boxes) if ordered_boxes is not None else None

    errors: list[str] = []
    container = result.container
    if type(container) is not Container:
        errors.append("result.container is not a Container")
        return tuple(errors)

    container_ok = True
    for name, value in (("W", container.W), ("H", container.H), ("D", container.D)):
        problem = _int_field_error(value, 1)
        if problem:
            errors.append(f"result.container.{name} {problem}")
            container_ok = False

    if expected_container is not None:
        if container_ok and container != expected_container:
            errors.append(
                "container mismatch: "
                f"result=({_safe_text(container.W)},{_safe_text(container.H)},{_safe_text(container.D)}) "
                f"expected=({_safe_text(expected_container.W)},{_safe_text(expected_container.H)},{_safe_text(expected_container.D)})"
            )

    placements = result.placements
    if type(placements) is not tuple:
        errors.append("result.placements is not a tuple")
        return tuple(errors)

    seen_ids: dict[int, int] = {}
    successful: list[tuple[int, Placement]] = []
    recomputed_placed_count = 0
    recomputed_failed_count = 0
    recomputed_volume = 0
    aggregation_safe = container_ok

    for index, placement in enumerate(placements):
        if type(placement) is not Placement:
            errors.append(f"placement[{index}] is not a Placement")
            aggregation_safe = False
            continue
        box = placement.box
        if type(box) is not Box:
            errors.append(f"placement[{index}].box is not a Box")
            aggregation_safe = False
            continue

        box_ok = True
        for name, value, minimum in (
            ("id", box.id, 0), ("w", box.w, 1), ("h", box.h, 1), ("d", box.d, 1)
        ):
            problem = _int_field_error(value, minimum)
            if problem:
                errors.append(f"placement[{index}].box.{name} {problem}")
                box_ok = False
        if box_ok:
            previous = seen_ids.get(box.id)
            if previous is not None:
                errors.append(f"duplicate box id {_safe_text(box.id)} at indices {previous} and {index}")
            else:
                seen_ids[box.id] = index
        else:
            aggregation_safe = False

        pos = placement.position
        if pos is None:
            recomputed_failed_count += 1
            continue
        if type(pos) is not Point:
            errors.append(f"placement[{index}] id={_safe_text(box.id)} position is neither Point nor None")
            aggregation_safe = False
            continue

        point_ok = True
        for name, value in (("x", pos.x), ("y", pos.y), ("z", pos.z)):
            problem = _int_field_error(value, 0)
            if problem:
                errors.append(f"placement[{index}].position.{name} {problem}")
                point_ok = False
        if not point_ok:
            aggregation_safe = False
            continue

        recomputed_placed_count += 1
        if box_ok:
            recomputed_volume += box.w * box.h * box.d
        else:
            aggregation_safe = False
        if box_ok and container_ok:
            successful.append((index, placement))
            if pos.x + box.w > container.W or pos.y + box.h > container.H or pos.z + box.d > container.D:
                errors.append(
                    f"placement[{index}] id={_safe_text(box.id)} is outside container: "
                    f"position=({_safe_text(pos.x)},{_safe_text(pos.y)},{_safe_text(pos.z)}) "
                    f"size=({_safe_text(box.w)},{_safe_text(box.h)},{_safe_text(box.d)})"
                )

    # Independent pairwise overlap test using positive common interval lengths.
    for left in range(len(successful)):
        i, a = successful[left]
        assert a.position is not None
        ap = a.position
        ab = a.box
        for right in range(left + 1, len(successful)):
            j, b = successful[right]
            assert b.position is not None
            bp = b.position
            bb = b.box
            ox = _intersection_length(ap.x, ap.x + ab.w, bp.x, bp.x + bb.w)
            oy = _intersection_length(ap.y, ap.y + ab.h, bp.y, bp.y + bb.h)
            oz = _intersection_length(ap.z, ap.z + ab.d, bp.z, bp.z + bb.d)
            if ox > 0 and oy > 0 and oz > 0:
                errors.append(
                    f"placements[{i}] id={_safe_text(ab.id)} and [{j}] id={_safe_text(bb.id)} overlap with positive volume"
                )

    total = len(placements)
    if recomputed_placed_count + recomputed_failed_count != total:
        errors.append(
            "placed_count + failed_count does not equal placement count: "
            f"{recomputed_placed_count} + {recomputed_failed_count} != {total}"
        )

    # Compare model-derived values against an independent aggregation only when
    # every primitive needed for the calculation passed the contract checks.
    if aggregation_safe:
        try:
            if result.placed_count != recomputed_placed_count:
                errors.append(
                    f"placed_count mismatch: property={result.placed_count} recomputed={recomputed_placed_count}"
                )
            if result.failed_count != recomputed_failed_count:
                errors.append(
                    f"failed_count mismatch: property={result.failed_count} recomputed={recomputed_failed_count}"
                )
            if result.placed_volume != recomputed_volume:
                errors.append(
                    f"placed_volume mismatch: property={result.placed_volume} recomputed={recomputed_volume}"
                )
            container_volume = container.W * container.H * container.D
            if recomputed_volume > container_volume:
                errors.append("density outside [0,1]: placed volume exceeds container volume")

            # Keep the model's documented float density semantics for valid-size
            # values, but do not let an invalid gigantic result escape validation
            # as a raw conversion/overflow exception.  The integer comparison
            # above still records the legality failure when float conversion is
            # impossible.
            expected_density = recomputed_volume / container_volume
            actual_density = result.density
            if actual_density != expected_density:
                errors.append(f"density mismatch: property={actual_density} recomputed={expected_density}")
        except (TypeError, ValueError, AttributeError, ZeroDivisionError, OverflowError) as exc:
            errors.append(f"could not safely evaluate derived values: {type(exc).__name__}: {exc}")

    if expected_boxes is not None:
        if len(placements) != len(expected_boxes):
            errors.append(
                f"placement count mismatch: result={len(placements)} expected={len(expected_boxes)}"
            )
        for index, (placement, expected) in enumerate(zip(placements, expected_boxes)):
            if type(placement) is not Placement or type(placement.box) is not Box:
                continue
            actual = placement.box
            if (actual.id, actual.w, actual.h, actual.d) != (expected.id, expected.w, expected.h, expected.d):
                errors.append(
                    f"box mismatch at index {index}: "
                    f"result=({_safe_text(actual.id)},{_safe_text(actual.w)},{_safe_text(actual.h)},{_safe_text(actual.d)}) "
                    f"expected=({_safe_text(expected.id)},{_safe_text(expected.w)},{_safe_text(expected.h)},{_safe_text(expected.d)})"
                )

    return tuple(errors)


def validate_support(
    result: PackingResult,
    *,
    settings: DecoderConfig,
) -> tuple[str, ...]:
    """Check final support using explicit settings after ordinary geometry checks.

    This verifies the final support area for each successful placement. It does
    not infer the order in which boxes were accepted or validate decoder
    candidate minimality.
    """
    if type(settings) is not DecoderConfig:
        raise TypeError("settings must be a DecoderConfig")

    errors = validate_result(result)
    if errors:
        return errors
    if settings.min_support_percent == 0:
        return ()

    placements = result.placements
    support_errors: list[str] = []
    for index, placement in enumerate(placements):
        position = placement.position
        if position is None:
            continue
        box = placement.box
        support_area = _independent_support_area(box, position, placements)
        if support_area * 100 < box.w * box.d * settings.min_support_percent:
            support_errors.append(
                f"placement[{index}] does not meet the configured support requirement"
            )
    return tuple(support_errors)
