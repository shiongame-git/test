"""Shared input checks for the BL and PP decoder boundaries.

Keep result legality checking in validate.py independent of decoder geometry.
"""

from collections.abc import Iterable

from .model import Box


def freeze_decoder_boxes(ordered_boxes: Iterable[Box]) -> tuple[Box, ...]:
    """Consume once, preserve order, and reject invalid or duplicate boxes."""
    boxes = tuple(ordered_boxes)
    seen: set[int] = set()
    for index, box in enumerate(boxes):
        if type(box) is not Box:
            raise TypeError(f"ordered_boxes[{index}] must be a Box")
        for name, value, minimum in (
            ("id", box.id, 0), ("w", box.w, 1), ("h", box.h, 1), ("d", box.d, 1)
        ):
            if type(value) is not int:
                raise TypeError(f"ordered_boxes[{index}].{name} must be a Python int")
            if value < minimum:
                raise ValueError(f"ordered_boxes[{index}].{name} must be >= {minimum}")
        if box.id in seen:
            raise ValueError(f"duplicate box id: {box.id}")
        seen.add(box.id)
    return boxes
