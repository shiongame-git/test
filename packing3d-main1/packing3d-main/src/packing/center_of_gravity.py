"""Center-of-gravity calculations for a packed container.

Calculations include successfully placed boxes and use each box's geometric
center. A box's explicit weight is used when available; otherwise volume is
used as a uniform-density fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
import random
from collections.abc import Iterable

from .model import Box, PackingResult


UNITY_MIN_BOX_WEIGHT = 1.0
UNITY_MAX_BOX_WEIGHT = 10.0


def with_unity_random_weights(boxes: Iterable[Box], seed: int) -> tuple[Box, ...]:
    """Assign a stable Unity-style random weight in [1, 10] to each box ID.

    The mapping is generated once per run, independently of the GA RNG, and is
    stable for a fixed set of IDs and seed. Explicit Box.weight values win.
    """
    frozen = tuple(boxes)
    if any(type(box) is not Box for box in frozen):
        raise TypeError("boxes must contain only Box instances")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a Python int >= 0")
    weight_rng = random.Random(seed ^ 0x554E495459434F47)
    weights = {
        box_id: weight_rng.uniform(UNITY_MIN_BOX_WEIGHT, UNITY_MAX_BOX_WEIGHT)
        for box_id in sorted(box.id for box in frozen)
    }
    return tuple(
        box if box.weight is not None else replace(box, weight=weights[box.id])
        for box in frozen
    )


@dataclass(frozen=True, slots=True)
class CenterOfGravity:
    """Mass-weighted center of gravity in container coordinates."""

    x: float
    y: float
    z: float


def calculate_center_of_gravity(result: PackingResult) -> CenterOfGravity | None:
    """Return mass-weighted CoG, using explicit weights or volume as fallback."""
    if type(result) is not PackingResult:
        raise TypeError("result must be a PackingResult")

    total_mass = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    weighted_z = 0.0

    for placement in result.placements:
        point = placement.position
        if point is None:
            continue
        mass = placement.box.weight if placement.box.weight is not None else float(placement.box.volume)
        weighted_x += (point.x + placement.box.w / 2.0) * mass
        weighted_y += (point.y + placement.box.h / 2.0) * mass
        weighted_z += (point.z + placement.box.d / 2.0) * mass
        total_mass += mass

    if total_mass == 0:
        return None
    return CenterOfGravity(
        weighted_x / total_mass,
        weighted_y / total_mass,
        weighted_z / total_mass,
    )


def calculate_center_distance(result: PackingResult) -> float | None:
    """Return CoG distance from the container's geometric center.

    The result is in the same length units as the container dimensions. Returns
    ``None`` when no boxes have been placed.
    """
    center = calculate_center_of_gravity(result)
    if center is None:
        return None

    container = result.container
    dx = center.x - container.W / 2.0
    dy = center.y - container.H / 2.0
    dz = center.z - container.D / 2.0
    return sqrt(dx * dx + dy * dy + dz * dz)


def calculate_balance_distance(result: PackingResult) -> float | None:
    """Normalized distance from the low, horizontal center used by the Unity GA.

    The target is (container.W / 2, 0, container.D / 2): centered on the
    floor and kept low. Explicit box weights are used when present; otherwise
    volume weighting assumes uniform density.
    The Euclidean distance is divided by the farthest target-to-corner distance,
    making the result dimensionless and bounded by [0, 1].
    """
    center = calculate_center_of_gravity(result)
    if center is None:
        return None
    container = result.container
    dx = center.x - container.W / 2.0
    dy = center.y
    dz = center.z - container.D / 2.0
    max_distance = sqrt(
        (container.W / 2.0) ** 2 + container.H**2 + (container.D / 2.0) ** 2
    )
    if max_distance == 0.0:
        return 0.0
    return sqrt(dx * dx + dy * dy + dz * dz) / max_distance
