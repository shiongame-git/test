from __future__ import annotations

import math

import pytest

from packing import (
    Box,
    Container,
    PackingResult,
    Placement,
    Point,
    calculate_center_distance,
    calculate_center_of_gravity,
    calculate_balance_distance,
)


def test_center_of_gravity_single_box() -> None:
    result = PackingResult(
        Container(10, 10, 10),
        (Placement(Box(0, 2, 4, 6), Point(1, 2, 3)),),
    )
    center = calculate_center_of_gravity(result)
    assert center is not None
    assert (center.x, center.y, center.z) == (2.0, 4.0, 6.0)


def test_center_of_gravity_is_volume_weighted() -> None:
    result = PackingResult(
        Container(10, 10, 10),
        (
            Placement(Box(0, 2, 2, 2), Point(0, 0, 0)),
            Placement(Box(1, 4, 2, 2), Point(4, 0, 0)),
            Placement(Box(2, 1, 1, 1), None),
        ),
    )
    center = calculate_center_of_gravity(result)
    assert center is not None
    # Volumes are 8 and 16; their x-centers are 1 and 6.
    assert center.x == pytest.approx((1 * 8 + 6 * 16) / 24)
    assert center.y == pytest.approx(1.0)
    assert center.z == pytest.approx(1.0)


def test_empty_result_has_no_center_or_distance() -> None:
    result = PackingResult(Container(10, 10, 10), (Placement(Box(0, 1, 1, 1), None),))
    assert calculate_center_of_gravity(result) is None
    assert calculate_center_distance(result) is None


def test_center_distance_from_container_center() -> None:
    result = PackingResult(
        Container(10, 10, 10),
        (Placement(Box(0, 2, 2, 2), Point(0, 0, 0)),),
    )
    assert calculate_center_distance(result) == pytest.approx(math.sqrt(48))


def test_balance_distance_is_normalized_by_container_dimensions() -> None:
    small = PackingResult(
        Container(10, 10, 10),
        (Placement(Box(0, 2, 2, 2), Point(0, 0, 0)),),
    )
    large = PackingResult(
        Container(20, 20, 20),
        (Placement(Box(0, 4, 4, 4), Point(0, 0, 0)),),
    )
    distance = math.sqrt(4**2 + 1**2 + 4**2)
    scale = math.sqrt(5**2 + 10**2 + 5**2)
    assert calculate_balance_distance(small) == pytest.approx(distance / scale)
    assert calculate_balance_distance(large) == pytest.approx(distance / scale)
    assert 0 <= calculate_balance_distance(small) <= 1


def test_invalid_result_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="result must be a PackingResult"):
        calculate_center_of_gravity(object())  # type: ignore[arg-type]
