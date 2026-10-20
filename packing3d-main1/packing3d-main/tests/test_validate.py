import pytest

from packing import Box, Container, PackingResult, Placement, Point, pack, validate_result


def test_contacts_are_legal_but_positive_volume_overlap_is_not():
    c = Container(10, 10, 10)
    a = Placement(Box(1, 2, 2, 2), Point(0, 0, 0))
    for point in [Point(2, 0, 0), Point(2, 2, 0), Point(2, 2, 2), Point(1, 1, 3), Point(1, 3, 3)]:
        b = Placement(Box(2, 2, 2, 2), point)
        assert validate_result(PackingResult(c, (a, b))) == ()
    for point in [Point(1, 1, 1), Point(0, 0, 0)]:
        b = Placement(Box(2, 2, 2, 2), point)
        errors = validate_result(PackingResult(c, (a, b)))
        assert any("overlap" in e for e in errors)


def test_outside_container_and_density_above_one_detected():
    c = Container(2, 2, 2)
    bad = PackingResult(c, (Placement(Box(1, 2, 2, 2), Point(1, 0, 0)),))
    errors = validate_result(bad)
    assert any("outside" in e for e in errors)


def test_expected_container_and_ordered_boxes_checks():
    boxes = (Box(7, 1, 1, 1), Box(2, 1, 1, 1))
    result = pack(Container(2, 1, 1), boxes)
    assert validate_result(result, expected_container=Container(2, 1, 1), ordered_boxes=boxes) == ()
    assert any("container mismatch" in e for e in validate_result(result, expected_container=Container(3, 1, 1)))
    assert any("box mismatch" in e for e in validate_result(result, ordered_boxes=tuple(reversed(boxes))))


def test_all_none_can_be_legal_without_bl_minimality_check():
    boxes = (Box(1, 1, 1, 1), Box(2, 1, 1, 1))
    result = PackingResult(Container(2, 1, 1), tuple(Placement(b, None) for b in boxes))
    assert validate_result(result, ordered_boxes=boxes) == ()
    assert result != pack(result.container, boxes)


def test_existing_geometry_validator_does_not_apply_support_threshold():
    result = PackingResult(
        Container(2, 2, 2),
        (Placement(Box(1, 1, 1, 1), Point(1, 1, 1)),),
    )

    assert validate_result(result) == ()


def test_duplicate_id_in_result_detected():
    result = PackingResult(
        Container(2, 2, 2),
        (Placement(Box(1, 1, 1, 1), None), Placement(Box(1, 1, 1, 1), None)),
    )
    assert any("duplicate" in e for e in validate_result(result))


def test_wall_contact_one_unit_overflow_and_containment():
    c = Container(4, 4, 4)
    wall_touch = PackingResult(c, (Placement(Box(1, 2, 2, 2), Point(2, 2, 2)),))
    assert validate_result(wall_touch) == ()

    overflow = PackingResult(c, (Placement(Box(1, 2, 2, 2), Point(3, 2, 2)),))
    assert any("outside" in e for e in validate_result(overflow))

    containing = PackingResult(
        c,
        (
            Placement(Box(1, 4, 4, 4), Point(0, 0, 0)),
            Placement(Box(2, 1, 1, 1), Point(1, 1, 1)),
        ),
    )
    assert any("overlap" in e for e in validate_result(containing))


def test_gigantic_invalid_result_returns_errors_instead_of_overflow():
    result = PackingResult(
        Container(1, 1, 1),
        (Placement(Box(1, 10**309, 1, 1), Point(0, 0, 0)),),
    )
    errors = validate_result(result)
    assert any("outside" in error for error in errors)
    assert any("density outside [0,1]" in error for error in errors)
    assert any("could not safely evaluate derived values" in error for error in errors)


def test_validator_diagnostics_do_not_depend_on_runtime_int_string_limit():
    import sys

    if not hasattr(sys, "set_int_max_str_digits"):
        pytest.skip("runtime has no configurable int string digit guard")

    original = sys.get_int_max_str_digits()
    huge = 10**999
    result = PackingResult(
        Container(1, 1, 1),
        (Placement(Box(1, huge, 1, 1), Point(0, 0, 0)),),
    )
    try:
        sys.set_int_max_str_digits(640)
        errors = validate_result(result)
    finally:
        sys.set_int_max_str_digits(original)

    assert any("outside" in error for error in errors)
    assert any("density outside [0,1]" in error for error in errors)
    assert any("could not safely evaluate derived values" in error for error in errors)
