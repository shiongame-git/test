from pathlib import Path

import pytest

from packing import Box, Container, PackingResult, Placement, Point, write_unity_result_data


def test_exact_coordinate_conversion_and_failure_sentinel(tmp_path: Path):
    result = PackingResult(
        Container(20, 20, 20),
        (
            Placement(Box(149, 3, 4, 5), Point(0, 0, 0)),
            Placement(Box(176, 5, 5, 5), None),
        ),
    )
    p = tmp_path / "ResultData.txt"
    write_unity_result_data(p, result)
    assert p.read_text(encoding="utf-8") == (
        "{149:3,4,5};{1.5,2,-2.5}"
        "{176:5,5,5};{-10,-10,-10}\n"
    )


def test_complete_placement_state_is_exactly_one_line(tmp_path: Path):
    result = PackingResult(
        Container(10, 10, 10),
        (
            Placement(Box(1, 1, 1, 1), Point(0, 0, 0)),
            Placement(Box(2, 2, 2, 2), Point(1, 0, 0)),
            Placement(Box(3, 3, 3, 3), None),
        ),
    )
    p = tmp_path / "ResultData.txt"
    write_unity_result_data(p, result)

    text = p.read_text(encoding="utf-8")
    assert text.count("\n") == 1
    assert text.endswith("\n")
    assert text[:-1] == (
        "{1:1,1,1};{0.5,0.5,-0.5}"
        "{2:2,2,2};{2,1,-1}"
        "{3:3,3,3};{-10,-10,-10}"
    )


def test_axis_movements_and_large_integer_exactness(tmp_path: Path):
    huge = 2**53 + 1
    result = PackingResult(
        Container(huge + 10, 20, 20),
        (Placement(Box(1, 3, 4, 5), Point(huge, 2, 3)),),
    )
    p = tmp_path / "out.txt"
    write_unity_result_data(p, result)
    assert p.read_text(encoding="utf-8") == f"{{1:3,4,5}};{{{huge + 1}.5,4,-5.5}}\n"


def test_empty_result_is_zero_bytes(tmp_path: Path):
    p = tmp_path / "out.txt"
    write_unity_result_data(p, PackingResult(Container(1, 1, 1), ()))
    assert p.read_bytes() == b""


def test_validation_failure_does_not_modify_existing_file(tmp_path: Path):
    p = tmp_path / "out.txt"
    p.write_text("keep", encoding="utf-8")
    bad = PackingResult(Container(1, 1, 1), (Placement(Box(1, 1, 1, 1), Point(1, 0, 0)),))
    with pytest.raises(ValueError):
        write_unity_result_data(p, bad)
    assert p.read_text(encoding="utf-8") == "keep"


def test_gigantic_invalid_result_export_does_not_modify_existing_file(tmp_path: Path):
    p = tmp_path / "out.txt"
    p.write_bytes(b"keep")
    bad = PackingResult(
        Container(1, 1, 1),
        (Placement(Box(1, 10**309, 1, 1), Point(0, 0, 0)),),
    )
    with pytest.raises(ValueError):
        write_unity_result_data(p, bad)
    assert p.read_bytes() == b"keep"


def test_public_integer_limit_unity_rejects_without_modifying_destination(tmp_path: Path):
    p = tmp_path / "ResultData.txt"
    p.write_bytes(b"keep")
    result = PackingResult(Container(10**1000, 1, 1), ())
    with pytest.raises(ValueError, match="1000"):
        write_unity_result_data(p, result)
    assert p.read_bytes() == b"keep"


def test_unity_1000_digit_id_works_under_lower_runtime_digit_guard(tmp_path: Path):
    import sys

    if not hasattr(sys, "set_int_max_str_digits"):
        pytest.skip("runtime has no configurable int string digit guard")

    original = sys.get_int_max_str_digits()
    huge = 10**999
    result = PackingResult(
        Container(huge, 1, 1),
        (Placement(Box(huge, 1, 1, 1), Point(0, 0, 0)),),
    )
    p = tmp_path / "ResultData.txt"
    try:
        sys.set_int_max_str_digits(640)
        write_unity_result_data(p, result)
        assert sys.get_int_max_str_digits() == 640
    finally:
        sys.set_int_max_str_digits(original)
    text = p.read_text(encoding="utf-8")
    assert text.startswith("{" + "1" + "0" * 999 + ":1,1,1};")
