from __future__ import annotations

import json
from pathlib import Path

import pytest

from packing import (
    Box, Container, InputFormatError, PackingResult, Placement, Point,
    read_result_json, write_result_json,
)


def sample_result() -> PackingResult:
    return PackingResult(
        Container(20, 20, 20),
        (
            Placement(Box(7, 5, 4, 3), Point(0, 0, 0)),
            Placement(Box(2, 3, 3, 3), None),
        ),
    )


def test_round_trip_canonical_and_preserves_order(tmp_path: Path):
    p = tmp_path / "result.json"
    result = sample_result()
    write_result_json(p, result)
    text1 = p.read_text(encoding="utf-8")
    assert text1.endswith("\n")
    assert not p.read_bytes().startswith(b"\xef\xbb\xbf")
    loaded = read_result_json(p)
    assert loaded == result
    assert [x.box.id for x in loaded.placements] == [7, 2]
    write_result_json(p, result)
    assert p.read_text(encoding="utf-8") == text1


def test_empty_result_roundtrip(tmp_path: Path):
    p = tmp_path / "empty.json"
    r = PackingResult(Container(1, 1, 1), ())
    write_result_json(p, r)
    assert read_result_json(p) == r
    assert '"placements": []' in p.read_text(encoding="utf-8")


def minimal_json(position='{"x":0,"y":0,"z":0}', extra_root="") -> str:
    return (
        '{"format":"packing-result","version":1,'
        '"container":{"width":2,"height":2,"depth":2},'
        f'"placements":[{{"id":1,"size":{{"width":1,"height":1,"depth":1}},"position":{position}}}]'
        f'{extra_root}}}'
    )


@pytest.mark.parametrize("text", [
    minimal_json().replace('"version":1', '"version":2'),
    minimal_json().replace('"packing-result"', '"other"'),
    minimal_json().replace('"version":1', '"version":1.0'),
    minimal_json().replace('"version":1', '"version":1e0'),
    minimal_json().replace('"version":1', '"version":true'),
    minimal_json().replace('"width":2', '"width":"2"'),
    minimal_json(position='{"x":-1,"y":0,"z":0}'),
    minimal_json(position='{"x":0.0,"y":0,"z":0}'),
    minimal_json(position='{"x":0,"y":0,"z":0,"extra":1}'),
    minimal_json(extra_root=',"extra":1'),
    minimal_json() + '{}',
    '',
    '   ',
])
def test_invalid_json_schema_and_types(tmp_path: Path, text: str):
    p = tmp_path / "bad.json"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(InputFormatError):
        read_result_json(p)


def test_duplicate_keys_and_nonstandard_numbers_rejected(tmp_path: Path):
    for text in [
        '{"format":"packing-result","format":"packing-result","version":1,"container":{"width":1,"height":1,"depth":1},"placements":[]}',
        '{"format":"packing-result","version":1,"container":{"width":NaN,"height":1,"depth":1},"placements":[]}',
        '{"format":"packing-result","version":1,"container":{"width":Infinity,"height":1,"depth":1},"placements":[]}',
    ]:
        p = tmp_path / "bad.json"
        p.write_text(text, encoding="utf-8")
        with pytest.raises(InputFormatError):
            read_result_json(p)


def test_duplicate_ids_outside_and_overlap_rejected(tmp_path: Path):
    cases = [
        {
            "format":"packing-result","version":1,
            "container":{"width":2,"height":2,"depth":2},
            "placements":[
                {"id":1,"size":{"width":1,"height":1,"depth":1},"position":None},
                {"id":1,"size":{"width":1,"height":1,"depth":1},"position":None},
            ],
        },
        {
            "format":"packing-result","version":1,
            "container":{"width":2,"height":2,"depth":2},
            "placements":[{"id":1,"size":{"width":2,"height":2,"depth":2},"position":{"x":1,"y":0,"z":0}}],
        },
        {
            "format":"packing-result","version":1,
            "container":{"width":2,"height":2,"depth":2},
            "placements":[
                {"id":1,"size":{"width":2,"height":2,"depth":2},"position":{"x":0,"y":0,"z":0}},
                {"id":2,"size":{"width":1,"height":1,"depth":1},"position":{"x":0,"y":0,"z":0}},
            ],
        },
    ]
    for i, data in enumerate(cases):
        p = tmp_path / f"bad{i}.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(InputFormatError):
            read_result_json(p)


def test_large_integer_roundtrip_without_float(tmp_path: Path):
    huge = 2**53 + 1
    result = PackingResult(Container(huge + 2, 2, 2), (Placement(Box(1, 1, 1, 1), Point(huge, 0, 0)),))
    p = tmp_path / "huge.json"
    write_result_json(p, result)
    loaded = read_result_json(p)
    assert loaded == result
    assert str(huge) in p.read_text(encoding="utf-8")


def test_write_validation_failure_does_not_modify_existing_file(tmp_path: Path):
    p = tmp_path / "result.json"
    p.write_text("keep me", encoding="utf-8")
    bad = PackingResult(Container(2, 2, 2), (Placement(Box(1, 2, 2, 2), Point(1, 0, 0)),))
    with pytest.raises(ValueError):
        write_result_json(p, bad)
    assert p.read_text(encoding="utf-8") == "keep me"


def test_gigantic_invalid_result_read_is_input_format_error(tmp_path: Path):
    huge = 10**309
    p = tmp_path / "bad-huge.json"
    p.write_text(
        '{"format":"packing-result","version":1,'
        '"container":{"width":1,"height":1,"depth":1},'
        + f'"placements":[{{"id":1,"size":{{"width":{huge},"height":1,"depth":1}},'
        '"position":{"x":0,"y":0,"z":0}}]}',
        encoding="utf-8",
    )
    with pytest.raises(InputFormatError):
        read_result_json(p)


def test_gigantic_invalid_result_write_does_not_modify_existing_file(tmp_path: Path):
    p = tmp_path / "result.json"
    p.write_bytes(b"keep me")
    bad = PackingResult(
        Container(1, 1, 1),
        (Placement(Box(1, 10**309, 1, 1), Point(0, 0, 0)),),
    )
    with pytest.raises(ValueError):
        write_result_json(p, bad)
    assert p.read_bytes() == b"keep me"


def _json_with_integer(field: str, token: str) -> str:
    values = {
        "container_width": "2",
        "id": "1",
        "x": "0",
    }
    values[field] = token
    return (
        '{"format":"packing-result","version":1,'
        f'"container":{{"width":{values["container_width"]},"height":2,"depth":2}},'
        f'"placements":[{{"id":{values["id"]},"size":{{"width":1,"height":1,"depth":1}},'
        f'"position":{{"x":{values["x"]},"y":0,"z":0}}}}]}}'
    )


def test_public_integer_limit_1000_digit_id_roundtrips(tmp_path: Path):
    digits = "9" * 1000
    p = tmp_path / "large.json"
    p.write_text(_json_with_integer("id", digits), encoding="utf-8")
    result = read_result_json(p)
    assert result.placements[0].box.id == 10**1000 - 1

    out = tmp_path / "out.json"
    write_result_json(out, result)
    loaded = read_result_json(out)
    assert loaded == result
    assert digits in out.read_text(encoding="utf-8")


@pytest.mark.parametrize("field", ["container_width", "id", "x"])
def test_public_integer_limit_rejects_1001_digit_json_fields(tmp_path: Path, field: str):
    p = tmp_path / f"bad-{field}.json"
    p.write_text(_json_with_integer(field, "9" * 1001), encoding="utf-8")
    with pytest.raises(InputFormatError, match="1000"):
        read_result_json(p)


def test_public_integer_limit_write_rejects_before_modifying_destination(tmp_path: Path):
    p = tmp_path / "result.json"
    p.write_bytes(b"keep")
    over_limit = 10**1000
    result = PackingResult(Container(over_limit, 1, 1), ())
    with pytest.raises(ValueError, match="1000"):
        write_result_json(p, result)
    assert p.read_bytes() == b"keep"


def test_public_integer_contract_is_independent_of_lower_runtime_digit_guard(tmp_path: Path):
    import sys

    if not hasattr(sys, "set_int_max_str_digits"):
        pytest.skip("runtime has no configurable int string digit guard")

    original = sys.get_int_max_str_digits()
    huge = 10**999
    result = PackingResult(
        Container(huge, 1, 1),
        (Placement(Box(huge, 1, 1, 1), Point(0, 0, 0)),),
    )
    p = tmp_path / "guard.json"
    try:
        sys.set_int_max_str_digits(640)
        write_result_json(p, result)
        loaded = read_result_json(p)
        assert loaded == result
        assert sys.get_int_max_str_digits() == 640
    finally:
        sys.set_int_max_str_digits(original)


def test_illegal_1000_digit_json_under_lower_runtime_guard_is_input_format_error(tmp_path: Path):
    import sys

    if not hasattr(sys, "set_int_max_str_digits"):
        pytest.skip("runtime has no configurable int string digit guard")

    original = sys.get_int_max_str_digits()
    p = tmp_path / "illegal-large.json"
    p.write_text(_json_with_integer("container_width", "1"), encoding="utf-8")
    huge = "9" * 1000
    p.write_text(
        '{"format":"packing-result","version":1,'
        '"container":{"width":1,"height":1,"depth":1},'
        '"placements":[{"id":1,"size":{"width":' + huge + ',"height":1,"depth":1},'
        '"position":{"x":0,"y":0,"z":0}}]}',
        encoding="utf-8",
    )
    try:
        sys.set_int_max_str_digits(640)
        with pytest.raises(InputFormatError, match="illegal packing result"):
            read_result_json(p)
    finally:
        sys.set_int_max_str_digits(original)
