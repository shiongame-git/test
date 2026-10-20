from pathlib import Path

import pytest

from packing import InputFormatError, read_box_sizes


def test_read_one_line_and_assign_ids(tmp_path: Path):
    p = tmp_path / "box.txt"
    p.write_text("{20,20,20}{5,4,3}{1,2,3}", encoding="utf-8")
    c, boxes = read_box_sizes(p)
    assert (c.W, c.H, c.D) == (20, 20, 20)
    assert [(b.id, b.w, b.h, b.d) for b in boxes] == [(1, 5, 4, 3), (2, 1, 2, 3)]


def test_whitespace_bom_crlf_and_container_only(tmp_path: Path):
    p = tmp_path / "box.txt"
    p.write_bytes(b"\xef\xbb\xbf{ 2 ,\t3, 4 }\r\n")
    c, boxes = read_box_sizes(str(p))
    assert (c.W, c.H, c.D) == (2, 3, 4)
    assert boxes == ()


@pytest.mark.parametrize("content", ["", "  \n", "{1,2}", "{1,2,3}x", "{1,2,3}{0,1,1}", "{+1,2,3}", "{1.0,2,3}", "{1e0,2,3}"])
def test_malformed_inputs_rejected(tmp_path: Path, content: str):
    p = tmp_path / "bad.txt"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(InputFormatError) as exc:
        read_box_sizes(p)
    assert str(p) in str(exc.value)
    assert ":" in str(exc.value)


def test_public_integer_limit_accepts_1000_digits(tmp_path: Path):
    digits = "9" * 1000
    p = tmp_path / "box.txt"
    p.write_text("{" + digits + ",1,1}", encoding="utf-8")
    container, boxes = read_box_sizes(p)
    assert container.W == 10**1000 - 1
    assert (container.H, container.D) == (1, 1)
    assert boxes == ()


def test_public_integer_limit_rejects_1001_digits_before_conversion_with_location(tmp_path: Path):
    digits = "9" * 1001
    p = tmp_path / "box.txt"
    p.write_text("\n  {" + digits + ",1,1}", encoding="utf-8")
    with pytest.raises(InputFormatError) as exc:
        read_box_sizes(p)
    message = str(exc.value)
    assert f"{p}:2:4:" in message
    assert "1000" in message
