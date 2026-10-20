"""Strict packing-result JSON v1 reader/writer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import InputFormatError
from ._io_integer import (
    PublicIntegerTokenTooLong,
    parse_public_decimal_token,
    public_int_text,
    require_public_io_int,
)
from .model import Box, Container, PackingResult, Placement, Point
from .validate import validate_result

_ROOT_KEYS = ("format", "version", "container", "placements")
_CONTAINER_KEYS = ("width", "height", "depth")
_PLACEMENT_KEYS = ("id", "size", "position")
_SIZE_KEYS = ("width", "height", "depth")
_POSITION_KEYS = ("x", "y", "z")


class _DuplicateKey(ValueError):
    pass


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise _DuplicateKey(f"duplicate key {key!r}")
        obj[key] = value
    return obj


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-standard numeric constant {token!r} is not allowed")


def _parse_json_int(token: str) -> int:
    return parse_public_decimal_token(token)


def _path(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be str or os.PathLike")
    return Path(path)


def _fmt_error(filename: Path, reason: str, *, path: str | None = None) -> InputFormatError:
    where = f" at {path}" if path else ""
    return InputFormatError(f"{filename}{where}: {reason}")


def _require_exact_keys(obj: Any, expected: tuple[str, ...], filename: Path, path: str) -> dict[str, Any]:
    if type(obj) is not dict:
        raise _fmt_error(filename, "expected JSON object", path=path)
    actual = set(obj)
    wanted = set(expected)
    missing = wanted - actual
    extra = actual - wanted
    if missing:
        raise _fmt_error(filename, f"missing key(s): {', '.join(sorted(missing))}", path=path)
    if extra:
        raise _fmt_error(filename, f"unknown key(s): {', '.join(sorted(extra))}", path=path)
    return obj


def _require_int(value: Any, filename: Path, path: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise _fmt_error(filename, "expected JSON integer (no float, exponent, string, or boolean conversion)", path=path)
    if value < minimum:
        raise _fmt_error(filename, f"value must be >= {minimum}", path=path)
    return value


def read_result_json(path: str | os.PathLike[str]) -> PackingResult:
    p = _path(path)
    try:
        raw = p.read_bytes()
    except OSError:
        raise
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputFormatError(f"{p}:1:1: input is not valid UTF-8: {exc.reason}") from exc
    if not text.strip():
        raise InputFormatError(f"{p}:1:1: file is empty or contains only whitespace")

    try:
        data = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
            parse_int=_parse_json_int,
        )
    except json.JSONDecodeError as exc:
        raise InputFormatError(f"{p}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}") from exc
    except PublicIntegerTokenTooLong as exc:
        raise InputFormatError(f"{p}: {exc}") from exc
    except (_DuplicateKey, ValueError) as exc:
        raise InputFormatError(f"{p}: {exc}") from exc

    root = _require_exact_keys(data, _ROOT_KEYS, p, "$")
    if type(root["format"]) is not str or root["format"] != "packing-result":
        raise _fmt_error(p, "format must be exactly 'packing-result'", path="$.format")
    version = _require_int(root["version"], p, "$.version", minimum=0)
    if version != 1:
        raise _fmt_error(
            p,
            f"unsupported version {public_int_text(version, name='version')}; only version 1 is supported",
            path="$.version",
        )

    cobj = _require_exact_keys(root["container"], _CONTAINER_KEYS, p, "$.container")
    W = _require_int(cobj["width"], p, "$.container.width", minimum=1)
    H = _require_int(cobj["height"], p, "$.container.height", minimum=1)
    D = _require_int(cobj["depth"], p, "$.container.depth", minimum=1)
    container = Container(W, H, D)

    values = root["placements"]
    if type(values) is not list:
        raise _fmt_error(p, "expected JSON array", path="$.placements")

    placements: list[Placement] = []
    seen: set[int] = set()
    for index, item in enumerate(values):
        base = f"$.placements[{index}]"
        pobj = _require_exact_keys(item, _PLACEMENT_KEYS, p, base)
        box_id = _require_int(pobj["id"], p, f"{base}.id", minimum=0)
        if box_id in seen:
            raise _fmt_error(
                p,
                f"duplicate box id {public_int_text(box_id, name=f'{base}.id')}",
                path=f"{base}.id",
            )
        seen.add(box_id)

        sobj = _require_exact_keys(pobj["size"], _SIZE_KEYS, p, f"{base}.size")
        w = _require_int(sobj["width"], p, f"{base}.size.width", minimum=1)
        h = _require_int(sobj["height"], p, f"{base}.size.height", minimum=1)
        d = _require_int(sobj["depth"], p, f"{base}.size.depth", minimum=1)
        box = Box(box_id, w, h, d)

        position_value = pobj["position"]
        if position_value is None:
            position = None
        else:
            posobj = _require_exact_keys(position_value, _POSITION_KEYS, p, f"{base}.position")
            x = _require_int(posobj["x"], p, f"{base}.position.x", minimum=0)
            y = _require_int(posobj["y"], p, f"{base}.position.y", minimum=0)
            z = _require_int(posobj["z"], p, f"{base}.position.z", minimum=0)
            position = Point(x, y, z)
        placements.append(Placement(box, position))

    result = PackingResult(container, tuple(placements))
    errors = validate_result(result)
    if errors:
        raise _fmt_error(p, "illegal packing result: " + "; ".join(errors), path="$")
    return result


def _validate_public_io_integers(result: PackingResult) -> None:
    require_public_io_int(result.container.W, name="result.container.W")
    require_public_io_int(result.container.H, name="result.container.H")
    require_public_io_int(result.container.D, name="result.container.D")
    for index, placement in enumerate(result.placements):
        box = placement.box
        require_public_io_int(box.id, name=f"result.placements[{index}].box.id")
        require_public_io_int(box.w, name=f"result.placements[{index}].box.w")
        require_public_io_int(box.h, name=f"result.placements[{index}].box.h")
        require_public_io_int(box.d, name=f"result.placements[{index}].box.d")
        if placement.position is not None:
            pos = placement.position
            require_public_io_int(pos.x, name=f"result.placements[{index}].position.x")
            require_public_io_int(pos.y, name=f"result.placements[{index}].position.y")
            require_public_io_int(pos.z, name=f"result.placements[{index}].position.z")


def _render_result_json(result: PackingResult) -> str:
    number = lambda value, name: public_int_text(value, name=name)
    lines = [
        "{",
        '  "format": "packing-result",',
        '  "version": 1,',
        '  "container": {',
        f'    "width": {number(result.container.W, "result.container.W")},',
        f'    "height": {number(result.container.H, "result.container.H")},',
        f'    "depth": {number(result.container.D, "result.container.D")}',
        "  },",
    ]
    if not result.placements:
        lines.append('  "placements": []')
        lines.append("}")
        return "\n".join(lines) + "\n"

    lines.append('  "placements": [')
    for index, placement in enumerate(result.placements):
        box = placement.box
        lines.extend([
            "    {",
            f'      "id": {number(box.id, f"result.placements[{index}].box.id")},',
            '      "size": {',
            f'        "width": {number(box.w, f"result.placements[{index}].box.w")},',
            f'        "height": {number(box.h, f"result.placements[{index}].box.h")},',
            f'        "depth": {number(box.d, f"result.placements[{index}].box.d")}',
            "      },",
        ])
        if placement.position is None:
            lines.append('      "position": null')
        else:
            pos = placement.position
            lines.extend([
                '      "position": {',
                f'        "x": {number(pos.x, f"result.placements[{index}].position.x")},',
                f'        "y": {number(pos.y, f"result.placements[{index}].position.y")},',
                f'        "z": {number(pos.z, f"result.placements[{index}].position.z")}',
                "      }",
            ])
        lines.append("    }," if index + 1 < len(result.placements) else "    }")
    lines.extend(["  ]", "}"])
    return "\n".join(lines) + "\n"


def write_result_json(path: str | os.PathLike[str], result: PackingResult) -> None:
    p = _path(path)
    if type(result) is not PackingResult:
        raise TypeError("result must be a PackingResult")
    # Check the public decimal bound before validation can stringify a gigantic
    # illegal value in an error description.
    _validate_public_io_integers(result)
    errors = validate_result(result)
    if errors:
        raise ValueError("cannot write illegal PackingResult: " + "; ".join(errors))

    text = _render_result_json(result)
    # Validate fully before opening the destination so contract failures do not alter it.
    with p.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
