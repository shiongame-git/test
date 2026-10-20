"""One-way export from PackingResult to legacy Unity display ResultData."""

from __future__ import annotations

import os
from pathlib import Path

from ._io_integer import public_int_text, require_public_io_int
from .model import PackingResult
from .validate import validate_result


def _path(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be str or os.PathLike")
    return Path(path)


def _format_half_units(value2: int) -> str:
    sign = "-" if value2 < 0 else ""
    magnitude = abs(value2)
    whole, half = divmod(magnitude, 2)
    whole_text = public_int_text(whole, name="Unity coordinate integer part")
    return f"{sign}{whole_text}.5" if half else f"{sign}{whole_text}"


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


def write_unity_result_data(path: str | os.PathLike[str], result: PackingResult) -> None:
    p = _path(path)
    if type(result) is not PackingResult:
        raise TypeError("result must be a PackingResult")
    _validate_public_io_integers(result)
    errors = validate_result(result)
    if errors:
        raise ValueError("cannot export illegal PackingResult: " + "; ".join(errors))

    records: list[str] = []
    for placement in result.placements:
        b = placement.box
        if placement.position is None:
            coord = "{-10,-10,-10}"
        else:
            pos = placement.position
            x2 = 2 * pos.x + b.w
            y2 = 2 * pos.y + b.h
            z2 = -(2 * pos.z + b.d)
            coord = "{" + ",".join((_format_half_units(x2), _format_half_units(y2), _format_half_units(z2))) + "}"
        box_text = ":".join((
            public_int_text(b.id, name=f"result.placements[{len(records)}].box.id"),
            ",".join((
                public_int_text(b.w, name=f"result.placements[{len(records)}].box.w"),
                public_int_text(b.h, name=f"result.placements[{len(records)}].box.h"),
                public_int_text(b.d, name=f"result.placements[{len(records)}].box.d"),
            )),
        ))
        records.append("{" + box_text + "};" + coord)

    # Legacy GA/Unity ResultData uses one complete placement state per line.
    # A PackingResult represents one such state, so concatenate all box
    # records without separators and terminate the state with one newline.
    text = "".join(records)
    if records:
        text += "\n"
    with p.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
