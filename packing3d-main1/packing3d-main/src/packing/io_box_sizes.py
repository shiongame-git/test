"""Strict parser for the legacy box_sizes.txt input adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import NoReturn

from .errors import InputFormatError
from ._io_integer import PUBLIC_IO_MAX_DECIMAL_DIGITS, parse_public_decimal_token
from .model import Box, Container


class _Parser:
    def __init__(self, text: str, filename: str) -> None:
        self.text = text
        self.filename = filename
        self.i = 0

    def _line_col(self, index: int | None = None) -> tuple[int, int]:
        pos = self.i if index is None else index
        line = self.text.count("\n", 0, pos) + 1
        last_newline = self.text.rfind("\n", 0, pos)
        col = pos + 1 if last_newline < 0 else pos - last_newline
        return line, col

    def fail(self, reason: str) -> NoReturn:
        line, col = self._line_col()
        raise InputFormatError(f"{self.filename}:{line}:{col}: {reason}")

    def skip_ws(self) -> None:
        while self.i < len(self.text) and self.text[self.i] in " \t\r\n":
            self.i += 1

    def expect(self, ch: str) -> None:
        self.skip_ws()
        if self.i >= len(self.text) or self.text[self.i] != ch:
            self.fail(f"expected {ch!r}")
        self.i += 1

    def integer(self) -> int:
        self.skip_ws()
        start = self.i
        while self.i < len(self.text) and "0" <= self.text[self.i] <= "9":
            self.i += 1
        if self.i == start:
            self.fail("expected an ASCII positive integer")
        token = self.text[start:self.i]
        if len(token) > PUBLIC_IO_MAX_DECIMAL_DIGITS:
            self.i = start
            self.fail(f"integer token exceeds {PUBLIC_IO_MAX_DECIMAL_DIGITS} decimal digits")
        value = parse_public_decimal_token(token)
        if value < 1:
            self.i = start
            self.fail("dimensions must be >= 1")
        return value

    def triple(self) -> tuple[int, int, int]:
        self.expect("{")
        a = self.integer()
        self.expect(",")
        b = self.integer()
        self.expect(",")
        c = self.integer()
        self.expect("}")
        return a, b, c

    def parse_all(self) -> list[tuple[int, int, int]]:
        self.skip_ws()
        if self.i >= len(self.text):
            self.fail("file is empty or contains only whitespace")
        records: list[tuple[int, int, int]] = []
        while True:
            records.append(self.triple())
            self.skip_ws()
            if self.i == len(self.text):
                return records
            if self.text[self.i] != "{":
                self.fail(f"unexpected character {self.text[self.i]!r}")


def _coerce_path(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be str or os.PathLike")
    return Path(path)


def read_box_sizes(path: str | os.PathLike[str]) -> tuple[Container, tuple[Box, ...]]:
    p = _coerce_path(path)
    try:
        raw = p.read_bytes()
    except OSError:
        raise
    return parse_box_sizes_bytes(raw, source=str(p))


def parse_box_sizes_bytes(raw: bytes, *, source: str = "<bytes>") -> tuple[Container, tuple[Box, ...]]:
    """Parse the exact frozen bytes used for recording a fixture digest."""
    if type(raw) is not bytes:
        raise TypeError("raw must be bytes")
    p = source
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputFormatError(
            f"{p}:1:1: input is not valid UTF-8: {exc.reason}"
        ) from exc

    records = _Parser(text, str(p)).parse_all()
    W, H, D = records[0]
    container = Container(W, H, D)
    boxes = tuple(Box(index, w, h, d) for index, (w, h, d) in enumerate(records[1:], start=1))
    return container, boxes
