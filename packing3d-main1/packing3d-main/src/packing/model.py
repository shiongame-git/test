"""Immutable core data model for packing results."""

from __future__ import annotations

from dataclasses import dataclass


def _require_builtin_int(name: str, value: object, *, minimum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int (bool and other numeric types are not accepted)")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


@dataclass(frozen=True, slots=True)
class Container:
    W: int
    H: int
    D: int

    def __post_init__(self) -> None:
        _require_builtin_int("W", self.W, minimum=1)
        _require_builtin_int("H", self.H, minimum=1)
        _require_builtin_int("D", self.D, minimum=1)

    @property
    def volume(self) -> int:
        return self.W * self.H * self.D


@dataclass(frozen=True, slots=True)
class Box:
    id: int
    w: int
    h: int
    d: int

    def __post_init__(self) -> None:
        _require_builtin_int("id", self.id, minimum=0)
        _require_builtin_int("w", self.w, minimum=1)
        _require_builtin_int("h", self.h, minimum=1)
        _require_builtin_int("d", self.d, minimum=1)

    @property
    def volume(self) -> int:
        return self.w * self.h * self.d


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        _require_builtin_int("x", self.x, minimum=0)
        _require_builtin_int("y", self.y, minimum=0)
        _require_builtin_int("z", self.z, minimum=0)


@dataclass(frozen=True, slots=True)
class Placement:
    box: Box
    position: Point | None

    def __post_init__(self) -> None:
        if type(self.box) is not Box:
            raise TypeError("box must be a Box")
        if self.position is not None and type(self.position) is not Point:
            raise TypeError("position must be a Point or None")


@dataclass(frozen=True, slots=True)
class PackingResult:
    container: Container
    placements: tuple[Placement, ...]

    def __post_init__(self) -> None:
        if type(self.container) is not Container:
            raise TypeError("container must be a Container")
        if type(self.placements) is not tuple:
            raise TypeError("placements must be a tuple")
        for index, placement in enumerate(self.placements):
            if type(placement) is not Placement:
                raise TypeError(f"placements[{index}] must be a Placement")

    @property
    def placed_count(self) -> int:
        return sum(p.position is not None for p in self.placements)

    @property
    def failed_count(self) -> int:
        return len(self.placements) - self.placed_count

    @property
    def placed_volume(self) -> int:
        return sum(p.box.volume for p in self.placements if p.position is not None)

    @property
    def density(self) -> float:
        return self.placed_volume / self.container.volume

    @property
    def failed_boxes(self) -> tuple[Box, ...]:
        return tuple(p.box for p in self.placements if p.position is None)
