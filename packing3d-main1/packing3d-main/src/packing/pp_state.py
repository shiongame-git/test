"""Immutable inherited-position state used by the PP decoder boundary."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType

from .model import Point


class PPState(Mapping[int, Point | None]):
    """Immutable mapping from ``box_id`` to inherited ``Point`` or ``None``.

    The value type validates only the state-local contract.  Problem-specific
    checks (exact ID set and container fit) happen when the PP decoder
    is called.  Inherited positions may still conflict with each other because
    conflict resolution is decoder behavior, not a PPState construction error.
    """

    __slots__ = ("_items", "_lookup", "_locked")

    def __init__(
        self,
        entries: Mapping[int, Point | None]
        | Iterable[tuple[int, Point | None]] = (),
    ) -> None:
        if isinstance(entries, Mapping):
            raw_items = tuple(entries.items())
        else:
            try:
                raw_items = tuple(entries)
            except TypeError:
                raise TypeError("entries must be a mapping or iterable of (box_id, position) pairs") from None

        checked: list[tuple[int, Point | None]] = []
        seen: set[int] = set()
        for index, item in enumerate(raw_items):
            if type(item) not in (tuple, list) or len(item) != 2:
                raise TypeError(f"entries[{index}] must be a (box_id, position) pair")
            box_id, position = item
            if type(box_id) is not int:
                raise TypeError(f"entries[{index}].box_id must be a Python int")
            if box_id < 0:
                raise ValueError(f"entries[{index}].box_id must be >= 0")
            if box_id in seen:
                raise ValueError(f"duplicate box id in PPState: {box_id}")
            if position is not None and type(position) is not Point:
                raise TypeError(f"entries[{index}].position must be a Point or None")
            seen.add(box_id)
            checked.append((box_id, position))

        # Mapping meaning is by ID, not by the input order.  Canonical ordering
        # also makes equality and repr stable across equivalent construction paths.
        checked.sort(key=lambda item: item[0])
        object.__setattr__(self, "_items", tuple(checked))
        object.__setattr__(self, "_lookup", MappingProxyType(dict(checked)))
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("PPState is immutable")
        object.__setattr__(self, name, value)

    def __getitem__(self, box_id: int) -> Point | None:
        return self._lookup[box_id]

    def __iter__(self) -> Iterator[int]:
        return (box_id for box_id, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        body = ", ".join(f"{box_id}: {position!r}" for box_id, position in self._items)
        return f"PPState({{{body}}})"

    def __eq__(self, other: object) -> bool:
        if type(other) is not PPState:
            return NotImplemented
        return self._items == other._items

    def items_tuple(self) -> tuple[tuple[int, Point | None], ...]:
        """Return the canonical immutable ID-to-position entries."""
        return self._items

    def __reduce__(self):
        """Reconstruct from canonical entries for spawn-process transport."""
        return (type(self), (self._items,))
