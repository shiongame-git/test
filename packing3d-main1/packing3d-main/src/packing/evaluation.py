"""Immutable inputs passed across the decoder evaluation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from ._decoder_config import DecoderConfig, DEFAULT_DECODER_CONFIG
from .model import Box, Container


DecoderStateT = TypeVar("DecoderStateT")


@dataclass(frozen=True, slots=True)
class DecoderInput(Generic[DecoderStateT]):
    """Complete per-evaluation input owned by the decoder boundary.

    ``decoder_state`` is deliberately opaque here.  BL can use ``None`` while a
    future PP decoder can receive a ``PPState`` without changing the box-order
    representation or teaching GA code how PP state is interpreted.
    """

    container: Container
    ordered_boxes: tuple[Box, ...]
    decoder_state: DecoderStateT
    decoder_config: DecoderConfig = DEFAULT_DECODER_CONFIG

    def __post_init__(self) -> None:
        if type(self.container) is not Container:
            raise TypeError("container must be a Container")
        if type(self.ordered_boxes) is not tuple:
            raise TypeError("ordered_boxes must be a tuple")
        if type(self.decoder_config) is not DecoderConfig:
            raise TypeError("decoder_config must be a DecoderConfig")

        seen: set[int] = set()
        for index, box in enumerate(self.ordered_boxes):
            if type(box) is not Box:
                raise TypeError(f"ordered_boxes[{index}] must be a Box")
            if box.id in seen:
                raise ValueError(f"duplicate box id in ordered_boxes: {box.id}")
            seen.add(box.id)
