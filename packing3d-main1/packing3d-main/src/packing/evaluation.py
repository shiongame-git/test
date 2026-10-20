"""Immutable inputs passed across the decoder evaluation boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Generic, TypeVar

from ._decoder_config import DecoderConfig, DEFAULT_DECODER_CONFIG
from .model import Box, Container, PackingResult


def apply_cog_penalty(
    result: PackingResult, density: float, use_center_of_gravity: bool, weight: float
) -> float:
    """Apply the shared, dimensionless CoG penalty to a density score."""
    if type(result) is not PackingResult:
        raise TypeError("result must be a PackingResult")
    if type(density) is not float:
        raise TypeError("density must be a Python float")
    if type(use_center_of_gravity) is not bool:
        raise TypeError("use_center_of_gravity must be a Python bool")
    if type(weight) is not float:
        raise TypeError("weight must be a Python float")
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("weight must be finite and >= 0.0")
    if not use_center_of_gravity:
        return density
    from .center_of_gravity import calculate_balance_distance

    distance = calculate_balance_distance(result)
    return density if distance is None else density - weight * distance


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
