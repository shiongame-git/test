"""Exercise the shared input boundary through both public decoders."""

import pytest

from packing import Box, Container, DecoderInput, PPState, pack, pack_pp
from packing._decoder_config import DecoderConfig


@pytest.fixture(params=["BL", "PP"])
def decode(request):
    def call(boxes):
        container = Container(2, 1, 1)
        if request.param == "BL":
            return pack(container, boxes)
        return pack_pp(container, boxes, PPState({7: None, 2: None}))
    return call


def test_single_pass_iterable_preserves_order_and_failed_boxes(decode):
    boxes = (Box(7, 3, 1, 1), Box(2, 1, 1, 1))
    consumed = []

    def source():
        for box in boxes:
            consumed.append(box.id)
            yield box

    result = decode(source())
    assert consumed == [7, 2]
    assert tuple(p.box for p in result.placements) == boxes
    assert result.placements[0].position is None
    assert result.placements[1].position is not None


def test_iterator_failure_is_not_masked(decode):
    error = TypeError("upstream iterator failed")

    def source():
        yield Box(7, 1, 1, 1)
        raise error

    with pytest.raises(TypeError) as caught:
        decode(source())
    assert caught.value is error


def test_duplicate_id_is_rejected(decode):
    with pytest.raises(ValueError, match="duplicate box id: 7"):
        decode((Box(7, 1, 1, 1), Box(7, 2, 1, 1)))


def test_non_box_is_rejected(decode):
    with pytest.raises(TypeError, match=r"ordered_boxes\[0\] must be a Box"):
        decode((object(),))


@pytest.mark.parametrize("field,value,error", [
    ("id", True, TypeError), ("id", -1, ValueError),
    ("w", 1.0, TypeError), ("w", 0, ValueError),
    ("h", False, TypeError), ("h", 0, ValueError),
    ("d", "1", TypeError), ("d", -1, ValueError),
])
def test_decoder_rechecks_box_fields(decode, field, value, error):
    box = Box(7, 1, 1, 1)
    # Model construction normally prevents this; decoder boundaries also guard
    # against objects reconstructed or modified outside the normal constructor.
    object.__setattr__(box, field, value)
    with pytest.raises(error, match=rf"ordered_boxes\[0\].{field}"):
        decode((box,))


def test_decoder_input_carries_immutable_effective_settings():
    container = Container(2, 1, 1)
    boxes = (Box(7, 1, 1, 1),)
    default = DecoderInput(container, boxes, None)
    explicit = DecoderConfig(min_support_percent=55)
    configured = DecoderInput(container, boxes, None, explicit)
    legacy = DecoderInput(container, boxes, None, DecoderConfig(min_support_percent=0))

    assert default.decoder_config == DecoderConfig()
    assert default.decoder_config.min_support_percent == 55
    assert configured.decoder_config is explicit
    assert legacy.decoder_config.min_support_percent == 0
    with pytest.raises(TypeError, match="decoder_config"):
        DecoderInput(container, boxes, None, None)  # type: ignore[arg-type]
