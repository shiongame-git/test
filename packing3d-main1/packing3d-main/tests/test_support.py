"""Reference support checks and independent support-union oracle tests."""

import pytest

import packing._support as decoder_support
from packing import Box, Container, PackingResult, Placement, Point, validate_result
from packing._decoder_config import DecoderConfig
from packing._support import (
    _rectangle_union_area,
    has_sufficient_support,
    support_union_area,
)
from packing import validate_support
from packing.validate import _validation_rectangle_union_area
from oracle_support import rectangle_union_area_oracle, support_area_oracle, support_requirement_oracle


def _placement(box_id, size, point):
    return Placement(Box(box_id, *size), point)


def _assert_matches_oracle(box, point, placements, percent):
    area = support_union_area(box, point, placements)
    settings = DecoderConfig(min_support_percent=percent)
    assert area == support_area_oracle(box, point, placements)
    assert has_sufficient_support(box, point, placements, settings) == support_requirement_oracle(
        area, box, percent
    )


def test_floor_supports_the_whole_bottom_at_one_hundred_percent():
    box = Box(1, 8, 1, 5)
    _assert_matches_oracle(box, Point(3, 0, 4), (), 100)
    assert support_union_area(box, Point(3, 0, 4), ()) == 40


def test_exact_fifty_five_percent_and_fifty_four_percent_boundary():
    box = Box(2, 100, 1, 1)
    support_55 = (_placement(1, (55, 1, 1), Point(0, 0, 0)),)
    support_54 = (_placement(1, (54, 1, 1), Point(0, 0, 0)),)
    _assert_matches_oracle(box, Point(0, 1, 0), support_55, 55)
    _assert_matches_oracle(box, Point(0, 1, 0), support_54, 55)
    assert has_sufficient_support(box, Point(0, 1, 0), support_55, DecoderConfig())
    assert not has_sufficient_support(box, Point(0, 1, 0), support_54, DecoderConfig())


def test_zero_percent_allows_no_support_and_one_hundred_requires_full_coverage():
    box = Box(2, 10, 1, 1)
    point = Point(0, 1, 0)
    assert has_sufficient_support(box, point, (), DecoderConfig(min_support_percent=0))
    assert not has_sufficient_support(box, point, (), DecoderConfig(min_support_percent=100))
    full = (_placement(1, (10, 1, 1), Point(0, 0, 0)),)
    assert has_sufficient_support(box, point, full, DecoderConfig(min_support_percent=100))


def test_two_disjoint_thirty_percent_faces_contribute_sixty_percent_union():
    box = Box(3, 10, 1, 1)
    supports = (
        _placement(1, (3, 1, 1), Point(0, 0, 0)),
        _placement(2, (3, 1, 1), Point(3, 0, 0)),
    )
    _assert_matches_oracle(box, Point(0, 1, 0), supports, 55)
    assert support_union_area(box, Point(0, 1, 0), supports) == 6
    assert has_sufficient_support(box, Point(0, 1, 0), supports, DecoderConfig())


def test_overlapping_rectangles_are_counted_once_by_the_area_layer():
    rectangles = ((0, 6, 0, 10), (4, 10, 0, 10), (2, 8, 3, 7))

    assert rectangle_union_area_oracle(rectangles) == 100
    assert _rectangle_union_area(rectangles) == 100


def test_validator_union_area_matches_coordinate_cell_oracle_independently():
    rectangles = (
        (0, 6, 0, 10),
        (4, 10, 0, 10),
        (2, 8, 3, 7),
        (12, 14, 0, 2),
        (20, 20, 0, 4),
    )

    assert _validation_rectangle_union_area(rectangles) == rectangle_union_area_oracle(
        rectangles
    )


def test_support_faces_are_clipped_to_the_candidate_bottom():
    box = Box(2, 5, 1, 5)
    supports = (_placement(1, (5, 1, 5), Point(0, 0, 0)),)

    _assert_matches_oracle(box, Point(2, 1, 2), supports, 100)
    assert support_union_area(box, Point(2, 1, 2), supports) == 9
    assert not has_sufficient_support(box, Point(2, 1, 2), supports, DecoderConfig(min_support_percent=40))


def test_only_successful_faces_with_matching_top_y_support():
    box = Box(3, 4, 1, 1)
    point = Point(0, 2, 0)
    supports = (
        _placement(1, (4, 1, 1), Point(0, 0, 0)),  # top_y=1, not 2
        _placement(2, (4, 1, 1), None),
    )

    _assert_matches_oracle(box, point, supports, 1)
    assert support_union_area(box, point, supports) == 0
    assert not has_sufficient_support(box, point, supports, DecoderConfig(min_support_percent=1))


def test_edge_contact_has_zero_support_area():
    box = Box(2, 1, 1, 1)
    supports = (_placement(1, (1, 1, 1), Point(0, 0, 0)),)

    _assert_matches_oracle(box, Point(1, 1, 0), supports, 1)
    assert support_union_area(box, Point(1, 1, 0), supports) == 0


def test_support_need_not_be_directly_below_the_candidate_origin():
    box = Box(2, 10, 1, 1)
    supports = (_placement(1, (6, 1, 1), Point(2, 0, 0)),)

    _assert_matches_oracle(box, Point(0, 1, 0), supports, 55)
    assert support_union_area(box, Point(0, 1, 0), supports) == 6
    assert has_sufficient_support(box, Point(0, 1, 0), supports, DecoderConfig())


def test_support_comparison_handles_gigantic_builtin_integers():
    width = 10**400
    box = Box(2, width, 1, 1)
    supports = (_placement(1, (width // 2, 1, 1), Point(0, 0, 0)),)

    _assert_matches_oracle(box, Point(0, 1, 0), supports, 50)
    assert has_sufficient_support(box, Point(0, 1, 0), supports, DecoderConfig(min_support_percent=50))
    assert not has_sufficient_support(box, Point(0, 1, 0), supports, DecoderConfig(min_support_percent=51))


def test_final_support_inspection_runs_geometry_validation_first():
    result = PackingResult(
        Container(4, 4, 4),
        (_placement(1, (1, 1, 1), Point(1, 1, 1)),),
    )

    assert validate_result(result) == ()
    errors = validate_support(result, settings=DecoderConfig(min_support_percent=1))
    assert errors == ("placement[0] does not meet the configured support requirement",)
    assert validate_support(result, settings=DecoderConfig(min_support_percent=0)) == ()


def test_final_support_inspection_reports_geometry_errors_before_support_errors():
    result = PackingResult(
        Container(2, 2, 2),
        (_placement(1, (1, 1, 1), Point(2, 1, 1)),),
    )

    errors = validate_support(result, settings=DecoderConfig(min_support_percent=100))
    assert any("outside container" in error for error in errors)
    assert not any("support requirement" in error for error in errors)


def test_support_inspector_requires_explicit_validated_settings():
    result = PackingResult(Container(1, 1, 1), ())

    with pytest.raises(TypeError, match="settings must be a DecoderConfig"):
        validate_support(result, settings=None)


def _partially_supported_result():
    return PackingResult(
        Container(5, 3, 5),
        (
            _placement(1, (5, 1, 5), Point(0, 0, 0)),
            _placement(2, (4, 1, 5), Point(0, 1, 0)),
            _placement(3, (3, 1, 5), Point(2, 2, 0)),
        ),
    )


def test_validator_support_search_clips_faces_and_compares_integer_threshold():
    result = _partially_supported_result()

    assert validate_result(result) == ()
    assert validate_support(result, settings=DecoderConfig(min_support_percent=66)) == ()
    assert validate_support(result, settings=DecoderConfig(min_support_percent=67)) == (
        "placement[2] does not meet the configured support requirement",
    )


def test_validator_detects_under_support_when_decoder_union_overstates_area(monkeypatch):
    result = _partially_supported_result()
    target = result.placements[2]
    assert target.position is not None
    settings = DecoderConfig(min_support_percent=67)

    monkeypatch.setattr(
        decoder_support,
        "support_union_area",
        lambda box, *_args, **_kwargs: box.w * box.d,
    )
    assert decoder_support.has_sufficient_support(
        target.box, target.position, result.placements, settings
    )

    assert validate_support(result, settings=settings) == (
        "placement[2] does not meet the configured support requirement",
    )
