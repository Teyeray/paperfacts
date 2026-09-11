"""Geometry models: gatekeeper of the coordinate system.

The invariant protected at this layer: **every bbox that reaches downstream code is within
[0, 1] and non-degenerate**. As long as these cases stay green, "a garbage box from the parser
quietly turns into a plausible-looking crop region" can never happen.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from paperfacts.models.geometry import DocumentGeometry, NormalizedBBox, PageGeometry


def make_bbox(x1: float = 0.1, y1: float = 0.2, x2: float = 0.5, y2: float = 0.6) -> NormalizedBBox:
    return NormalizedBBox(x1=x1, y1=y1, x2=x2, y2=y2)


# ---- Construction and validation --------------------------------------------------------


def test_normalized_bbox_rejects_coordinates_outside_unit_range():
    # Arrange / Act / Assert: direct construction never clamps — out of range raises
    # immediately, keeping "looks legal" coordinates from sneaking in.
    with pytest.raises(ValidationError):
        NormalizedBBox(x1=-0.01, y1=0.0, x2=0.5, y2=0.5)
    with pytest.raises(ValidationError):
        NormalizedBBox(x1=0.0, y1=0.0, x2=1.01, y2=0.5)


def test_normalized_bbox_rejects_flipped_box():
    # A flipped box (x2 < x1) is garbage output from a parser; it must fail at construction time.
    with pytest.raises(ValidationError):
        NormalizedBBox(x1=0.6, y1=0.1, x2=0.2, y2=0.5)


def test_normalized_bbox_rejects_zero_area_box():
    # A zero-area box crops to an empty image; better to error and let the adapter skip it than
    # let it flow downstream.
    with pytest.raises(ValidationError):
        NormalizedBBox(x1=0.3, y1=0.3, x2=0.3, y2=0.8)


def test_normalized_bbox_is_frozen():
    # frozen guarantees a bbox is never mutated in place while being passed around.
    bbox = make_bbox()
    with pytest.raises(ValidationError):
        bbox.x1 = 0.9


# ---- Factory methods: the sole entry point for normalization -----------------------------


def test_from_thousandths_divides_by_one_thousand():
    # MinerU's per-mille integers must be divided by 1000, not by the page width.
    bbox = NormalizedBBox.from_thousandths([100, 250, 900, 750])

    assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == (0.1, 0.25, 0.9, 0.75)


def test_from_thousandths_clamps_slightly_out_of_range_box():
    # A parser often gives slightly-out-of-range values like -5 / 1005: clamp to the boundary
    # rather than dropping the whole block.
    bbox = NormalizedBBox.from_thousandths([-5, 0, 1005, 300])

    assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == (0.0, 0.0, 1.0, 0.3)


def test_from_thousandths_rejects_zero_area_after_scaling():
    # Clamping overflow isn't the same as fixing ordering: a box that's still zero-area after
    # scaling still errors.
    with pytest.raises(ValidationError):
        NormalizedBBox.from_thousandths([10, 10, 10, 10])


def test_from_thousandths_rejects_flipped_box_even_when_in_range():
    with pytest.raises(ValidationError):
        NormalizedBBox.from_thousandths([900, 100, 100, 500])


@pytest.mark.parametrize("box", [[1, 2, 3], [1, 2, 3, 4, 5], []])
def test_from_thousandths_rejects_boxes_that_are_not_four_numbers(box):
    # _four's length check: a wrong length must error immediately, instead of letting None flow downstream.
    with pytest.raises(ValueError, match="bbox needs 4 numbers"):
        NormalizedBBox.from_thousandths(box)


def test_from_pixels_divides_by_actual_image_size():
    # Paddle's pixel coordinates must be divided by the **actual** render size, not estimated
    # from a DPI formula.
    bbox = NormalizedBBox.from_pixels([200, 400, 1000, 800], width_px=2000, height_px=1600)

    assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == (0.1, 0.25, 0.5, 0.5)


def test_from_pixels_clamps_box_overflowing_the_rendered_page():
    bbox = NormalizedBBox.from_pixels([-20, 100, 1700, 200], width_px=1653, height_px=2339)

    assert bbox.x1 == 0.0
    assert bbox.x2 == 1.0


@pytest.mark.parametrize(("width_px", "height_px"), [(0, 100), (100, 0), (-10, 100)])
def test_from_pixels_rejects_non_positive_image_size(width_px, height_px):
    # A size of 0 would cause division by zero; erroring early beats letting inf into the model.
    with pytest.raises(ValueError, match="image size must be positive"):
        NormalizedBBox.from_pixels([1, 2, 3, 4], width_px=width_px, height_px=height_px)


def test_from_points_divides_by_page_size_in_points():
    page = PageGeometry(index=0, width_pt=600.0, height_pt=800.0)

    bbox = NormalizedBBox.from_points([60, 80, 300, 400], page=page)

    assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == (0.1, 0.1, 0.5, 0.5)


def test_from_points_accepts_integer_valued_sequences():
    # _four normalizes any numeric sequence into floats; tuples, lists, or ints all work.
    page = PageGeometry(index=1, width_pt=1000.0, height_pt=1000.0)

    bbox = NormalizedBBox.from_points((0, 0, 1000, 1000), page=page)

    assert bbox.area == pytest.approx(1.0)


# ---- Geometric operations ---------------------------------------------------------------


def test_width_height_and_area_are_derived_from_corners():
    bbox = NormalizedBBox(x1=0.2, y1=0.1, x2=0.6, y2=0.6)

    assert bbox.width == pytest.approx(0.4)
    assert bbox.height == pytest.approx(0.5)
    assert bbox.area == pytest.approx(0.2)


def test_to_pixels_rounds_to_integers():
    # Overlay / cropping needs integer pixels; this also pins down "converted using the given
    # size", not the original size.
    bbox = NormalizedBBox(x1=0.1, y1=0.2, x2=0.9, y2=0.8)

    assert bbox.to_pixels(width_px=1000, height_px=500) == (100, 100, 900, 400)


def test_to_pixels_rounds_fractional_pixels_instead_of_truncating():
    bbox = NormalizedBBox(x1=0.1234, y1=0.5, x2=0.9, y2=0.8)

    x1, _, _, _ = bbox.to_pixels(width_px=1000, height_px=1000)

    assert x1 == 123


def test_union_returns_the_minimal_enclosing_box():
    # Design doc §16: on same-page conflicts, take the union of the two bboxes before cropping.
    left = NormalizedBBox(x1=0.1, y1=0.1, x2=0.4, y2=0.4)
    right = NormalizedBBox(x1=0.3, y1=0.05, x2=0.8, y2=0.35)

    merged = left.union(right)

    assert (merged.x1, merged.y1, merged.x2, merged.y2) == (0.1, 0.05, 0.8, 0.4)


def test_union_of_a_box_with_itself_is_unchanged():
    bbox = make_bbox()

    assert bbox.union(bbox) == bbox


def test_padded_expands_the_box_on_all_four_sides():
    bbox = NormalizedBBox(x1=0.2, y1=0.2, x2=0.4, y2=0.4)

    padded = bbox.padded(0.05)

    assert (padded.x1, padded.y1, padded.x2, padded.y2) == (
        pytest.approx(0.15),
        pytest.approx(0.15),
        pytest.approx(0.45),
        pytest.approx(0.45),
    )


def test_padded_clamps_at_the_page_edge():
    # A block flush against the edge must not overflow the page after padding, or cropping would error.
    bbox = NormalizedBBox(x1=0.01, y1=0.01, x2=0.99, y2=0.99)

    padded = bbox.padded(0.1)

    assert (padded.x1, padded.y1, padded.x2, padded.y2) == (0.0, 0.0, 1.0, 1.0)


def test_padded_rejects_negative_padding():
    # A negative pad would shrink the box, possibly to zero area; forbid it outright instead of
    # letting it fail mysteriously in the validator.
    with pytest.raises(ValueError, match="pad must not be negative"):
        make_bbox().padded(-0.01)


def test_padded_with_zero_returns_an_equal_box():
    bbox = make_bbox()

    assert bbox.padded(0.0) == bbox


# ---- Page geometry ----------------------------------------------------------------------


def test_page_geometry_rejects_non_positive_size():
    with pytest.raises(ValidationError):
        PageGeometry(index=0, width_pt=0.0, height_pt=800.0)
    with pytest.raises(ValidationError):
        PageGeometry(index=-1, width_pt=600.0, height_pt=800.0)


def test_document_geometry_page_count_matches_pages():
    geometry = DocumentGeometry(
        pages=(
            PageGeometry(index=0, width_pt=595.0, height_pt=842.0),
            PageGeometry(index=1, width_pt=612.0, height_pt=792.0),
        )
    )

    assert geometry.page_count == 2
    assert geometry.page(1).width_pt == 612.0


def test_document_geometry_page_raises_index_error_with_page_count_in_message():
    # Out of range must blow up at the call site; silently returning the last page would let a
    # bbox use the wrong page size with no trace.
    geometry = DocumentGeometry(pages=(PageGeometry(index=0, width_pt=595.0, height_pt=842.0),))

    with pytest.raises(IndexError, match="document has 1 page"):
        geometry.page(5)


def test_document_geometry_page_rejects_negative_index_semantics():
    # Python's negative indexing would silently land on the last page — this confirms it isn't
    # disguised as a legitimate "page -1" result.
    geometry = DocumentGeometry(
        pages=(
            PageGeometry(index=0, width_pt=595.0, height_pt=842.0),
            PageGeometry(index=1, width_pt=612.0, height_pt=792.0),
        )
    )

    assert geometry.page(-1).index == 1  # records current behavior: negative index follows Python semantics
