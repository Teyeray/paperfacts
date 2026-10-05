"""The crop store: a box is rendered once, described without being decoded, and named reproducibly."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from paperfacts.crops import CropStore, Region, bbox_key, png_size, sha256_hex
from paperfacts.models import NormalizedBBox
from paperfacts.pdf import render_page
from paperfacts.storage import DataLayout
from support.factories import make_blank_pdf

BOX = NormalizedBBox(x1=0.1, y1=0.1, x2=0.6, y2=0.5)
DOC = "b" * 64


@pytest.fixture
def store(tmp_path: Path) -> CropStore:
    pdf_path = make_blank_pdf(tmp_path / "paper.pdf")
    return CropStore(DataLayout(tmp_path / "data"), DOC, pdf_path, dpi=72)


def test_crop_writes_one_file_and_describes_it(store: CropStore) -> None:
    data, crop = store.crop(Region(page=0, bbox=BOX, source_ids=("mineru_p0_b1",)))

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert (crop.width_px, crop.height_px) == png_size(data)
    assert crop.image_sha256 == sha256_hex(data)
    assert crop.page == 0 and crop.dpi == 72
    assert crop.source_ids == ("mineru_p0_b1",)
    assert store.path_for(0, BOX).read_bytes() == data


def test_png_size_agrees_with_pillow(store: CropStore) -> None:
    """The header is read rather than the image decoded, so the shortcut must agree with a real decode."""
    import io

    data, crop = store.crop(Region(page=0, bbox=BOX))
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (crop.width_px, crop.height_px)
        assert image.size == png_size(data)


def test_same_box_renders_once(store: CropStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two values cited from one table ask for the same box; the second must not reach PDFium."""
    store.crop(Region(page=0, bbox=BOX, source_ids=("a",)))

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("rendered twice")

    monkeypatch.setattr("paperfacts.crops.render_region", fail)
    monkeypatch.setattr("paperfacts.crops.render_page", fail)
    again, _ = store.crop(Region(page=0, bbox=BOX, source_ids=("b",)))
    assert again[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_later_run_reads_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh store with an empty memory answers from what the first run wrote."""
    pdf_path = make_blank_pdf(tmp_path / "paper.pdf")
    layout = DataLayout(tmp_path / "data")
    first, _ = CropStore(layout, DOC, pdf_path, dpi=72).crop(Region(page=0, bbox=BOX))

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("re-rendered")

    monkeypatch.setattr("paperfacts.crops.render_region", fail)
    second, crop = CropStore(layout, DOC, pdf_path, dpi=72).crop(Region(page=0, bbox=BOX))
    assert second == first
    assert crop.image_sha256 == sha256_hex(first)


def test_dpi_and_page_are_separate_files(store: CropStore) -> None:
    paths = {
        store.path_for(0, BOX),
        store.path_for(1, BOX),
        CropStore(store.layout, DOC, store.pdf_path, dpi=144).path_for(0, BOX),
    }
    assert len(paths) == 3


def test_page_cache_renders_a_page_once_for_many_boxes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mode figure panels want: several boxes on one page, one page render between them."""
    pdf_path = make_blank_pdf(tmp_path / "paper.pdf")
    store = CropStore(DataLayout(tmp_path / "data"), DOC, pdf_path, dpi=72, page_cache=True)
    renders = 0
    real = render_page

    def counting(*args: object, **kwargs: object) -> Image.Image:
        nonlocal renders
        renders += 1
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("paperfacts.crops.render_page", counting)
    for x in (0.1, 0.3, 0.5):
        store.crop(Region(page=0, bbox=NormalizedBBox(x1=x, y1=0.1, x2=x + 0.2, y2=0.4)))
    assert renders == 1


def test_max_pixels_bounds_the_image(tmp_path: Path) -> None:
    pdf_path = make_blank_pdf(tmp_path / "paper.pdf")
    layout = DataLayout(tmp_path / "data")
    whole = NormalizedBBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    _, big = CropStore(layout, DOC, pdf_path, dpi=200).crop(Region(page=0, bbox=whole))
    _, bounded = CropStore(layout, "c" * 64, pdf_path, dpi=200, max_pixels=10_000).crop(Region(page=0, bbox=whole))

    assert big.width_px * big.height_px > 10_000
    assert bounded.width_px * bounded.height_px <= 10_000


@pytest.mark.parametrize("page_cache", [False, True])
@pytest.mark.parametrize("limits", [(None, 10_000), (10_000, None), (10_000, 20_000), (20_000, 10_000)])
def test_same_document_and_box_keep_each_pixel_limit(
    tmp_path: Path, limits: tuple[int | None, int | None], page_cache: bool
) -> None:
    """A later stage's smaller or larger request must not inherit an earlier crop's resolution."""
    pdf_path = make_blank_pdf(tmp_path / "paper.pdf")
    layout = DataLayout(tmp_path / "data")
    region = Region(page=0, bbox=BOX)
    crops = []
    for limit in limits:
        store = CropStore(layout, DOC, pdf_path, dpi=72, max_pixels=limit, page_cache=page_cache)
        data, crop = store.crop(region)
        area = crop.width_px * crop.height_px
        if limit is None:
            assert area > 20_000
        else:
            assert 0.9 * limit < area <= limit
        crops.append((data, crop))

    assert crops[0][1].path != crops[1][1].path
    assert crops[0][1].image_sha256 != crops[1][1].image_sha256
    for data, crop in crops:
        assert (layout.crops_dir(DOC) / crop.path).read_bytes() == data


@pytest.mark.parametrize("max_pixels", [None, 10_000])
def test_legacy_crop_with_unknown_pixel_limit_is_preserved_but_not_reused(
    tmp_path: Path, max_pixels: int | None
) -> None:
    import io

    pdf_path = make_blank_pdf(tmp_path / "paper.pdf")
    layout = DataLayout(tmp_path / "data")
    legacy_path = layout.crops_dir(DOC) / "page_000.0.1000-0.1000-0.6000-0.5000.72dpi.png"
    legacy_path.parent.mkdir(parents=True)
    output = io.BytesIO()
    Image.new("RGB", (1, 1)).save(output, format="PNG")
    legacy_data = output.getvalue()
    legacy_path.write_bytes(legacy_data)

    data, crop = CropStore(layout, DOC, pdf_path, dpi=72, max_pixels=max_pixels).crop(Region(page=0, bbox=BOX))

    assert crop.width_px > 1 and crop.height_px > 1
    assert crop.path != legacy_path.name
    assert data != legacy_data
    assert legacy_path.read_bytes() == legacy_data


def test_pdf_byte_identity_is_separate_from_the_logical_document_identity(store: CropStore) -> None:
    # A merged SI upload retains its logical ID even if a later merge writes different PDF bytes.
    first = CropStore(store.layout, DOC, store.pdf_path, dpi=72, source_pdf_sha256="a" * 64)
    second = CropStore(store.layout, DOC, store.pdf_path, dpi=72, source_pdf_sha256="c" * 64)

    assert first.document_id == second.document_id == DOC
    assert len({store.path_for(0, BOX), first.path_for(0, BOX), second.path_for(0, BOX)}) == 3


@pytest.mark.parametrize("digest", ["", "../outside", "z" * 64, "a" * 63])
def test_source_pdf_digest_must_be_safe_before_forming_a_path(store: CropStore, digest: str) -> None:
    bound = CropStore(store.layout, DOC, store.pdf_path, dpi=72, source_pdf_sha256=digest)
    with pytest.raises(ValueError, match="source_pdf_sha256"):
        bound.path_for(0, BOX)


@pytest.mark.parametrize(
    "box",
    [
        NormalizedBBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
        NormalizedBBox(x1=0.999, y1=0.999, x2=1.0, y2=1.0),  # at the far edge
        NormalizedBBox(x1=0.5, y1=0.5, x2=0.50001, y2=0.50001),  # thinner than a pixel
    ],
)
def test_degenerate_boxes_still_yield_an_image(store: CropStore, box: NormalizedBBox) -> None:
    _, crop = store.crop(Region(page=0, bbox=box))
    assert crop.width_px >= 1 and crop.height_px >= 1


def test_bbox_key_is_stable_and_filename_safe() -> None:
    key = bbox_key(BOX)
    assert key == bbox_key(NormalizedBBox(x1=0.1, y1=0.1, x2=0.6, y2=0.5))
    assert key == "0.1000-0.1000-0.6000-0.5000"
    assert "/" not in key and " " not in key


def test_context_ids_are_kept_apart_from_cited(store: CropStore) -> None:
    """A reader of a verdict must be able to tell the cited blocks from the window's neighbours."""
    _, crop = store.crop(Region(page=0, bbox=BOX, source_ids=("cited",), context_ids=("before", "after")))
    assert crop.source_ids == ("cited",)
    assert crop.context_ids == ("before", "after")


def test_png_size_rejects_other_bytes() -> None:
    with pytest.raises(ValueError, match="not a PNG"):
        png_size(b"\xff\xd8\xff\xe0 jpeg-ish")


def test_crop_is_frozen(store: CropStore) -> None:
    _, crop = store.crop(Region(page=0, bbox=BOX))
    with pytest.raises(ValidationError):
        crop.page = 3  # type: ignore[misc]
