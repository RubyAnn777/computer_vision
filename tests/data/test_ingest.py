from pathlib import Path

import numpy as np
import pymupdf
import pytest
from hypothesis import given
from hypothesis import strategies as st

from computer_vision.data.ingest import render_pdf_page, standardise, target_size

sizes = st.integers(min_value=1, max_value=5000)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1807, 3189, (1360, 2400)),  # Aldi Süd page
        (1458, 2400, (1458, 2400)),  # Kaufland page, already the target size
        (3000, 2000, (2400, 1600)),  # landscape
        (1000, 1000, (2400, 2400)),  # square, enlarged
    ],
)
def test_target_size_examples(
    width: int, height: int, expected: tuple[int, int]
) -> None:
    assert target_size(width, height) == expected


@given(width=sizes, height=sizes)
def test_target_size_fixes_long_side_and_keeps_aspect(width: int, height: int) -> None:
    new_width, new_height = target_size(width, height, long_side=2400)
    assert max(new_width, new_height) == 2400
    short, long_ = sorted((width, height))
    new_short = min(new_width, new_height)
    assert abs(new_short - short * 2400 / long_) <= 0.5 or new_short == 1


@pytest.mark.parametrize("bad", [(0, 10, 10), (10, -1, 10), (10, 10, 0)])
def test_target_size_rejects_non_positive(bad: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        target_size(*bad)


def test_standardise_downscales_to_target_shape() -> None:
    image = np.zeros((3189, 1807, 3), dtype=np.uint8)
    result = standardise(image)
    assert result.shape == (2400, 1360, 3)
    assert result.dtype == np.uint8


def test_standardise_returns_input_when_already_at_size() -> None:
    image = np.zeros((2400, 1458, 3), dtype=np.uint8)
    assert standardise(image) is image


def test_standardise_keeps_a_constant_colour() -> None:
    image = np.full((300, 200, 3), (10, 120, 250), dtype=np.uint8)
    result = standardise(image, long_side=90)
    assert np.all(result == (10, 120, 250))


@pytest.fixture
def red_square_pdf(tmp_path: Path) -> Path:
    """An A4 page (595 x 842 pt) with a red square in its top-left quarter."""
    path = tmp_path / "flyer.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=595, height=842)
        page.draw_rect(pymupdf.Rect(0, 0, 297, 421), color=(1, 0, 0), fill=(1, 0, 0))
        document.save(path)
    return path


def test_render_pdf_page_has_target_long_side(red_square_pdf: Path) -> None:
    image = render_pdf_page(red_square_pdf, 0, long_side=600)
    assert image.shape == (600, 424, 3)
    assert image.dtype == np.uint8


def test_render_pdf_page_returns_bgr(red_square_pdf: Path) -> None:
    image = render_pdf_page(red_square_pdf, 0, long_side=600)
    np.testing.assert_array_equal(image[100, 100], (0, 0, 255))  # red in BGR
    np.testing.assert_array_equal(image[500, 400], (255, 255, 255))  # white
