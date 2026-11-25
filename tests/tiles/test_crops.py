import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from computer_vision.detection.threshold import BoxF
from computer_vision.tiles.crops import (
    Source,
    TileKey,
    crop,
    parse_tile_id,
    reading_order,
    round_half_up,
    tile_id,
    to_pixel_box,
)

PAGE = "mueller_prospekt-2024-kw46_p003"


# --- tile ids -----------------------------------------------------------------


def test_tile_id_known_value() -> None:
    assert tile_id(PAGE, "b", 7) == "mueller_prospekt-2024-kw46_p003_b007"


@pytest.mark.parametrize("source", ["b", "d"])
@pytest.mark.parametrize("index", [0, 7, 999])
def test_tile_id_round_trip(source: Source, index: int) -> None:
    value = tile_id(PAGE, source, index)
    assert parse_tile_id(value) == TileKey(PAGE, source, index)


@pytest.mark.parametrize(
    "value",
    [
        f"{PAGE}_b7",  # index not three digits
        f"{PAGE}_x007",  # unknown source letter
        PAGE,  # no tile suffix
        "not-a-page_b007",  # page part malformed
    ],
)
def test_parse_tile_id_rejects_malformed_ids(value: str) -> None:
    with pytest.raises(ValueError, match="not a valid"):
        parse_tile_id(value)


@pytest.mark.parametrize("index", [-1, 1000])
def test_tile_id_rejects_index_out_of_range(index: int) -> None:
    with pytest.raises(ValueError, match="index"):
        tile_id(PAGE, "d", index)


def test_tile_id_rejects_bad_page_id() -> None:
    with pytest.raises(ValueError, match="page id"):
        tile_id("Müller page 3", "b", 0)


# --- rounding and clipping ----------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"), [(10.5, 11), (11.5, 12), (10.49, 10), (-0.5, 0)]
)
def test_round_half_up(value: float, expected: int) -> None:
    assert round_half_up(value) == expected


def test_to_pixel_box_rounds_halves_up_unlike_python_round() -> None:
    assert round(10.5) == 10  # banker's rounding, the rule we avoid
    assert to_pixel_box((10.5, 20.4, 30.5, 40.6), 100, 100) == (11, 20, 31, 41)


def test_to_pixel_box_clips_to_the_image() -> None:
    assert to_pixel_box((-5.2, -1.0, 105.0, 50.0), 100, 80) == (0, 0, 100, 50)


@pytest.mark.parametrize(
    "box",
    [
        (10.2, 10.0, 10.4, 20.0),  # rounds to zero width
        (120.0, 10.0, 130.0, 20.0),  # entirely right of the image
    ],
)
def test_to_pixel_box_rejects_boxes_without_area(box: BoxF) -> None:
    with pytest.raises(ValueError, match="no area"):
        to_pixel_box(box, 100, 100)


@st.composite
def boxes_inside(draw: st.DrawFn) -> tuple[BoxF, int, int]:
    width = draw(st.integers(10, 500))
    height = draw(st.integers(10, 500))
    x0 = draw(st.floats(0, width - 2))
    y0 = draw(st.floats(0, height - 2))
    x1 = draw(st.floats(x0 + 1.5, width))
    y1 = draw(st.floats(y0 + 1.5, height))
    return (x0, y0, x1, y1), width, height


@given(boxes_inside())
def test_to_pixel_box_stays_inside_and_within_half_a_pixel(
    case: tuple[BoxF, int, int],
) -> None:
    box, width, height = case
    x0, y0, x1, y1 = to_pixel_box(box, width, height)
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height
    for rounded, original in zip((x0, y0, x1, y1), box, strict=True):
        assert abs(rounded - original) <= 0.5


# --- reading order ------------------------------------------------------------


def test_reading_order_top_to_bottom_then_left_to_right() -> None:
    boxes = [(50, 100, 60, 110), (10, 0, 20, 10), (0, 100, 5, 110), (40, 0, 45, 5)]
    assert reading_order(boxes) == [1, 3, 2, 0]


def test_reading_order_does_not_depend_on_input_order() -> None:
    boxes = [(50, 100, 60, 110), (10, 0, 20, 10), (0, 100, 5, 110), (40, 0, 45, 5)]
    shuffled = [boxes[i] for i in (2, 0, 3, 1)]
    assert [boxes[i] for i in reading_order(boxes)] == [
        shuffled[i] for i in reading_order(shuffled)
    ]


# --- cropping -----------------------------------------------------------------


def coordinate_image(height: int, width: int) -> np.ndarray:
    """Image whose pixel (y, x) holds (y, x, 7), so any off-by-one is visible."""
    ys, xs = np.mgrid[0:height, 0:width]
    return np.stack([ys, xs, np.full_like(ys, 7)], axis=-1).astype(np.uint8)


def test_crop_has_the_box_size_and_exact_pixels() -> None:
    image = coordinate_image(50, 80)
    tile = crop(image, (10, 20, 30, 25))

    assert tile.shape == (5, 20, 3)
    assert tile.dtype == np.uint8
    assert tile[0, 0].tolist() == [20, 10, 7]  # top-left is (y0, x0)
    assert tile[-1, -1].tolist() == [24, 29, 7]  # bottom-right is (y1 - 1, x1 - 1)


def test_crop_returns_a_copy() -> None:
    image = coordinate_image(50, 80)
    tile = crop(image, (0, 0, 10, 10))
    tile[:] = 255
    assert image[0, 0].tolist() == [0, 0, 7]


@pytest.mark.parametrize("box", [(0, 0, 0, 10), (70, 0, 81, 10), (-1, 0, 5, 5)])
def test_crop_rejects_empty_or_outside_boxes(box: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="empty or outside"):
        crop(coordinate_image(50, 80), box)
