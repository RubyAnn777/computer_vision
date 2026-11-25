import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from computer_vision.tiles.standardise import (
    TILE_ASPECT,
    border_colour,
    fit_to_aspect,
    fitted_size,
    kept_share,
    resize_tile,
)


def product_on_background(height: int, width: int) -> np.ndarray:
    """White crop with a red product block in the middle."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = (0, 0, 255)
    return image


def test_tile_aspect_is_the_regular_hexagon_box() -> None:
    assert pytest.approx(math.sqrt(3) / 2) == TILE_ASPECT


def test_border_colour_ignores_a_product_touching_the_edge() -> None:
    image = product_on_background(40, 30)
    image[0, :5] = (0, 0, 0)  # a few dark edge pixels
    assert border_colour(image).tolist() == [255, 255, 255]


def test_worked_example_from_the_docstring() -> None:
    # 100 x 300 bottle: cut to 70 % (100 x 210), then padded to 182 x 210
    assert fitted_size(100, 300, TILE_ASPECT, 0.7) == (182, 210)
    assert fit_to_aspect(product_on_background(300, 100)).shape == (210, 182, 3)


def test_pure_crop_when_min_keep_is_small() -> None:
    out = fit_to_aspect(product_on_background(300, 100), TILE_ASPECT, min_keep=0.0)
    assert out.shape[:2] == (round(100 / TILE_ASPECT), 100)


def test_pure_pad_when_min_keep_is_one() -> None:
    image = product_on_background(300, 100)
    out = fit_to_aspect(image, TILE_ASPECT, min_keep=1.0)
    height, width = out.shape[:2]
    assert height == 300
    x0 = (width - 100) // 2
    assert np.array_equal(out[:, x0 : x0 + 100], image)  # nothing cut
    assert out[:, 0].tolist() == [[255, 255, 255]] * 300  # padding is the border colour


def test_wide_crops_are_padded_top_and_bottom() -> None:
    out = fit_to_aspect(product_on_background(100, 400), TILE_ASPECT, min_keep=0.7)
    assert out.shape[1] == 280  # 70 % of 400
    assert out.shape[1] / out.shape[0] == pytest.approx(TILE_ASPECT, abs=0.01)


@given(
    st.integers(8, 600),
    st.integers(8, 600),
    st.floats(0.05, 1.0),
)
def test_fit_hits_the_aspect_and_keeps_at_least_min_keep(
    width: int, height: int, min_keep: float
) -> None:
    out_w, out_h = fitted_size(width, height, TILE_ASPECT, min_keep)
    out = fit_to_aspect(
        np.zeros((height, width, 3), dtype=np.uint8), TILE_ASPECT, min_keep
    )
    assert out.shape[:2] == (out_h, out_w)
    assert abs(out_w - out_h * TILE_ASPECT) <= 1  # within one pixel
    assert kept_share(width, height, TILE_ASPECT, min_keep) >= min(1.0, min_keep) - 1e-9


@pytest.mark.parametrize(
    ("aspect", "min_keep"), [(0.0, 0.7), (0.866, -0.1), (0.866, 1.5)]
)
def test_fit_rejects_bad_parameters(aspect: float, min_keep: float) -> None:
    with pytest.raises(ValueError, match="need aspect"):
        fit_to_aspect(product_on_background(30, 30), aspect, min_keep)


def test_kept_share_values() -> None:
    assert kept_share(100, 300, TILE_ASPECT, 0.01) == pytest.approx(
        (1 / 3) / TILE_ASPECT
    )
    assert kept_share(100, 300, TILE_ASPECT, 0.7) == pytest.approx(0.7)
    assert kept_share(866, 1000, TILE_ASPECT, 0.7) == pytest.approx(1.0, abs=1e-3)


def test_resize_tile_shape_and_constant_colour_is_preserved() -> None:
    image = np.full((210, 182, 3), (10, 200, 30), dtype=np.uint8)
    out = resize_tile(image, 78, 90)
    assert out.shape == (90, 78, 3)
    assert out.dtype == np.uint8
    assert out.reshape(-1, 3).mean(axis=0).tolist() == [10, 200, 30]
