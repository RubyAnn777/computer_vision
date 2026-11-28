import numpy as np
import pytest

from computer_vision.mosaic.descriptors import (
    describe_in_batches,
    mean_bgr,
    mean_lab,
    subblock_lab,
    to_lab,
)


def solid(bgr: tuple[int, int, int], n: int = 1, h: int = 6, w: int = 4) -> np.ndarray:
    return np.full((n, h, w, 3), bgr, dtype=np.uint8)


@pytest.mark.parametrize(
    ("bgr", "lab"),
    [
        ((255, 255, 255), (100.0, 0.0, 0.0)),  # white
        ((0, 0, 0), (0.0, 0.0, 0.0)),  # black
        ((0, 0, 255), (53.24, 80.09, 67.20)),  # sRGB red, textbook value
    ],
)
def test_to_lab_gives_true_cielab_ranges(
    bgr: tuple[int, int, int], lab: tuple[float, float, float]
) -> None:
    assert to_lab(solid(bgr))[0, 0, 0] == pytest.approx(lab, abs=0.1)


def test_mean_bgr_of_two_halves() -> None:
    image = solid((0, 0, 0))
    image[0, :3] = (200, 100, 50)  # top half of 6 rows
    assert mean_bgr(image)[0] == pytest.approx([100, 50, 25])


def test_mask_restricts_the_mean() -> None:
    image = solid((0, 0, 0))
    image[0, :3] = (200, 100, 50)
    mask = np.zeros((6, 4), dtype=bool)
    mask[:3] = True
    assert mean_bgr(image, mask)[0] == pytest.approx([200, 100, 50])


def test_empty_mask_is_an_error() -> None:
    with pytest.raises(ValueError, match="no pixels"):
        mean_lab(solid((1, 2, 3)), np.zeros((6, 4), dtype=bool))


def test_mean_lab_averages_converted_pixels() -> None:
    # black and white halves: mean L* is 50, not the L* of mid-grey (53.4)
    image = solid((0, 0, 0))
    image[0, :3] = 255
    assert mean_lab(image)[0, 0] == pytest.approx(50.0, abs=0.1)


def test_subblocks_keep_the_layout() -> None:
    top_dark = solid((255, 255, 255))
    top_dark[0, :3] = 0
    bottom_dark = solid((255, 255, 255))
    bottom_dark[0, 3:] = 0
    a, b = subblock_lab(np.concatenate([top_dark, bottom_dark]))
    assert a[0] == pytest.approx(0, abs=0.1)  # top-left block of a is black
    assert b[0] == pytest.approx(100, abs=0.1)  # top-left block of b is white
    # ... while their plain means are identical
    m = mean_lab(np.concatenate([top_dark, bottom_dark]))
    assert m[0] == pytest.approx(m[1])


def test_subblocks_need_divisible_sizes() -> None:
    with pytest.raises(ValueError, match="split"):
        subblock_lab(solid((0, 0, 0), h=5), k=2)


def test_batches_give_the_same_result() -> None:
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, (10, 6, 4, 3), dtype=np.uint8)
    assert np.allclose(describe_in_batches(images, mean_lab, batch=3), mean_lab(images))
