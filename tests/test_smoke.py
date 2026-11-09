"""Smoke tests: the package imports and OpenCV behaves as documented."""

import cv2
import numpy as np
import pytest

import computer_vision


def test_package_imports_with_docstring() -> None:
    assert computer_vision.__doc__


@pytest.mark.parametrize(
    ("bgr", "expected"),
    [
        ((0, 0, 255), round(0.299 * 255)),
        ((0, 255, 0), round(0.587 * 255)),
        ((255, 0, 0), round(0.114 * 255)),
    ],
)
def test_bgr_to_gray_follows_bt601(bgr: tuple[int, int, int], expected: int) -> None:
    image = np.full((4, 4, 3), bgr, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    assert gray.shape == (4, 4)
    assert gray.dtype == np.uint8
    np.testing.assert_allclose(gray, expected, atol=1)
