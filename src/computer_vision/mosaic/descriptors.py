"""Summarise cells and tiles by a few colour numbers (D25).

A descriptor turns each image of shape (H, W, 3) into a short vector, so that
"the tile that looks most like this cell" becomes "the nearest vector". All
functions take a stack of images, shape (N, H, W, 3), BGR uint8, and return
an array of shape (N, d), float32.

- `mean_bgr`: the average colour, d = 3 (the course baseline, Lecture 3).
- `mean_lab`: the average colour in CIELAB, d = 3. Euclidean distance there
  (Delta E 1976) roughly matches how different two colours look.
- `subblock_lab`: the average CIELAB colour of each of k x k sub-blocks,
  d = 3 k^2. Keeps some of the layout inside a cell, so edges line up.

Why the mean: it is the single colour with the smallest squared error over a
region, and the pixel error between a cell and a tile splits exactly into a
term that depends only on the two means and a structure term (see the
explanation doc, Step 2). Mean matching minimises the first term, which is
what survives when the mosaic is seen from a distance.

An optional ``mask`` of shape (H, W) restricts the mean to the pixels where it
is True. Rectangles do not need it; the hexagons of Task 2 will.
"""

from collections.abc import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

Descriptor = Callable[[NDArray[np.uint8]], NDArray[np.float32]]


def to_lab(images: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Convert BGR uint8 images to CIELAB.

    Parameters
    ----------
    images : NDArray[np.uint8]
        Shape (..., H, W, 3), BGR.

    Returns
    -------
    NDArray[np.float32]
        Same shape; L* in 0..100, a* and b* roughly in -128..127. OpenCV
        gives these true CIELAB ranges only for float input in [0, 1]; for
        uint8 input it rescales them to 0..255, which would distort distances.
    """
    shape = images.shape
    flat = images.reshape(-1, shape[-2], 3).astype(np.float32) / 255.0
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB)
    return np.asarray(lab, dtype=np.float32).reshape(shape)


def _masked_mean(
    values: NDArray[np.float32], mask: NDArray[np.bool_] | None
) -> NDArray[np.float32]:
    """Average (N, H, W, C) over H and W, only where ``mask`` is True."""
    if mask is None:
        return values.mean(axis=(1, 2), dtype=np.float32)
    if not mask.any():
        raise ValueError("mask selects no pixels")
    return values[:, mask].mean(axis=1, dtype=np.float32)


def mean_bgr(
    images: NDArray[np.uint8], mask: NDArray[np.bool_] | None = None
) -> NDArray[np.float32]:
    """Return the average blue, green and red of each image, shape (N, 3)."""
    return _masked_mean(images.astype(np.float32), mask)


def mean_lab(
    images: NDArray[np.uint8], mask: NDArray[np.bool_] | None = None
) -> NDArray[np.float32]:
    """Return the average L*, a*, b* of each image, shape (N, 3).

    Every pixel is converted first and then averaged, so the result is the
    mean of the colours as CIELAB sees them, not the CIELAB value of the mean
    BGR colour (the conversion is not linear, so the two differ).
    """
    return _masked_mean(to_lab(images), mask)


def subblock_lab(images: NDArray[np.uint8], k: int = 2) -> NDArray[np.float32]:
    """Return the average CIELAB colour of each of k x k sub-blocks.

    Parameters
    ----------
    images : NDArray[np.uint8]
        Shape (N, H, W, 3) with H and W divisible by ``k``.
    k : int, optional
        Sub-blocks per side; 2 gives four quarters.

    Returns
    -------
    NDArray[np.float32]
        Shape (N, 3 k^2), the blocks in reading order, each as (L*, a*, b*).

    Raises
    ------
    ValueError
        If H or W is not divisible by ``k``.
    """
    n, height, width, _ = images.shape
    if height % k or width % k:
        raise ValueError(f"{width} x {height} images do not split into {k} x {k}")
    lab = to_lab(images).reshape(n, k, height // k, k, width // k, 3)
    return lab.mean(axis=(2, 4), dtype=np.float32).reshape(n, 3 * k * k)


DESCRIPTORS: dict[str, Descriptor] = {
    "mean_bgr": mean_bgr,
    "mean_lab": mean_lab,
    "subblock_lab": subblock_lab,
}


def describe_in_batches(
    images: NDArray[np.uint8], descriptor: Descriptor, batch: int = 256
) -> NDArray[np.float32]:
    """Apply a descriptor to many images, ``batch`` at a time.

    Converting 4,000 cells of 78 x 90 pixels to float CIELAB at once needs
    about 340 MB; in batches of 256 it needs about 22 MB, with the same result.
    """
    parts = [descriptor(images[i : i + batch]) for i in range(0, len(images), batch)]
    return np.concatenate(parts, axis=0)
