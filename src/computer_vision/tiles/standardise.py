"""Bring every crop to the common tile shape and size (D22, D23).

All tiles share one width/height ratio, ``TILE_ASPECT`` = √3/2, the bounding
box of a regular pointy-top hexagon. A crop rarely has that ratio, so
`fit_to_aspect` first crops it towards the target (centre crop, never cutting
away more than ``1 - min_keep`` of the area) and pads whatever difference is
left with the crop's own border colour. Catalog packshots sit on a plain
background, so the padding continues that background instead of adding a
foreign colour that would drag the tile's mean colour away. `resize_tile`
then scales the tile to the master size with area interpolation, which
averages source pixels and avoids aliasing (Lecture 2).

Worked example: a bottle crop of 100 x 300 px (W/H 0.33) needs W/H 0.866.
Cropping alone would cut it to 100 x 115 and keep only 0.33 / 0.866 = 38 % of
it. With ``min_keep=0.7`` it is cut only down to 70 % of its area,
100 x 210 (W/H 0.48), and then padded left and right to 182 x 210
(W/H 0.866).
"""

import math

import cv2
import numpy as np
from numpy.typing import NDArray

TILE_ASPECT = math.sqrt(3) / 2  # width / height of a regular pointy-top hexagon box


def border_colour(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return the per-channel median of the image's outermost pixels.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Crop of shape (H, W, 3).

    Returns
    -------
    NDArray[np.uint8]
        Shape (3,). The median, not the mean, so a product touching the edge
        in a few places does not change the background estimate.
    """
    border = np.concatenate(
        [image[0], image[-1], image[1:-1, 0], image[1:-1, -1]], axis=0
    )
    return np.median(border, axis=0).astype(np.uint8)


def fit_to_aspect(
    image: NDArray[np.uint8], aspect: float = TILE_ASPECT, min_keep: float = 0.7
) -> NDArray[np.uint8]:
    """Crop towards, then pad to, a target width/height ratio.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Crop of shape (H, W, 3).
    aspect : float, optional
        Target width / height.
    min_keep : float, optional
        Smallest share of the crop's area that the centre crop may keep, in
        [0, 1]. ``1.0`` means pad only, ``0.0`` means a pure centre crop.

    Returns
    -------
    NDArray[np.uint8]
        Image whose width / height equals ``aspect`` to within one pixel.

    Raises
    ------
    ValueError
        If ``aspect`` is not positive or ``min_keep`` is outside [0, 1].
    """
    if aspect <= 0 or not 0 <= min_keep <= 1:
        raise ValueError(
            f"need aspect > 0 and 0 <= min_keep <= 1, got {aspect}, {min_keep}"
        )
    height, width = image.shape[:2]
    out_w, out_h = fitted_size(width, height, aspect, min_keep)
    if width / height > aspect:  # too wide: cut the sides, pad top and bottom
        left = (width - out_w) // 2
        cropped = image[:, left : left + out_w]
    else:  # too tall: cut top and bottom, pad the sides
        top = (height - out_h) // 2
        cropped = image[top : top + out_h]
    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    canvas[:] = border_colour(image)
    ch, cw = cropped.shape[:2]
    y0, x0 = (out_h - ch) // 2, (out_w - cw) // 2
    canvas[y0 : y0 + ch, x0 : x0 + cw] = cropped
    return canvas


def fitted_size(
    width: int, height: int, aspect: float = TILE_ASPECT, min_keep: float = 0.7
) -> tuple[int, int]:
    """Return the ``(width, height)`` that `fit_to_aspect` produces for a crop.

    Lets the size filter run on the index alone, without loading images.
    """
    if width / height > aspect:
        new_width = max(round(height * aspect), math.ceil(width * min_keep))
        return new_width, max(height, round(new_width / aspect))
    new_height = max(round(width / aspect), math.ceil(height * min_keep))
    return max(width, round(new_height * aspect)), new_height


def kept_share(
    width: int, height: int, aspect: float = TILE_ASPECT, min_keep: float = 0.7
) -> float:
    """Return the share of a crop's area that `fit_to_aspect` keeps.

    Used for the report: how much of each product survives standardisation.
    """
    ratio = width / height
    pure_crop = aspect / ratio if ratio > aspect else ratio / aspect
    return max(pure_crop, min_keep) if pure_crop < 1 else 1.0


def resize_tile(image: NDArray[np.uint8], width: int, height: int) -> NDArray[np.uint8]:
    """Scale a tile to ``width`` x ``height``.

    Area interpolation when shrinking (anti-aliased, Lecture 2), cubic when
    enlarging, as in `computer_vision.data.ingest.standardise`.
    """
    shrinking = width < image.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    return np.asarray(resized, dtype=np.uint8)
