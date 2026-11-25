"""Contact sheets: many tiles on one image, each labelled with a number.

Used for the manual review of the automatic crops (the reviewer types the
number of a bad crop into ``review.csv``) and for the report's figures of
each collection.
"""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

LABEL_HEIGHT = 22
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def letterbox(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """Scale an image to fit a ``size`` x ``size`` white square, keeping its shape.

    For viewing only: the reviewer must see the whole crop, undistorted.
    """
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    small = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    square = np.full((size, size, 3), 255, dtype=np.uint8)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    square[y0 : y0 + new_h, x0 : x0 + new_w] = small
    return square


def contact_sheet(
    images: Sequence[NDArray[np.uint8]],
    labels: Sequence[str],
    columns: int = 10,
    cell: int = 160,
) -> NDArray[np.uint8]:
    """Arrange images in a grid, each above its label.

    Parameters
    ----------
    images : Sequence[NDArray[np.uint8]]
        BGR images of any size.
    labels : Sequence[str]
        One short label per image (e.g. its review number).
    columns : int, optional
        Images per row.
    cell : int, optional
        Side of the square each image is fitted into, in pixels.

    Returns
    -------
    NDArray[np.uint8]
        The sheet, white background, a thin grey line around each cell.

    Raises
    ------
    ValueError
        If there are no images or the numbers of images and labels differ.
    """
    if not images or len(images) != len(labels):
        raise ValueError("need at least one image and exactly one label per image")
    rows = -(-len(images) // columns)  # ceiling division
    tile_h = cell + LABEL_HEIGHT
    sheet = np.full((rows * tile_h, columns * cell, 3), 255, dtype=np.uint8)
    for n, (image, label) in enumerate(zip(images, labels, strict=True)):
        r, c = divmod(n, columns)
        y, x = r * tile_h, c * cell
        sheet[y : y + cell, x : x + cell] = letterbox(image, cell)
        cv2.putText(
            sheet,
            label,
            (x + 4, y + cell + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            BLACK,
            1,
        )
        cv2.rectangle(sheet, (x, y), (x + cell - 1, y + tile_h - 1), (200, 200, 200), 1)
    return sheet
