"""Cut a photo into a grid of tile-sized cells and put tiles back together.

A mosaic of ``n_rows`` x ``n_cols`` tiles of ``tile_w`` x ``tile_h`` pixels is
an image of ``n_rows * tile_h`` x ``n_cols * tile_w`` pixels. This module does
the bookkeeping between that image and its cells:

1. `grid_shape` chooses the number of rows that keeps the photo's shape
   (Lecture 3, Photomosaic Variant 1).
2. `fit_to_grid` resizes the photo to exactly that canvas.
3. `split_cells` views the canvas as an array of cells, one per tile slot.
4. `assemble` does the reverse: it puts one tile into every slot.

Worked example: a 4000 x 3000 photo with 80 columns of 78 x 90 tiles gives
``round(80 * 78 * 3000 / (90 * 4000)) = 52`` rows, a canvas of 6240 x 4680
pixels and 4160 cells.
"""

import cv2
import numpy as np
from numpy.typing import NDArray


def grid_shape(
    image_w: int, image_h: int, tile_w: int, tile_h: int, n_cols: int
) -> tuple[int, int]:
    """Return ``(n_rows, n_cols)`` for a photo, keeping its aspect ratio.

    The mosaic is ``n_cols * tile_w`` wide; for its height/width to equal the
    photo's ``image_h / image_w``, ``n_rows * tile_h`` must equal
    ``n_cols * tile_w * image_h / image_w``. Rows are whole, so the ratio is
    rounded to the nearest integer (at least 1).

    Parameters
    ----------
    image_w, image_h : int
        Size of the photo in pixels.
    tile_w, tile_h : int
        Size of one tile in pixels.
    n_cols : int
        Number of tiles per row, chosen by the user.

    Returns
    -------
    tuple[int, int]
        ``(n_rows, n_cols)``.

    Raises
    ------
    ValueError
        If any size or ``n_cols`` is not positive.
    """
    if min(image_w, image_h, tile_w, tile_h, n_cols) <= 0:
        raise ValueError("sizes and n_cols must be positive")
    n_rows = max(1, round(n_cols * tile_w * image_h / (tile_h * image_w)))
    return n_rows, n_cols


def fit_to_grid(
    image: NDArray[np.uint8], tile_w: int, tile_h: int, n_cols: int
) -> NDArray[np.uint8]:
    """Resize a photo to the canvas of its mosaic.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Photo of shape (H, W, 3).
    tile_w, tile_h : int
        Size of one tile in pixels.
    n_cols : int
        Number of tiles per row.

    Returns
    -------
    NDArray[np.uint8]
        Shape ``(n_rows * tile_h, n_cols * tile_w, 3)`` with ``n_rows`` from
        `grid_shape`. Area interpolation when shrinking (averages the source
        pixels, no aliasing, Lecture 2), cubic when enlarging.
    """
    height, width = image.shape[:2]
    n_rows, _ = grid_shape(width, height, tile_w, tile_h, n_cols)
    out_w, out_h = n_cols * tile_w, n_rows * tile_h
    shrinking = out_w * out_h < width * height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    resized = cv2.resize(image, (out_w, out_h), interpolation=interpolation)
    return np.asarray(resized, dtype=np.uint8)


def split_cells(
    canvas: NDArray[np.uint8], tile_w: int, tile_h: int
) -> NDArray[np.uint8]:
    """View a canvas as a grid of cells.

    Parameters
    ----------
    canvas : NDArray[np.uint8]
        Image of shape ``(n_rows * tile_h, n_cols * tile_w, 3)``.
    tile_w, tile_h : int
        Size of one cell in pixels.

    Returns
    -------
    NDArray[np.uint8]
        Shape ``(n_rows, n_cols, tile_h, tile_w, 3)``; ``cells[i, j]`` is the
        cell in row ``i``, column ``j``. A view, not a copy: no pixels are
        moved, only the way NumPy walks through them.

    Raises
    ------
    ValueError
        If the canvas is not a whole number of cells in both directions.
    """
    height, width, channels = canvas.shape
    if height % tile_h or width % tile_w:
        raise ValueError(
            f"canvas {width} x {height} is not a whole number of "
            f"{tile_w} x {tile_h} cells"
        )
    n_rows, n_cols = height // tile_h, width // tile_w
    # (rows*th, cols*tw, 3) -> (rows, th, cols, tw, 3) -> (rows, cols, th, tw, 3)
    blocks = canvas.reshape(n_rows, tile_h, n_cols, tile_w, channels)
    return blocks.swapaxes(1, 2)


def assemble(
    tiles: NDArray[np.uint8], index_map: NDArray[np.intp]
) -> NDArray[np.uint8]:
    """Build the mosaic image from the chosen tile of every cell.

    Parameters
    ----------
    tiles : NDArray[np.uint8]
        Tile library of shape (N, tile_h, tile_w, 3).
    index_map : NDArray[np.intp]
        Shape (n_rows, n_cols); entry ``[i, j]`` is the index of the tile for
        cell ``(i, j)``.

    Returns
    -------
    NDArray[np.uint8]
        Mosaic of shape ``(n_rows * tile_h, n_cols * tile_w, 3)``; the exact
        inverse of `split_cells`.
    """
    n_rows, n_cols = index_map.shape
    _, tile_h, tile_w, channels = tiles.shape
    placed = tiles[index_map]  # (rows, cols, th, tw, 3)
    return placed.swapaxes(1, 2).reshape(n_rows * tile_h, n_cols * tile_w, channels)
