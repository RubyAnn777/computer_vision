"""Cut product crops out of catalog pages and name them.

Both tile collections are built from boxes on page images: the hand-drawn
boxes (manual collection) and the detector's boxes (automatic collection).
This module holds the shared, pure steps, so both collections are cut in
exactly the same way:

1. `to_pixel_box` turns a float box (the detector's output) into whole pixel
   edges inside the page.
2. `reading_order` numbers the boxes of a page from top left to bottom right.
3. `crop` cuts one box out of the page image.
4. `tile_id` names the crop ``<page_id>_<source><NNN>``, for example
   ``mueller_prospekt-2024-kw46_p003_b007``: box 7 on that page.

Coordinates are pixel *edges*, as in the labelling tool: a box
``(x0, y0, x1, y1)`` covers columns ``x0 .. x1 - 1`` and rows
``y0 .. y1 - 1``, so it is ``x1 - x0`` pixels wide and NumPy's half-open
slice ``image[y0:y1, x0:x1]`` cuts exactly that region.
"""

import math
import re
from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from computer_vision.data.ids import parse_page_id
from computer_vision.data.yolo import Box
from computer_vision.detection.threshold import BoxF

Source = Literal["b", "d"]  # b: hand-drawn box, d: detection
INDEX_WIDTH = 3
MAX_INDEX = 10**INDEX_WIDTH - 1

_TILE_ID = re.compile(rf"(.+)_([bd])(\d{{{INDEX_WIDTH}}})")


class TileKey(NamedTuple):
    """The three fields encoded in a tile id."""

    page_id: str
    source: Source
    index: int


def tile_id(page_id: str, source: Source, index: int) -> str:
    """Build the id of one crop.

    Parameters
    ----------
    page_id : str
        Id of the page the crop comes from.
    source : {"b", "d"}
        ``"b"`` for a hand-drawn box, ``"d"`` for a detection.
    index : int
        0-based position of the box in reading order on its page.

    Returns
    -------
    str
        ``<page_id>_<source><NNN>``.

    Raises
    ------
    ValueError
        If the page id is malformed, the source is not ``"b"`` or ``"d"``, or
        the index is outside ``0 .. MAX_INDEX``.
    """
    parse_page_id(page_id)
    if source not in ("b", "d"):
        raise ValueError(f"source must be 'b' or 'd', got {source!r}")
    if not 0 <= index <= MAX_INDEX:
        raise ValueError(f"index must be in 0..{MAX_INDEX}, got {index}")
    return f"{page_id}_{source}{index:0{INDEX_WIDTH}d}"


def parse_tile_id(value: str) -> TileKey:
    """Split a tile id back into its fields.

    Parameters
    ----------
    value : str
        An id produced by `tile_id`.

    Returns
    -------
    TileKey
        Page id, source and index.

    Raises
    ------
    ValueError
        If the string is not a well-formed tile id.
    """
    match = _TILE_ID.fullmatch(value)
    if match is None:
        raise ValueError(f"not a valid tile id: {value!r}")
    page_id, source, index = match.groups()
    parse_page_id(page_id)  # raises if the page part is malformed
    return TileKey(page_id, "b" if source == "b" else "d", int(index))


def round_half_up(value: float) -> int:
    """Round to the nearest integer, halves upwards (like JavaScript's Math.round).

    Python's `round` and NumPy's `rint` send halves to the nearest even number
    (``round(10.5) == 10``). The labelling tool uses ``Math.round``, so this
    function makes detector boxes follow the same rule as hand-drawn ones.
    """
    return math.floor(value + 0.5)


def to_pixel_box(box: BoxF, width: int, height: int) -> Box:
    """Round a float box to whole pixel edges and clip it to the image.

    Parameters
    ----------
    box : BoxF
        ``(xmin, ymin, xmax, ymax)`` in pixels, possibly fractional.
    width, height : int
        Size of the page image in pixels.

    Returns
    -------
    Box
        Integer ``(x0, y0, x1, y1)`` with ``0 <= x0 < x1 <= width`` and
        ``0 <= y0 < y1 <= height``.

    Raises
    ------
    ValueError
        If the box has no area once rounded and clipped.
    """
    x0, y0, x1, y1 = (round_half_up(v) for v in box)
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"box {box} has no area inside a {width} x {height} image")
    return x0, y0, x1, y1


def reading_order(boxes: Sequence[Box]) -> list[int]:
    """Return the box indices sorted from top to bottom, then left to right.

    Numbering by position, not by file order or detector confidence, makes
    the tile ids depend only on where the products are on the page.

    Parameters
    ----------
    boxes : Sequence[Box]
        Boxes of one page.

    Returns
    -------
    list[int]
        Indices into ``boxes``, sorted by ``(y0, x0)``; ties keep the
        original order, so the result is deterministic.
    """
    return sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))


def crop(image: NDArray[np.uint8], box: Box) -> NDArray[np.uint8]:
    """Cut one box out of an image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Page image of shape (H, W, 3).
    box : Box
        Integer pixel edges ``(x0, y0, x1, y1)`` inside the image.

    Returns
    -------
    NDArray[np.uint8]
        A copy of shape ``(y1 - y0, x1 - x0, 3)``. A copy, not a view, so the
        crop does not keep the whole page in memory and editing it can never
        change the page.

    Raises
    ------
    ValueError
        If the box has no area or reaches outside the image.
    """
    x0, y0, x1, y1 = box
    height, width = image.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"box {box} is empty or outside a {width} x {height} image")
    return image[y0:y1, x0:x1].copy()
