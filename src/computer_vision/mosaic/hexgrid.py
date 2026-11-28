"""Build a honeycomb photomosaic from hexagonal tiles (Task 2).

Geometry (pointy-top hexagons, D22 chose the tile shape for this)
-----------------------------------------------------------------
A pointy-top regular hexagon of circumradius R fits a box sqrt(3) R wide and
2 R high: width / height = sqrt(3) / 2 = 0.866. Our tiles are 78 x 90 pixels
(78 / 90 = 0.867), so each tile is exactly the box of one hexagon and the
hexagon is the tile minus its four corner triangles (25 % of the area). No
product is stretched. A flat-top hexagon would need a box 2 / sqrt(3) = 1.15
times wider than high, so inside our portrait tiles it could use only 56 % of
each product image.

In the honeycomb, hexagons touch edge to edge:

- along a row the centres are one tile width ``W`` apart;
- rows are ``3 H / 4`` apart (67.5 pixels for H = 90), because each row's
  pointed top fits into the notches of the row above;
- every odd row is shifted right by ``W / 2``.

Hexagon (r, c) therefore has its box at top ``r * 3H/4`` and left
``c * W + (r odd) * W / 2``. The tiling is exact for any W and H (it is a
stretched regular honeycomb), so every pixel of the output belongs to exactly
one hexagon. Instead of pasting masked tiles and rounding the half-pixel row
spacing, we ask for every output pixel which hexagon it falls in, and copy
the matching pixel of that hexagon's tile. That leaves no gaps and no overlaps.

Steps of `build_hex_mosaic`
---------------------------
1. resize the photo to ``n_cols`` tile widths, keeping its aspect ratio;
2. find the hexagon of every pixel (`hex_labels`);
3. describe every hexagon of the photo by the mean colour of its pixels, and
   every tile by the mean over its hexagon (or, to compare, its whole box);
4. match with the same nearest-neighbour rule as Task 1, with the no-repeat
   radius counted in hexagon steps (each hexagon has six neighbours);
5. paint every pixel from the chosen tile, and score as in Task 1.

Boundaries: hexagons do not fit a rectangle. ``boundary="crop"`` covers the
photo with hexagons that reach past its edges and crops the result to the
photo (the lecture's method); edge hexagons are matched on their visible part
only. ``boundary="whole"`` keeps only hexagons that lie fully inside the
photo, and fills the zigzag margin with a background colour.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from computer_vision.mosaic.descriptors import (
    describe_in_batches,
    mean_bgr,
    mean_lab,
    to_lab,
)
from computer_vision.mosaic.library import TileLibrary
from computer_vision.mosaic.matching import UsageStats, distance_table, usage_stats
from computer_vision.mosaic.quality import Scores, viewing_scores

Boundary = Literal["crop", "whole"]
BAND_ROWS = 256  # canvas rows handled at once; keeps memory low for big mosaics


def hex_mask(tile_w: int, tile_h: int) -> NDArray[np.bool_]:
    """Return the pointy-top hexagon inscribed in a ``tile_w`` x ``tile_h`` box.

    A pixel is inside when its centre is: at horizontal distance ``dx`` from
    the middle, the hexagon reaches ``H/2 - (H/4) * dx / (W/2)`` above and
    below the middle (the slanted edges), and ``dx`` is at most ``W/2``.

    Returns
    -------
    NDArray[np.bool_]
        Shape (tile_h, tile_w), True inside the hexagon; about 75 % of pixels.
    """
    y, x = np.mgrid[0:tile_h, 0:tile_w] + 0.5
    return _inside(x, y, tile_w, tile_h)


def _inside(
    x: NDArray[np.float64], y: NDArray[np.float64], tile_w: float, tile_h: float
) -> NDArray[np.bool_]:
    """Test whether box-relative points (x, y) lie in the box's hexagon."""
    dx = np.abs(x - tile_w / 2)
    dy = np.abs(y - tile_h / 2)
    return (dx <= tile_w / 2) & (dy <= tile_h / 2 - (tile_h / 4) * dx / (tile_w / 2))


@dataclass(frozen=True)
class HexLayout:
    """Where the hexagons of one mosaic are."""

    tile_w: int
    tile_h: int
    width: int  # canvas size in pixels
    height: int
    n_rows: int  # hexagon rows touching the canvas, counting row -1
    n_cols: int  # hexagon columns per row, counting column -1

    @property
    def row_step(self) -> float:
        """Vertical distance between hexagon rows, 3/4 of the tile height."""
        return 0.75 * self.tile_h

    def box(self, r: NDArray[np.intp], c: NDArray[np.intp]) -> tuple[NDArray, NDArray]:
        """Top and left of the boxes of hexagons (r, c), in pixels (floats)."""
        return r * self.row_step, c * self.tile_w + (r % 2) * (self.tile_w / 2)

    def position(
        self, ids: NDArray[np.intp]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Row and column of hexagon ids (inverse of the id formula)."""
        r, c = np.divmod(ids, self.n_cols)
        return r - 1, c - 1


def make_layout(width: int, height: int, tile_w: int, tile_h: int) -> HexLayout:
    """Lay hexagons over a ``width`` x ``height`` canvas, past every edge.

    Row 0 starts at the top edge; row -1 fills the notches above it. Column
    0 starts at the left edge; column -1 fills the half hexagons that odd rows
    leave on the left. Hexagon (r, c) gets id ``(r + 1) * n_cols + (c + 1)``.
    """
    n_rows = int(np.ceil(height / (0.75 * tile_h))) + 2
    n_cols = int(np.ceil(width / tile_w)) + 2
    return HexLayout(tile_w, tile_h, width, height, n_rows, n_cols)


def hex_labels(
    layout: HexLayout, y0: int, y1: int
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Find the hexagon of every pixel in canvas rows ``y0`` to ``y1``.

    A pixel at height y lies in the box of row ``floor(y / step)`` or of the
    row above (boxes are 4/3 steps high), so only those two rows are tested.

    Returns
    -------
    ids, ty, tx : NDArray[np.intp]
        Each of shape (y1 - y0, width): the hexagon id of every pixel and the
        pixel's row and column inside that hexagon's tile.
    """
    w, h = layout.tile_w, layout.tile_h
    py, px = np.mgrid[y0:y1, 0 : layout.width].astype(np.float64) + 0.5
    lower = np.floor(py / layout.row_step).astype(np.intp)

    def column(r: NDArray[np.intp]) -> NDArray[np.intp]:
        return np.floor((px - (r % 2) * (w / 2)) / w).astype(np.intp)

    top, left = layout.box(lower, column(lower))
    r = np.where(_inside(px - left, py - top, w, h), lower, lower - 1)
    c = column(r)
    top, left = layout.box(r, c)
    ty = np.clip(np.floor(py - top), 0, h - 1).astype(np.intp)
    tx = np.clip(np.floor(px - left), 0, w - 1).astype(np.intp)
    ids = (r + 1) * layout.n_cols + (c + 1)
    return ids, ty, tx


def _bands(layout: HexLayout) -> Iterator[tuple[int, int]]:
    for y0 in range(0, layout.height, BAND_ROWS):
        yield y0, min(y0 + BAND_ROWS, layout.height)


def whole_hexagons(layout: HexLayout) -> NDArray[np.bool_]:
    """Which hexagon ids lie fully inside the canvas (their box does)."""
    ids = np.arange(layout.n_rows * layout.n_cols)
    r, c = layout.position(ids)
    top, left = layout.box(r, c)
    return (
        (top >= 0)
        & (left >= 0)
        & (top + layout.tile_h <= layout.height)
        & (left + layout.tile_w <= layout.width)
    )


def hex_distance(
    r1: NDArray[np.intp] | int,
    c1: NDArray[np.intp] | int,
    r2: NDArray[np.intp] | int,
    c2: NDArray[np.intp] | int,
) -> NDArray[np.intp]:
    """Count the steps between hexagons (r1, c1) and (r2, c2) in the honeycomb.

    Converts the row/column ("odd rows shifted right") coordinates to cube
    coordinates, where the distance is the largest coordinate difference.
    """

    def cube(r, c):
        x = np.asarray(c) - (np.asarray(r) - (np.asarray(r) & 1)) // 2
        z = np.asarray(r)
        return x, z, -x - z

    a, b = cube(r1, c1), cube(r2, c2)
    return np.maximum.reduce([np.abs(a[i] - b[i]) for i in range(3)])


def _earlier_neighbours(radius: int) -> dict[int, list[tuple[int, int]]]:
    """Offsets (dr, dc) to hexagons visited earlier and within ``radius``.

    Hexagons are visited in reading order, so "earlier" means a row above, or
    the same row to the left. The offsets differ for even and odd rows.
    """
    offsets: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    for parity in (0, 1):
        for dr in range(-radius, 1):
            for dc in range(-radius - 1, radius + 2):
                if dr == 0 and dc >= 0:
                    continue
                if hex_distance(parity, 0, parity + dr, dc) <= radius:
                    offsets[parity].append((dr, dc))
    return offsets


def hex_no_repeat(
    table: NDArray[np.float32],
    rows: NDArray[np.intp],
    cols: NDArray[np.intp],
    radius: int,
) -> NDArray[np.intp]:
    """Nearest tile per hexagon; no tile twice within ``radius`` hexagon steps.

    Same rule as `matching.nearest_no_repeat` (Task 1, D26) with the distance
    counted in the honeycomb: hexagons are visited in reading order and each
    takes the nearest tile not already used by an earlier hexagon within
    ``radius`` steps. ``radius`` 0 is pure nearest neighbour.

    Parameters
    ----------
    table : NDArray[np.float32]
        Squared distances, shape (M, N), one row per hexagon.
    rows, cols : NDArray[np.intp]
        Row and column of each hexagon, shape (M,), in reading order.
    radius : int
        Size of the no-repeat neighbourhood in hexagon steps.

    Returns
    -------
    NDArray[np.intp]
        Tile index per hexagon, shape (M,).
    """
    if radius == 0:
        return np.argmin(table, axis=1)
    offsets = _earlier_neighbours(radius)
    most = max(len(o) for o in offsets.values())
    if table.shape[1] <= most:
        raise ValueError(f"radius {radius} needs at least {most + 1} tiles")
    r0, c0 = rows.min(), cols.min()
    grid = np.full((rows.max() - r0 + 1, cols.max() - c0 + 1), -1, dtype=np.intp)
    chosen = np.empty(len(rows), dtype=np.intp)
    for i, (r, c) in enumerate(zip(rows, cols, strict=True)):
        cost = table[i].astype(np.float64)
        for dr, dc in offsets[int(r) % 2]:
            rr, cc = r - r0 + dr, c - c0 + dc
            if (
                0 <= rr < grid.shape[0]
                and 0 <= cc < grid.shape[1]
                and grid[rr, cc] >= 0
            ):
                cost[grid[rr, cc]] = np.inf
        chosen[i] = grid[r - r0, c - c0] = int(np.argmin(cost))
    return chosen


@dataclass(frozen=True)
class HexMosaicResult:
    """One honeycomb mosaic and its numbers."""

    image: NDArray[np.uint8]  # same size as canvas, BGR
    canvas: NDArray[np.uint8]  # the resized photo it was built from
    rows: NDArray[np.intp]  # row of each placed hexagon
    cols: NDArray[np.intp]  # column of each placed hexagon
    chosen: NDArray[np.intp]  # tile index of each placed hexagon
    n_partial: int  # placed hexagons cut by the canvas edge
    descriptor: str
    mask_tiles: bool
    boundary: Boundary
    repeat_radius: int
    usage: UsageStats
    scores: Scores


HEX_DESCRIPTORS = {"mean_bgr": mean_bgr, "mean_lab": mean_lab}


def build_hex_mosaic(
    photo: NDArray[np.uint8],
    library: TileLibrary,
    n_cols: int = 90,
    descriptor: str = "mean_lab",
    repeat_radius: int = 2,
    mask_tiles: bool = True,
    boundary: Boundary = "crop",
    background: tuple[int, int, int] = (255, 255, 255),
) -> HexMosaicResult:
    """Turn a photo into a honeycomb photomosaic.

    Parameters
    ----------
    photo : NDArray[np.uint8]
        Input photo, BGR, shape (H, W, 3).
    library : TileLibrary
        Tiles whose box is the hexagon's box (78 x 90, D22).
    n_cols : int, optional
        Hexagons per row; the canvas is ``n_cols`` tile widths wide and keeps
        the photo's aspect ratio. The default 90 matches the rectangular
        mosaics (D26): reconstruction quality first.
    descriptor : str, optional
        ``"mean_lab"`` (default) or ``"mean_bgr"``: mean colour of the
        photo's pixels inside each hexagon and of each tile.
    repeat_radius : int, optional
        No tile twice within this many hexagon steps; 0 turns the rule off.
    mask_tiles : bool, optional
        True: a tile is described by its pixels inside the hexagon, the part
        that is shown. False: by its whole box, corners included (to compare).
    boundary : {"crop", "whole"}, optional
        "crop": cover past the edges and crop, matching edge hexagons on
        their visible part. "whole": only hexagons fully inside the photo.
    background : tuple[int, int, int], optional
        BGR colour of the margin left by ``boundary="whole"``.

    Returns
    -------
    HexMosaicResult
        The mosaic, the canvas, the placed hexagons and their statistics.
    """
    if descriptor not in HEX_DESCRIPTORS:
        raise ValueError(
            f"hexagons support {sorted(HEX_DESCRIPTORS)}, got {descriptor!r}"
        )
    if boundary not in ("crop", "whole"):
        raise ValueError(f"boundary must be 'crop' or 'whole', got {boundary!r}")
    tile_w, tile_h = library.tile_size
    width = n_cols * tile_w
    height = max(1, round(width * photo.shape[0] / photo.shape[1]))
    shrink = width < photo.shape[1]
    canvas = np.asarray(
        cv2.resize(
            photo,
            (width, height),
            interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC,
        ),
        dtype=np.uint8,
    )
    layout = make_layout(width, height, tile_w, tile_h)
    n_ids = layout.n_rows * layout.n_cols

    # step 3a: sum the colour of every hexagon's visible pixels, band by band
    sums = np.zeros((n_ids, 3))
    counts = np.zeros(n_ids)
    for y0, y1 in _bands(layout):
        ids, _, _ = hex_labels(layout, y0, y1)
        band = canvas[y0:y1]
        values = to_lab(band) if descriptor == "mean_lab" else band.astype(np.float32)
        flat = ids.ravel()
        counts += np.bincount(flat, minlength=n_ids)
        for k in range(3):
            sums[:, k] += np.bincount(
                flat, weights=values[..., k].ravel(), minlength=n_ids
            )
    placed = counts > 0
    if boundary == "whole":
        placed &= whole_hexagons(layout)
    if not placed.any():
        raise ValueError("no whole hexagon fits; use more columns or boundary='crop'")
    ids_placed = np.flatnonzero(placed)  # reading order: ids grow along rows
    cell_vectors = (sums[ids_placed] / counts[ids_placed, None]).astype(np.float32)

    # step 3b: tile descriptors, over the hexagon or over the whole box
    mask = hex_mask(tile_w, tile_h) if mask_tiles else None
    describe = HEX_DESCRIPTORS[descriptor]
    tile_vectors = describe_in_batches(library.images, lambda im: describe(im, mask))

    # step 4: match
    table = distance_table(cell_vectors, tile_vectors)
    rows, cols = layout.position(ids_placed)
    chosen = hex_no_repeat(table, rows, cols, repeat_radius)

    # step 5: paint every pixel from its hexagon's tile
    tile_of = np.full(n_ids, -1, dtype=np.intp)
    tile_of[ids_placed] = chosen
    image = np.empty_like(canvas)
    for y0, y1 in _bands(layout):
        ids, ty, tx = hex_labels(layout, y0, y1)
        k = tile_of[ids]
        band = library.images[np.maximum(k, 0), ty, tx]
        band[k < 0] = background
        image[y0:y1] = band
    whole = whole_hexagons(layout)
    return HexMosaicResult(
        image=image,
        canvas=canvas,
        rows=rows,
        cols=cols,
        chosen=chosen,
        n_partial=int((~whole[ids_placed]).sum()),
        descriptor=descriptor,
        mask_tiles=mask_tiles,
        boundary=boundary,
        repeat_radius=repeat_radius,
        usage=usage_stats(table, chosen),
        scores=viewing_scores(canvas, image),
    )
