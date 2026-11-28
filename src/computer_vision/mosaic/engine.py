"""Build a rectangular photomosaic from a photo and a tile library (Task 1).

`build_mosaic` runs the steps in order:

1. grid: resize the photo to a whole number of tile-sized cells;
2. describe: one descriptor vector per cell and per tile;
3. match: nearest tile per cell, optionally never repeating a tile nearby;
4. assemble: paste the chosen tiles into one image, optionally tinted towards
   the photo's colour in each cell;
5. measure: tile usage and viewing-distance scores.

It returns everything the report needs about one mosaic in a `MosaicResult`.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from computer_vision.mosaic.descriptors import DESCRIPTORS, describe_in_batches
from computer_vision.mosaic.grid import assemble, fit_to_grid, split_cells
from computer_vision.mosaic.library import TileLibrary
from computer_vision.mosaic.matching import (
    UsageStats,
    distance_table,
    nearest,
    nearest_no_repeat,
    usage_stats,
)
from computer_vision.mosaic.quality import Scores, viewing_scores


@dataclass(frozen=True)
class MosaicResult:
    """One mosaic and its numbers."""

    image: NDArray[np.uint8]  # (n_rows * tile_h, n_cols * tile_w, 3), BGR
    index_map: NDArray[np.intp]  # (n_rows, n_cols) tile index per cell
    grid: tuple[int, int]  # (n_rows, n_cols)
    descriptor: str
    repeat_radius: int  # 0: pure nearest neighbour
    reuse_penalty: float
    tint: float  # 0: tiles unchanged
    usage: UsageStats
    scores: Scores


def build_mosaic(
    photo: NDArray[np.uint8],
    library: TileLibrary,
    n_cols: int = 90,
    descriptor: str = "mean_lab",
    repeat_radius: int = 2,
    reuse_penalty: float = 0.0,
    tint: float = 0.0,
) -> MosaicResult:
    """Turn a photo into a rectangular photomosaic.

    Parameters
    ----------
    photo : NDArray[np.uint8]
        Input photo, BGR, shape (H, W, 3).
    library : TileLibrary
        The tiles to build from.
    n_cols : int, optional
        Tiles per row; rows follow from the photo's aspect ratio. The default
        90 favours reconstruction quality over recognisable products (D26).
    descriptor : str, optional
        A key of `DESCRIPTORS`: ``"mean_bgr"``, ``"mean_lab"`` or
        ``"subblock_lab"``.
    repeat_radius : int, optional
        No tile appears twice within this many cells (D26); 0 turns the rule
        off (pure nearest neighbour, the course method).
    reuse_penalty : float, optional
        Extra cost per earlier use of a tile, in descriptor distance units
        (Delta E for CIELAB); only used when ``repeat_radius`` > 0. Off (0) by
        default (D26): the penalty grows with every use, so on photos the
        library covers poorly it forces visibly wrong colours, while
        ``repeat_radius`` already prevents nearby repeats.
    tint : float, optional
        Share of the photo's mean cell colour mixed into each placed tile, see
        `tint_cells`; 0 (the default) leaves the tiles untouched.

    Returns
    -------
    MosaicResult
        The mosaic, the tile chosen for each cell and its statistics.
    """
    describe = DESCRIPTORS[descriptor]
    tile_w, tile_h = library.tile_size
    canvas = fit_to_grid(photo, tile_w, tile_h, n_cols)
    cells = split_cells(canvas, tile_w, tile_h)
    n_rows = cells.shape[0]
    cell_vectors = describe_in_batches(cells.reshape(-1, tile_h, tile_w, 3), describe)
    tile_vectors = describe_in_batches(library.images, describe)
    table = distance_table(cell_vectors, tile_vectors)
    if repeat_radius:
        chosen = nearest_no_repeat(table, n_cols, repeat_radius, reuse_penalty)
    else:
        chosen = nearest(table)
    index_map = chosen.reshape(n_rows, n_cols)
    image = tint_cells(
        assemble(library.images, index_map), canvas, tile_w, tile_h, tint
    )
    return MosaicResult(
        image=image,
        index_map=index_map,
        grid=(n_rows, n_cols),
        descriptor=descriptor,
        repeat_radius=repeat_radius,
        reuse_penalty=reuse_penalty,
        tint=tint,
        usage=usage_stats(table, chosen),
        scores=viewing_scores(canvas, image),
    )


def tint_cells(
    mosaic: NDArray[np.uint8],
    canvas: NDArray[np.uint8],
    tile_w: int,
    tile_h: int,
    alpha: float,
) -> NDArray[np.uint8]:
    """Mix every placed tile with the photo's mean colour in its cell.

    ``out = (1 - alpha) * tile + alpha * mean(cell)``, pixel by pixel. This
    pulls each tile's colour towards its target and lowers the tile's inner
    contrast (white backgrounds, dark logos) by the factor ``1 - alpha``, which
    is what makes a mosaic look noisy from a distance. Only the cell's single
    mean colour is mixed in, never the photo's own detail inside the cell, so
    the structure still comes from the products. The report must say when a
    mosaic is tinted, because it no longer shows the tiles' true colours.

    Parameters
    ----------
    mosaic : NDArray[np.uint8]
        Assembled mosaic, shape (n_rows * tile_h, n_cols * tile_w, 3).
    canvas : NDArray[np.uint8]
        The photo resized to the same shape (from `fit_to_grid`).
    tile_w, tile_h : int
        Tile size in pixels.
    alpha : float
        Mixing share in [0, 1]: 0 returns the mosaic unchanged, 1 a grid of
        flat cell colours.

    Returns
    -------
    NDArray[np.uint8]
        The tinted mosaic, same shape as ``mosaic``.

    Raises
    ------
    ValueError
        If ``alpha`` is outside [0, 1] or the two images differ in shape.
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"tint must be between 0 and 1, got {alpha}")
    if mosaic.shape != canvas.shape:
        raise ValueError(f"shapes differ: {mosaic.shape} vs {canvas.shape}")
    if alpha == 0:
        return mosaic
    means = split_cells(canvas, tile_w, tile_h).mean(axis=(2, 3), keepdims=True)
    tiles = split_cells(mosaic, tile_w, tile_h).astype(np.float64)
    mixed = (1 - alpha) * tiles + alpha * means  # (rows, cols, th, tw, 3)
    n_rows, n_cols, _, _, channels = mixed.shape
    image = mixed.swapaxes(1, 2).reshape(n_rows * tile_h, n_cols * tile_w, channels)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)
