"""Build a rectangular photomosaic from a photo and a tile library (Task 1).

`build_mosaic` runs the steps in order:

1. grid: resize the photo to a whole number of tile-sized cells;
2. describe: one descriptor vector per cell and per tile;
3. match: nearest tile per cell, optionally never repeating a tile nearby;
4. assemble: paste the chosen tiles into one image;
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
    usage: UsageStats
    scores: Scores


def build_mosaic(
    photo: NDArray[np.uint8],
    library: TileLibrary,
    n_cols: int,
    descriptor: str = "mean_lab",
    repeat_radius: int = 2,
    reuse_penalty: float = 0.0,
) -> MosaicResult:
    """Turn a photo into a rectangular photomosaic.

    Parameters
    ----------
    photo : NDArray[np.uint8]
        Input photo, BGR, shape (H, W, 3).
    library : TileLibrary
        The tiles to build from.
    n_cols : int
        Tiles per row; rows follow from the photo's aspect ratio.
    descriptor : str, optional
        A key of `DESCRIPTORS`: ``"mean_bgr"``, ``"mean_lab"`` or
        ``"subblock_lab"``.
    repeat_radius : int, optional
        No tile appears twice within this many cells (D26); 0 turns the rule
        off (pure nearest neighbour, the course method).
    reuse_penalty : float, optional
        Extra cost per earlier use of a tile, in descriptor distance units
        (Delta E for CIELAB); only used when ``repeat_radius`` > 0.

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
    image = assemble(library.images, index_map)
    return MosaicResult(
        image=image,
        index_map=index_map,
        grid=(n_rows, n_cols),
        descriptor=descriptor,
        repeat_radius=repeat_radius,
        reuse_penalty=reuse_penalty,
        usage=usage_stats(table, chosen),
        scores=viewing_scores(canvas, image),
    )
