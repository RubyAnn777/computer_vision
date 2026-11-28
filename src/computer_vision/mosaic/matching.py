"""Choose a tile for every cell (D26).

After the descriptors, every cell and every tile is a vector. The best tile
for a cell is the nearest vector: a nearest-neighbour search with k = 1.

1. `distance_table` computes the squared Euclidean distance between every
   cell and every tile at once, an array of shape (cells, tiles).
2. `nearest` takes the smallest entry in each row: pure nearest neighbour,
   the course method.
3. `nearest_no_repeat` walks the grid in reading order and takes the nearest
   tile not already used within a small square around the cell, with an
   optional extra cost for every earlier use of a tile, so a flat region
   becomes neither a block of one product nor a regular pattern of a few.

Cost: the table needs cells x tiles x d multiply-adds and cells x tiles
numbers of memory; for 4,160 cells and 1,204 tiles that is 5 million numbers
(20 MB in float32), computed in well under a second. A KD-tree would only pay
off for far larger libraries, and less so at d = 12 (sub-blocks).
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


def distance_table(
    cells: NDArray[np.float32], tiles: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Return squared Euclidean distances between every cell and every tile.

    Uses ``|c - t|^2 = |c|^2 + |t|^2 - 2 c.t``, so the work is one matrix
    product instead of a (cells, tiles, d) difference array, which would need
    d times more memory.

    Parameters
    ----------
    cells : NDArray[np.float32]
        Cell descriptors, shape (M, d).
    tiles : NDArray[np.float32]
        Tile descriptors, shape (N, d).

    Returns
    -------
    NDArray[np.float32]
        Shape (M, N); entry ``[i, k]`` is the squared distance from cell ``i``
        to tile ``k``. Clipped at 0, because rounding can make the formula
        return tiny negative numbers for identical vectors.
    """
    c = cells.astype(np.float64)
    t = tiles.astype(np.float64)
    squared = (c**2).sum(axis=1)[:, None] + (t**2).sum(axis=1)[None, :] - 2 * c @ t.T
    return np.maximum(squared, 0).astype(np.float32)


def nearest(table: NDArray[np.float32]) -> NDArray[np.intp]:
    """Return the index of the nearest tile for every cell, shape (M,)."""
    return np.argmin(table, axis=1)


def nearest_no_repeat(
    table: NDArray[np.float32],
    n_cols: int,
    radius: int = 2,
    reuse_penalty: float = 0.0,
) -> NDArray[np.intp]:
    """Nearest tile per cell, with two rules against visible repetition.

    Cells are visited in reading order (row by row, left to right), and each
    takes the tile with the lowest cost

    ``cost(k) = distance(cell, k) + reuse_penalty * uses_so_far(k)``

    among the tiles it is allowed to use:

    1. **Local rule.** A tile already placed within ``radius`` cells
       (counting diagonals, a (2r + 1) x (2r + 1) square around the cell) is
       not allowed. Only placed cells can conflict: the ``radius`` cells to the
       left and the ``radius`` rows above. Two cells at most ``radius`` steps
       apart therefore never share a tile.
    2. **Global rule.** Every earlier use makes a tile ``reuse_penalty``
       distance units more expensive, so a flat region spreads over many
       similar tiles instead of cycling through the same few. Without it, the
       local rule alone tiles a flat region with a regular pattern of
       ``(radius + 1) ** 2`` tiles (radius 1: a 2 x 2 checkerboard).

    Distances are Euclidean (square roots of ``table``), so for CIELAB
    descriptors the penalty is in Delta E units: ``reuse_penalty=1`` means a
    tile used 3 times must be 3 Delta E closer than an unused one to win.

    Parameters
    ----------
    table : NDArray[np.float32]
        Squared distances of shape (n_rows * n_cols, N), rows in reading order.
    n_cols : int
        Cells per row of the grid.
    radius : int, optional
        Size of the no-repeat neighbourhood (at least 1).
    reuse_penalty : float, optional
        Extra cost per earlier use of a tile, in distance units; 0 turns the
        global rule off.

    Returns
    -------
    NDArray[np.intp]
        Tile index per cell, shape (M,).

    Raises
    ------
    ValueError
        If ``radius`` < 1, ``reuse_penalty`` < 0, there are too few tiles to
        satisfy the local rule, or the table is not a whole grid.
    """
    n_cells, n_tiles = table.shape
    if radius < 1 or reuse_penalty < 0:
        raise ValueError(
            f"need radius >= 1 and reuse_penalty >= 0, got {radius}, {reuse_penalty}"
        )
    most_forbidden = radius + radius * (2 * radius + 1)
    if n_tiles <= most_forbidden:
        raise ValueError(f"radius {radius} needs at least {most_forbidden + 1} tiles")
    if n_cells % n_cols:
        raise ValueError(f"{n_cells} cells do not form rows of {n_cols}")
    distances = np.sqrt(table, dtype=np.float64)
    uses = np.zeros(n_tiles)
    grid = np.full((n_cells // n_cols, n_cols), -1, dtype=np.intp)  # -1: not yet
    for i in range(n_cells):
        row, col = divmod(i, n_cols)
        above = grid[
            max(0, row - radius) : row, max(0, col - radius) : col + radius + 1
        ]
        left = grid[row, max(0, col - radius) : col]
        cost = distances[i] + reuse_penalty * uses
        cost[above.ravel()] = np.inf  # local rule
        cost[left] = np.inf
        k = int(np.argmin(cost))
        grid[row, col] = k
        uses[k] += 1
    return grid.ravel()


class UsageStats(NamedTuple):
    """How a mosaic uses its tile library."""

    distinct: int  # different tiles used
    max_uses: int  # uses of the most frequent tile
    mean_distance: float  # mean Euclidean descriptor distance, cell to tile


def usage_stats(table: NDArray[np.float32], chosen: NDArray[np.intp]) -> UsageStats:
    """Summarise tile reuse and matching error for one assignment."""
    counts = np.bincount(chosen)
    distances = np.sqrt(table[np.arange(len(chosen)), chosen])
    return UsageStats(
        int((counts > 0).sum()), int(counts.max()), float(distances.mean())
    )
