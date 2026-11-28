"""Load a standardised tile collection into memory.

A collection folder (built by ``scripts/build_tile_collection.py``) holds
``tiles.csv`` and one PNG per tile in ``tiles/``. The engine needs the tiles
as one array, so that choosing tile ``k`` for a cell is just ``tiles[k]``.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TileLibrary:
    """All tiles of one collection, in a fixed order."""

    name: str
    tile_ids: tuple[str, ...]
    images: NDArray[np.uint8]  # (N, tile_h, tile_w, 3), BGR

    @property
    def tile_size(self) -> tuple[int, int]:
        """Return ``(tile_w, tile_h)`` in pixels."""
        return self.images.shape[2], self.images.shape[1]

    def __len__(self) -> int:
        """Return the number of tiles."""
        return len(self.tile_ids)


def load_library(folder: str | Path) -> TileLibrary:
    """Read every tile listed in ``tiles.csv``, sorted by tile id.

    Sorting makes the tile index of a run independent of file-system order,
    so the same collection always gives the same mosaic (unlike the session 3
    notebook's unsorted ``os.listdir``).

    Parameters
    ----------
    folder : str or Path
        A collection folder, e.g.
        ``data/processed/tiles/collections/manual_keep070_w78h90``.

    Returns
    -------
    TileLibrary
        The tiles and their ids; the name is the folder name.

    Raises
    ------
    FileNotFoundError
        If ``tiles.csv`` or a listed PNG is missing.
    ValueError
        If the collection is empty or the tiles differ in size.
    """
    folder = Path(folder)
    with (folder / "tiles.csv").open(newline="", encoding="utf-8") as handle:
        tile_ids = sorted(row["tile_id"] for row in csv.DictReader(handle))
    if not tile_ids:
        raise ValueError(f"no tiles listed in {folder / 'tiles.csv'}")
    images = []
    for tile_id in tile_ids:
        path = folder / "tiles" / f"{tile_id}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        images.append(image)
    if len({image.shape for image in images}) != 1:
        raise ValueError(f"tiles in {folder} differ in size")
    stacked = np.asarray(np.stack(images), dtype=np.uint8)
    return TileLibrary(folder.name, tuple(tile_ids), stacked)
