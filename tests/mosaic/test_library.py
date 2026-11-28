from pathlib import Path

import cv2
import numpy as np
import pytest

from computer_vision.mosaic.library import load_library


def write_collection(folder: Path, tiles: dict[str, np.ndarray]) -> None:
    (folder / "tiles").mkdir(parents=True)
    rows = ["tile_id,mean_b"]
    for tile_id, image in tiles.items():
        cv2.imwrite(str(folder / "tiles" / f"{tile_id}.png"), image)
        rows.append(f"{tile_id},0")
    (folder / "tiles.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_loads_tiles_sorted_by_id(tmp_path: Path) -> None:
    red = np.zeros((9, 8, 3), np.uint8)
    red[..., 2] = 255
    blue = np.zeros((9, 8, 3), np.uint8)
    blue[..., 0] = 255
    write_collection(tmp_path / "demo", {"b_tile": blue, "a_tile": red})
    library = load_library(tmp_path / "demo")
    assert library.tile_ids == ("a_tile", "b_tile")
    assert library.images.shape == (2, 9, 8, 3)
    assert library.tile_size == (8, 9)
    assert len(library) == 2
    assert library.images[0, 0, 0].tolist() == [0, 0, 255]  # red came first


def test_rejects_tiles_of_different_sizes(tmp_path: Path) -> None:
    write_collection(
        tmp_path / "mixed",
        {"a": np.zeros((9, 8, 3), np.uint8), "b": np.zeros((8, 8, 3), np.uint8)},
    )
    with pytest.raises(ValueError, match="differ in size"):
        load_library(tmp_path / "mixed")


def test_missing_png_fails_loudly(tmp_path: Path) -> None:
    write_collection(tmp_path / "c", {"a": np.zeros((9, 8, 3), np.uint8)})
    (tmp_path / "c" / "tiles" / "a.png").unlink()
    with pytest.raises(FileNotFoundError):
        load_library(tmp_path / "c")
