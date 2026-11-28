import numpy as np
import pytest

from computer_vision.mosaic.engine import build_mosaic, tint_cells
from computer_vision.mosaic.library import TileLibrary
from computer_vision.mosaic.quality import viewing_scores

TILE_W, TILE_H = 8, 10


def solid_library(colours: list[tuple[int, int, int]]) -> TileLibrary:
    images = np.stack([np.full((TILE_H, TILE_W, 3), c, np.uint8) for c in colours])
    return TileLibrary("solid", tuple(f"t{i}" for i in range(len(colours))), images)


@pytest.mark.parametrize("descriptor", ["mean_bgr", "mean_lab", "subblock_lab"])
def test_one_colour_photo_selects_the_matching_tile(descriptor: str) -> None:
    library = solid_library([(0, 0, 0), (40, 160, 220), (255, 255, 255)])
    photo = np.full((50, 64, 3), (40, 160, 220), np.uint8)
    result = build_mosaic(
        photo, library, n_cols=8, descriptor=descriptor, repeat_radius=0
    )
    assert (result.index_map == 1).all()
    assert result.usage.distinct == 1


def test_a_photo_made_of_tiles_is_rebuilt_exactly() -> None:
    rng = np.random.default_rng(3)
    colours: list[tuple[int, int, int]] = [
        (int(r), int(g), int(b)) for r, g, b in rng.integers(0, 256, (6, 3))
    ]
    library = solid_library(colours)
    truth = rng.integers(0, 6, (4, 5))
    photo = np.concatenate(
        [np.concatenate([library.images[k] for k in row], axis=1) for row in truth]
    )
    result = build_mosaic(photo, library, n_cols=5, repeat_radius=0)
    assert result.grid == (4, 5)
    assert np.array_equal(result.index_map, truth)
    assert np.array_equal(result.image, photo)
    assert result.scores.ssim == pytest.approx(1.0)
    assert result.scores.delta_e == pytest.approx(0.0, abs=1e-6)


def test_output_shape_follows_the_grid() -> None:
    library = solid_library([(v, v, v) for v in range(0, 256, 15)])  # 18 greys
    photo = np.zeros((300, 400, 3), np.uint8)
    result = build_mosaic(photo, library, n_cols=10)
    n_rows, n_cols = result.grid
    assert result.image.shape == (n_rows * TILE_H, n_cols * TILE_W, 3)
    assert result.index_map.shape == (n_rows, n_cols)


def test_scores_drop_for_an_unrelated_image() -> None:
    rng = np.random.default_rng(4)
    photo = np.zeros((80, 80, 3), np.uint8)
    photo[:, 40:] = 255
    noise = rng.integers(0, 256, photo.shape, dtype=np.uint8)
    same = viewing_scores(photo, photo)
    other = viewing_scores(photo, noise)
    assert same.ssim > other.ssim
    assert same.delta_e < other.delta_e


def test_tint_zero_keeps_the_tiles_and_one_gives_flat_cell_colours() -> None:
    rng = np.random.default_rng(5)
    mosaic = rng.integers(0, 256, (2 * TILE_H, 3 * TILE_W, 3), dtype=np.uint8)
    canvas = rng.integers(0, 256, mosaic.shape, dtype=np.uint8)
    assert np.array_equal(tint_cells(mosaic, canvas, TILE_W, TILE_H, 0.0), mosaic)
    flat = tint_cells(mosaic, canvas, TILE_W, TILE_H, 1.0)
    cell = canvas[:TILE_H, :TILE_W].reshape(-1, 3).mean(axis=0)
    assert np.abs(flat[:TILE_H, :TILE_W] - np.rint(cell)).max() <= 0.5
    assert len(np.unique(flat[:TILE_H, :TILE_W].reshape(-1, 3), axis=0)) == 1


def test_tint_halves_the_contrast_inside_a_tile() -> None:
    mosaic = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    mosaic[:, TILE_W // 2 :] = 200  # tile: half black, half light grey
    canvas = np.full_like(mosaic, 100)
    out = tint_cells(mosaic, canvas, TILE_W, TILE_H, 0.5).astype(int)
    assert out[0, 0].tolist() == [50, 50, 50]
    assert out[0, -1].tolist() == [150, 150, 150]


def test_tint_rejects_values_outside_zero_to_one() -> None:
    image = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    with pytest.raises(ValueError, match="tint"):
        tint_cells(image, image, TILE_W, TILE_H, 1.5)
