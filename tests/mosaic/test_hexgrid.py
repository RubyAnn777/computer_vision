import numpy as np
import pytest

from computer_vision.mosaic.hexgrid import (
    build_hex_mosaic,
    hex_distance,
    hex_labels,
    hex_mask,
    hex_no_repeat,
    make_layout,
    whole_hexagons,
)
from computer_vision.mosaic.library import TileLibrary
from computer_vision.mosaic.matching import distance_table

TILE_W, TILE_H = 26, 30  # same shape as 78 x 90, a third of the size


def solid_library(colours: list[tuple[int, int, int]]) -> TileLibrary:
    images = np.stack([np.full((TILE_H, TILE_W, 3), c, np.uint8) for c in colours])
    return TileLibrary("solid", tuple(f"t{i}" for i in range(len(colours))), images)


def test_hex_mask_keeps_three_quarters_and_drops_the_corners() -> None:
    mask = hex_mask(78, 90)
    assert mask.sum() / mask.size == pytest.approx(0.75, abs=0.01)
    assert mask[45, 39]
    assert mask[45, 0]
    assert mask[45, 77]
    assert not mask[0, 0]
    assert not mask[0, 77]
    assert not mask[89, 0]
    assert not mask[89, 77]
    assert np.array_equal(mask, mask[::-1])
    assert np.array_equal(mask, mask[:, ::-1])


def test_every_pixel_belongs_to_exactly_one_hexagon_of_the_right_size() -> None:
    layout = make_layout(10 * TILE_W, 300, TILE_W, TILE_H)
    ids, ty, tx = hex_labels(layout, 0, layout.height)
    assert ids.shape == (300, 10 * TILE_W)
    assert (ty >= 0).all()
    assert (ty < TILE_H).all()
    assert (tx >= 0).all()
    assert (tx < TILE_W).all()
    sizes = np.bincount(ids.ravel())
    whole = whole_hexagons(layout)[: len(sizes)]
    # whole hexagons cover as many pixels as the mask (up to rounding on edges)
    assert np.abs(sizes[whole] - hex_mask(TILE_W, TILE_H).sum()).max() <= TILE_W


def test_a_pixel_maps_into_its_hexagons_mask() -> None:
    layout = make_layout(6 * TILE_W, 200, TILE_W, TILE_H)
    _, ty, tx = hex_labels(layout, 0, layout.height)
    mask = hex_mask(TILE_W, TILE_H)
    assert mask[ty, tx].mean() > 0.98  # only pixels exactly on an edge may differ


def test_bands_give_the_same_labels_as_one_pass() -> None:
    layout = make_layout(4 * TILE_W, 120, TILE_W, TILE_H)
    full = hex_labels(layout, 0, 120)[0]
    parts = np.concatenate(
        [hex_labels(layout, 0, 50)[0], hex_labels(layout, 50, 120)[0]]
    )
    assert np.array_equal(full, parts)


@pytest.mark.parametrize("row", [4, 5])
def test_each_hexagon_has_six_neighbours(row: int) -> None:
    rows, cols = np.mgrid[row - 2 : row + 3, 2:9]
    d = hex_distance(row, 5, rows, cols)
    assert (d == 1).sum() == 6
    assert (d == 0).sum() == 1
    assert (d == 2).sum() == 12


@pytest.mark.parametrize("radius", [1, 2])
def test_no_tile_repeats_within_the_radius(radius: int) -> None:
    rng = np.random.default_rng(1)
    rows, cols = (a.ravel() for a in np.mgrid[0:8, 0:9])
    table = distance_table(
        rng.uniform(0, 255, (len(rows), 3)).astype(np.float32),
        rng.uniform(0, 255, (40, 3)).astype(np.float32),
    )
    chosen = hex_no_repeat(table, rows, cols, radius)
    d = hex_distance(rows[:, None], cols[:, None], rows[None, :], cols[None, :])
    close = (d <= radius) & (d > 0)
    assert not (close & (chosen[:, None] == chosen[None, :])).any()


def test_no_repeat_needs_enough_tiles() -> None:
    rows, cols = np.zeros(4, np.intp), np.arange(4)
    with pytest.raises(ValueError, match="at least"):
        hex_no_repeat(np.zeros((4, 3), np.float32), rows, cols, 2)


def test_one_colour_photo_selects_the_matching_tile() -> None:
    library = solid_library([(0, 0, 0), (40, 160, 220), (255, 255, 255)])
    photo = np.full((120, 160, 3), (40, 160, 220), np.uint8)
    result = build_hex_mosaic(photo, library, n_cols=6, repeat_radius=0)
    assert (result.chosen == 1).all()
    assert result.image.shape == result.canvas.shape
    assert np.array_equal(result.image, result.canvas)


def test_crop_covers_everything_and_whole_leaves_a_background_margin() -> None:
    library = solid_library([(v, v, v) for v in range(0, 256, 32)])
    photo = np.full((150, 156, 3), 64, np.uint8)
    crop = build_hex_mosaic(photo, library, n_cols=6, repeat_radius=0)
    whole = build_hex_mosaic(
        photo,
        library,
        n_cols=6,
        repeat_radius=0,
        boundary="whole",
        background=(0, 0, 255),
    )
    red = (whole.image == (0, 0, 255)).all(axis=2)
    assert not (crop.image == (0, 0, 255)).all(axis=2).any()
    assert crop.n_partial > 0
    assert whole.n_partial == 0
    assert red[0].any()
    assert red[:, 0].any()
    assert not red[red.shape[0] // 2, TILE_W:-TILE_W].any()


def test_masking_changes_what_a_tile_counts_as() -> None:
    # a tile that is red in its corners and grey in its hexagon
    tile = np.full((TILE_H, TILE_W, 3), 128, np.uint8)
    tile[~hex_mask(TILE_W, TILE_H)] = (0, 0, 255)
    plain = np.full((TILE_H, TILE_W, 3), (115, 120, 145), np.uint8)  # near grey
    library = TileLibrary("t", ("corners", "plain"), np.stack([tile, plain]))
    photo = np.full((90, 104, 3), 128, np.uint8)  # grey photo
    masked = build_hex_mosaic(photo, library, n_cols=4, repeat_radius=0)
    unmasked = build_hex_mosaic(
        photo, library, n_cols=4, repeat_radius=0, mask_tiles=False
    )
    assert (masked.chosen == 0).all()  # only the shown part counts: a grey tile
    assert (unmasked.chosen == 1).all()  # the hidden red corners mislead the match
