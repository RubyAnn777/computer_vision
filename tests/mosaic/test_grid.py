import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from computer_vision.mosaic.grid import assemble, fit_to_grid, grid_shape, split_cells

TILE_W, TILE_H = 78, 90


def test_worked_example_from_the_docstring() -> None:
    assert grid_shape(4000, 3000, TILE_W, TILE_H, 80) == (52, 80)


def test_rounds_to_nearest_not_down() -> None:
    # exact ratio 10 * 10 * 59 / (10 * 100) = 5.9: the course notebook gives 5
    assert grid_shape(100, 59, 10, 10, 10) == (6, 10)


@given(
    image_w=st.integers(50, 5000),
    image_h=st.integers(50, 5000),
    n_cols=st.integers(5, 200),
)
def test_aspect_error_is_at_most_half_a_row(
    image_w: int, image_h: int, n_cols: int
) -> None:
    n_rows, _ = grid_shape(image_w, image_h, TILE_W, TILE_H, n_cols)
    exact_rows = n_cols * TILE_W * image_h / (TILE_H * image_w)
    assert abs(n_rows - exact_rows) <= 0.5 or n_rows == 1


def test_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError, match="positive"):
        grid_shape(100, 100, TILE_W, TILE_H, 0)


def test_fit_to_grid_gives_a_whole_number_of_cells() -> None:
    photo = np.zeros((300, 400, 3), dtype=np.uint8)
    canvas = fit_to_grid(photo, TILE_W, TILE_H, 10)
    n_rows, n_cols = grid_shape(400, 300, TILE_W, TILE_H, 10)
    assert canvas.shape == (n_rows * TILE_H, n_cols * TILE_W, 3)
    assert canvas.dtype == np.uint8


def test_split_cells_finds_each_cell() -> None:
    canvas = np.zeros((2 * TILE_H, 3 * TILE_W, 3), dtype=np.uint8)
    canvas[TILE_H:, 2 * TILE_W :] = 200  # bottom-right cell
    cells = split_cells(canvas, TILE_W, TILE_H)
    assert cells.shape == (2, 3, TILE_H, TILE_W, 3)
    assert (cells[1, 2] == 200).all()
    assert (cells[0, 0] == 0).all()


def test_split_cells_rejects_partial_cells() -> None:
    with pytest.raises(ValueError, match="whole number"):
        split_cells(np.zeros((100, 100, 3), dtype=np.uint8), TILE_W, TILE_H)


def test_assemble_is_the_inverse_of_split_cells() -> None:
    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 256, (3 * TILE_H, 4 * TILE_W, 3), dtype=np.uint8)
    cells = split_cells(canvas, TILE_W, TILE_H)
    tiles = cells.reshape(-1, TILE_H, TILE_W, 3)  # every cell as its own tile
    index_map = np.arange(12).reshape(3, 4)
    assert np.array_equal(assemble(tiles, index_map), canvas)


def test_assemble_places_the_chosen_tile() -> None:
    tiles = np.stack([np.full((TILE_H, TILE_W, 3), v, np.uint8) for v in (0, 255)])
    mosaic = assemble(tiles, np.array([[1, 0]]))
    assert mosaic.shape == (TILE_H, 2 * TILE_W, 3)
    assert (mosaic[:, :TILE_W] == 255).all()
    assert (mosaic[:, TILE_W:] == 0).all()
