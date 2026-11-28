import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from computer_vision.mosaic.matching import (
    distance_table,
    nearest,
    nearest_no_repeat,
    usage_stats,
)

vectors = arrays(
    np.float32, st.tuples(st.integers(1, 6), st.just(3)), elements=st.floats(0, 255)
)


@given(cells=vectors, tiles=vectors)
def test_distance_table_matches_the_definition(
    cells: np.ndarray, tiles: np.ndarray
) -> None:
    direct = ((cells[:, None, :] - tiles[None, :, :]) ** 2).sum(axis=2)
    assert np.allclose(distance_table(cells, tiles), direct, rtol=1e-4, atol=1e-2)


@given(v=vectors)
def test_distance_is_symmetric_and_zero_on_itself(v: np.ndarray) -> None:
    table = distance_table(v, v)
    assert np.allclose(table, table.T, atol=1e-2)
    assert np.allclose(np.diag(table), 0, atol=1e-2)
    assert (table >= 0).all()


def test_nearest_picks_the_closest_tile() -> None:
    tiles = np.array([[0, 0, 0], [100, 100, 100], [255, 255, 255]], np.float32)
    cells = np.array([[90, 110, 100], [250, 250, 255]], np.float32)
    assert nearest(distance_table(cells, tiles)).tolist() == [1, 2]


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_no_tile_repeats_within_the_radius(radius: int) -> None:
    rng = np.random.default_rng(2)
    n_rows, n_cols = 9, 11
    table = distance_table(
        rng.uniform(0, 255, (n_rows * n_cols, 3)).astype(np.float32),
        rng.uniform(0, 255, (60, 3)).astype(np.float32),
    )
    grid = nearest_no_repeat(table, n_cols, radius).reshape(n_rows, n_cols)
    # every (radius + 1) x (radius + 1) square holds distinct tiles, and so does
    # every pair of cells at most `radius` apart: check all such windows
    for r in range(n_rows):
        for c in range(n_cols):
            window = grid[r : r + radius + 1, max(0, c - radius) : c + radius + 1]
            for (r2, c2), k in np.ndenumerate(window):
                rr, cc = r + r2, max(0, c - radius) + c2
                if (rr, cc) != (r, c) and max(abs(rr - r), abs(cc - c)) <= radius:
                    assert k != grid[r, c]


def flat_table(n_cells: int = 36, n_tiles: int = 40) -> np.ndarray:
    # every cell has the same colour; tile k is k units away from it
    tiles = np.arange(n_tiles, dtype=np.float32)[:, None].repeat(3, axis=1) / np.sqrt(3)
    return distance_table(np.zeros((n_cells, 3), np.float32), tiles)


def test_local_rule_alone_gives_a_regular_pattern_on_flat_regions() -> None:
    table = flat_table()
    assert set(nearest(table).tolist()) == {0}  # pure nearest: one tile
    grid = nearest_no_repeat(table, 6, radius=1).reshape(6, 6)
    assert set(grid.ravel().tolist()) == {0, 1, 2, 3}  # the 2 x 2 checkerboard


def test_reuse_penalty_spreads_flat_regions_over_more_tiles() -> None:
    table = flat_table()
    plain = nearest_no_repeat(table, 6, radius=1)
    spread = nearest_no_repeat(table, 6, radius=1, reuse_penalty=1.0)
    assert len(set(spread.tolist())) > len(set(plain.tolist()))
    assert np.bincount(spread).max() < np.bincount(plain).max()


def test_no_repeat_needs_enough_tiles() -> None:
    with pytest.raises(ValueError, match="at least 13 tiles"):
        nearest_no_repeat(np.zeros((4, 12), np.float32), 2, radius=2)


def test_radius_must_be_positive() -> None:
    with pytest.raises(ValueError, match="radius"):
        nearest_no_repeat(np.zeros((4, 20), np.float32), 2, radius=0)
    with pytest.raises(ValueError, match="reuse_penalty"):
        nearest_no_repeat(np.zeros((4, 20), np.float32), 2, reuse_penalty=-1)


def test_usage_stats() -> None:
    table = np.array([[0, 4], [9, 1], [0, 16]], np.float32)
    stats = usage_stats(table, np.array([0, 1, 0]))
    assert stats.distinct == 2
    assert stats.max_uses == 2
    assert stats.mean_distance == pytest.approx(1 / 3)
