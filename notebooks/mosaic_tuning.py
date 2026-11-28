import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Mosaic tuning")


@app.cell
def _():
    from functools import lru_cache
    from pathlib import Path

    import cv2
    import marimo as mo
    import numpy as np
    from skimage import data as skdata

    from computer_vision.mosaic.descriptors import DESCRIPTORS, describe_in_batches
    from computer_vision.mosaic.engine import build_mosaic
    from computer_vision.mosaic.grid import fit_to_grid, split_cells
    from computer_vision.mosaic.library import load_library
    from computer_vision.mosaic.matching import distance_table

    return (
        DESCRIPTORS,
        Path,
        build_mosaic,
        cv2,
        describe_in_batches,
        distance_table,
        fit_to_grid,
        load_library,
        lru_cache,
        mo,
        np,
        skdata,
        split_cells,
    )


@app.cell
def _(mo):
    mo.md("""
    # Mosaic tuning: compare settings side by side

    Pick a photo, then two settings to vary: one down the rows and one
    across the columns of the grid below. All other settings stay at the
    values in the right-hand panel. Every mosaic is shown small (roughly
    how it looks from across a room) and, if ticked, as a close-up of its
    centre (how it looks from arm's length). Under each one: SSIM (1 =
    same structure as the photo) and Delta E (colour error, about 2 is
    just noticeable), both measured at the viewing distance of
    `quality.py`, plus the uses of the most-used tile.

    Your own photos go in `data/raw/photos/` (JPEG or PNG; ignored by
    git). The scikit-image astronaut, coffee and chelsea photos are always
    there for comparison. The first run of a grid takes a few seconds per
    mosaic; repeats come from a cache.
    """)
    return


@app.cell
def _(Path):
    COLLECTIONS_DIR = Path("data/processed/tiles/collections")
    PHOTO_DIR = Path("data/raw/photos")
    PREVIEW_W = 360  # pixels; the "across the room" view
    ZOOM_CELLS = (8, 6)  # columns x rows of tiles in the close-up
    return COLLECTIONS_DIR, PHOTO_DIR, PREVIEW_W, ZOOM_CELLS


@app.cell
def _(COLLECTIONS_DIR, PHOTO_DIR, cv2, skdata):
    collections = sorted(
        p.name for p in COLLECTIONS_DIR.iterdir() if (p / "tiles.csv").exists()
    )
    photos = {
        "astronaut (scikit-image)": cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR),
        "coffee (scikit-image)": cv2.cvtColor(skdata.coffee(), cv2.COLOR_RGB2BGR),
        "chelsea (scikit-image)": cv2.cvtColor(skdata.chelsea(), cv2.COLOR_RGB2BGR),
    }
    unreadable = []
    if PHOTO_DIR.exists():
        for path in sorted(PHOTO_DIR.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image = cv2.imread(str(path))
                if image is None:
                    unreadable.append(path.name)
                else:
                    photos[path.name] = image
    return collections, photos, unreadable


@app.cell
def _(collections):
    # the values each setting takes when it is chosen as a grid axis
    AXES = {
        "collection": collections,
        "n_cols": [40, 60, 90, 120],
        "descriptor": ["mean_bgr", "mean_lab", "subblock_lab"],
        "repeat_radius": [0, 1, 2],
        "reuse_penalty": [0.0, 0.1, 0.3, 1.0],
        "tint": [0.0, 0.2, 0.4],
    }
    return (AXES,)


@app.cell
def _(AXES, collections, mo, photos, unreadable):
    photo_pick = mo.ui.dropdown(list(photos), value=list(photos)[-1], label="Photo")
    rows_axis = mo.ui.dropdown(list(AXES), value="collection", label="Rows vary")
    cols_axis = mo.ui.dropdown(list(AXES), value="reuse_penalty", label="Columns vary")
    zoom = mo.ui.checkbox(value=True, label="Show close-ups")
    fixed = mo.ui.dictionary(
        {
            "collection": mo.ui.dropdown(collections, value=collections[0]),
            "n_cols": mo.ui.dropdown([str(v) for v in AXES["n_cols"]], value="90"),
            "descriptor": mo.ui.dropdown(AXES["descriptor"], value="mean_lab"),
            "repeat_radius": mo.ui.dropdown(
                [str(v) for v in AXES["repeat_radius"]], value="2"
            ),
            "reuse_penalty": mo.ui.number(0, 3, step=0.05, value=0.0),
            "tint": mo.ui.number(0, 1, step=0.05, value=0.0),
        },
        label="Other settings",
    )
    warning = (
        mo.md(f"**Could not read:** {', '.join(unreadable)} (HEIC? export as JPEG)")
        if unreadable
        else mo.md("")
    )
    mo.hstack(
        [mo.vstack([photo_pick, rows_axis, cols_axis, zoom, warning]), fixed],
        justify="start",
        gap=4,
    )
    return cols_axis, fixed, photo_pick, rows_axis, zoom


@app.cell
def _(
    COLLECTIONS_DIR,
    PREVIEW_W,
    ZOOM_CELLS,
    build_mosaic,
    cv2,
    load_library,
    lru_cache,
    photos,
):
    @lru_cache(maxsize=8)
    def library(name):
        return load_library(COLLECTIONS_DIR / name)

    def png(image):
        return cv2.imencode(".png", image)[1].tobytes()

    @lru_cache(maxsize=128)
    def render(
        photo, collection, n_cols, descriptor, repeat_radius, reuse_penalty, tint
    ):
        """Build one mosaic; keep only the two views and the numbers."""
        tiles = library(collection)
        result = build_mosaic(
            photos[photo],
            tiles,
            n_cols=n_cols,
            descriptor=descriptor,
            repeat_radius=repeat_radius,
            reuse_penalty=reuse_penalty,
            tint=tint,
        )
        image = result.image
        height, width = image.shape[:2]
        small = cv2.resize(
            image,
            (PREVIEW_W, round(PREVIEW_W * height / width)),
            interpolation=cv2.INTER_AREA,
        )
        tile_w, tile_h = tiles.tile_size
        zw, zh = ZOOM_CELLS[0] * tile_w, ZOOM_CELLS[1] * tile_h
        top = max(0, (height - zh) // 2 // tile_h * tile_h)
        left = max(0, (width - zw) // 2 // tile_w * tile_w)
        close = image[top : top + zh, left : left + zw]
        close = cv2.resize(
            close, (PREVIEW_W, round(PREVIEW_W * close.shape[0] / close.shape[1]))
        )
        return {
            "small": png(small),
            "close": png(close),
            "grid": result.grid,
            "ssim": result.scores.ssim,
            "delta_e": result.scores.delta_e,
            "distinct": result.usage.distinct,
            "max_uses": result.usage.max_uses,
        }

    return library, render


@app.cell
def _(AXES, cols_axis, fixed, mo, photo_pick, render, rows_axis, zoom):
    base = {
        "collection": fixed.value["collection"],
        "n_cols": int(fixed.value["n_cols"]),
        "descriptor": fixed.value["descriptor"],
        "repeat_radius": int(fixed.value["repeat_radius"]),
        "reuse_penalty": float(fixed.value["reuse_penalty"]),
        "tint": float(fixed.value["tint"]),
    }
    row_key, col_key = rows_axis.value, cols_axis.value
    row_values = AXES[row_key] if row_key != col_key else [base[row_key]]
    table = []
    grid_rows = []
    with mo.status.progress_bar(
        total=len(row_values) * len(AXES[col_key]), title="Building mosaics"
    ) as bar:
        for row_value in row_values:
            cells = []
            for col_value in AXES[col_key]:
                params = base | {row_key: row_value, col_key: col_value}
                out = render(photo_pick.value, **params)
                bar.update()
                caption = (
                    f"**{row_key} = {row_value}**, **{col_key} = {col_value}**  \n"
                    f"SSIM {out['ssim']:.2f} · Delta E {out['delta_e']:.1f} · "
                    f"top tile {out['max_uses']}x · {out['distinct']} tiles"
                )
                views = [mo.image(out["small"])]
                if zoom.value:
                    views.append(mo.image(out["close"]))
                cells.append(mo.vstack([mo.md(caption), *views]))
                table.append(
                    params
                    | {
                        "grid": f"{out['grid'][0]} x {out['grid'][1]}",
                        "ssim": round(out["ssim"], 3),
                        "delta_e": round(out["delta_e"], 2),
                        "max_uses": out["max_uses"],
                        "distinct": out["distinct"],
                    }
                )
            grid_rows.append(
                mo.hstack(cells, justify="start", align="start", wrap=True)
            )
    mo.vstack(grid_rows, gap=2)
    return (table,)


@app.cell
def _(mo, photo_pick, photos):
    import cv2 as _cv2

    _photo = photos[photo_pick.value]
    _h, _w = _photo.shape[:2]
    _small = _cv2.resize(
        _photo, (360, round(360 * _h / _w)), interpolation=_cv2.INTER_AREA
    )
    mo.vstack(
        [
            mo.md("**The photo, at the same size**"),
            mo.image(_cv2.imencode(".png", _small)[1].tobytes()),
        ]
    )
    return


@app.cell
def _(mo, table):
    mo.vstack(
        [mo.md("**All numbers in this grid**"), mo.ui.table(table, selection=None)]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Where can the library match at all?

    For every cell: the colour distance to the closest tile in the whole
    collection, before any repeat rule (Delta E 1976 for the CIELAB
    descriptors). Bright areas have no tile of the right colour, so no
    setting can fix them; that is a limit of the tile library, not of the
    matcher. Uses the photo, collection, columns and descriptor chosen
    above.
    """)
    return


@app.cell
def _(
    DESCRIPTORS,
    PREVIEW_W,
    cv2,
    describe_in_batches,
    distance_table,
    fit_to_grid,
    fixed,
    library,
    mo,
    np,
    photo_pick,
    photos,
    split_cells,
):
    _tiles = library(fixed.value["collection"])
    _describe = DESCRIPTORS[fixed.value["descriptor"]]
    _tw, _th = _tiles.tile_size
    _canvas = fit_to_grid(
        photos[photo_pick.value], _tw, _th, int(fixed.value["n_cols"])
    )
    _cells = split_cells(_canvas, _tw, _th)
    _n_rows, _n_cols = _cells.shape[:2]
    _best = np.sqrt(
        distance_table(
            describe_in_batches(_cells.reshape(-1, _th, _tw, 3), _describe),
            describe_in_batches(_tiles.images, _describe),
        ).min(axis=1)
    ).reshape(_n_rows, _n_cols)
    _scale = 20.0  # distances at or above this are drawn at full brightness
    _grey = np.clip(_best / _scale * 255, 0, 255).astype(np.uint8)
    _heat = cv2.applyColorMap(_grey, cv2.COLORMAP_INFERNO)
    _heat = cv2.resize(
        _heat,
        (PREVIEW_W, round(PREVIEW_W * _n_rows * _th / (_n_cols * _tw))),
        interpolation=cv2.INTER_NEAREST,
    )
    mo.vstack(
        [
            mo.image(cv2.imencode(".png", _heat)[1].tobytes()),
            mo.md(
                f"Black = 0, bright yellow = {_scale:.0f} or more. "
                f"Median best distance {np.median(_best):.1f}, "
                f"90th percentile {np.percentile(_best, 90):.1f}, "
                f"share of cells above 10: {(_best > 10).mean():.0%}."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
