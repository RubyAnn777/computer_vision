import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Hexagonal mosaic decisions")


@app.cell
def _():
    from functools import lru_cache
    from pathlib import Path

    import cv2
    import marimo as mo
    import numpy as np
    from skimage import data as skdata

    from computer_vision.mosaic.engine import build_mosaic
    from computer_vision.mosaic.hexgrid import build_hex_mosaic, hex_mask
    from computer_vision.mosaic.library import load_library

    def data_url(image):
        """Encode a BGR image as an inline PNG, so marimo keeps no temp file."""
        import base64

        _, buf = cv2.imencode(".png", image)
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

    return (
        Path,
        build_hex_mosaic,
        build_mosaic,
        cv2,
        data_url,
        hex_mask,
        load_library,
        lru_cache,
        mo,
        np,
        skdata,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Hexagonal mosaics: compare the decisions side by side

    Task 2 asks you to explain four things: how the tiles are **positioned**,
    how they are **masked**, how **colour similarity** is computed inside a
    hexagon, and how the **boundaries** of the image are handled. Each of them
    is a switch below. Pick a photo and two switches to vary, one down the
    rows and one across the columns; the rest stay at the values on the
    right. Every mosaic is shown small (from across a room), as a close-up of
    its centre, and as a close-up of its top-left corner, where the boundary
    decision shows.

    Your photos go in `data/raw/photos/` (ignored by git). Each hexagonal
    mosaic takes a few seconds at 60 columns and about ten at 90; repeats
    come from a cache.
    """)
    return


@app.cell
def _(Path):
    COLLECTIONS_DIR = Path("data/processed/tiles/collections")
    PHOTO_DIR = Path("data/raw/photos")
    PREVIEW_W = 360  # pixels; the "across the room" view
    ZOOM_TILES = (6, 5)  # tile widths x tile heights in each close-up
    return COLLECTIONS_DIR, PHOTO_DIR, PREVIEW_W, ZOOM_TILES


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
    if PHOTO_DIR.exists():
        for path in sorted(PHOTO_DIR.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image = cv2.imread(str(path))
                if image is not None:
                    photos[path.name] = image
    return collections, photos


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Positioning and masking: why pointy-top hexagons

    A regular hexagon with its point up fits a box $\sqrt{3}R$ wide and $2R$
    high, so width / height $= \sqrt{3}/2 = 0.866$. Our tiles are 78 × 90
    ($78/90 = 0.867$, chosen for this in D22), so each tile *is* the box of
    one hexagon. The mask keeps the hexagon and drops the four corner
    triangles: 75 % of each product image is shown and nothing is stretched.
    A flat-top hexagon is wider than high; inside our portrait tiles it would
    show only about 56 %. In the honeycomb the centres are one tile width
    apart along a row, rows are $3H/4$ = 67.5 pixels apart, and every other
    row is shifted by half a tile.
    """)
    return


@app.cell
def _(collections, mo):
    figure_collection = mo.ui.dropdown(
        collections, value=collections[0], label="Collection"
    )
    figure_tile = mo.ui.number(0, 5000, step=1, value=0, label="Tile number")
    mo.hstack([figure_collection, figure_tile], justify="start")
    return figure_collection, figure_tile


@app.cell
def _(
    cv2,
    data_url,
    figure_collection,
    figure_tile,
    hex_mask,
    library,
    mo,
    np,
):
    _tiles = library(figure_collection.value)
    _tile = _tiles.images[int(figure_tile.value) % len(_tiles)]
    _h, _w = _tile.shape[:2]
    _pointy = hex_mask(_w, _h)
    # flat-top hexagon as wide as the tile: height sqrt(3)/2 * width
    _hh = np.sqrt(3) / 2 * _w
    _y0 = (_h - _hh) / 2
    _flat_pts = np.array(
        [
            (0, _h / 2),
            (_w / 4, _y0),
            (3 * _w / 4, _y0),
            (_w, _h / 2),
            (3 * _w / 4, _y0 + _hh),
            (_w / 4, _y0 + _hh),
        ]
    )
    _flat = np.zeros((_h, _w), np.uint8)
    cv2.fillPoly(_flat, [np.round(_flat_pts).astype(np.int32)], 1)
    _flat = _flat.astype(bool)

    def _show(image, scale=3):
        big = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        return mo.image(data_url(big))

    def _masked(mask):
        out = _tile.copy()
        out[~mask] = 255
        return out

    mo.hstack(
        [
            mo.vstack([mo.md("**Tile**"), _show(_tile)]),
            mo.vstack(
                [mo.md("**Mask (pointy-top)**"), _show(_pointy.astype(np.uint8) * 255)]
            ),
            mo.vstack(
                [
                    mo.md(f"**Pointy-top: {_pointy.mean():.0%} shown**"),
                    _show(_masked(_pointy)),
                ]
            ),
            mo.vstack(
                [
                    mo.md(f"**Flat-top: {_flat.mean():.0%} shown**"),
                    _show(_masked(_flat)),
                ]
            ),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Compare the decisions

    - **shape**: rectangle (Task 1, same columns) vs hexagon, for a direct
      comparison.
    - **mask_tiles** (colour similarity): describe a tile by the mean colour
      of the pixels *inside its hexagon* (what is shown) or of its whole box
      (corners included, though they are cut away). The photo side always
      uses the pixels inside each hexagon.
    - **boundary**: *crop* covers the photo with hexagons that reach past its
      edges and crops the result (edge hexagons matched on their visible
      part, as in the lecture); *whole* keeps only hexagons fully inside the
      photo and leaves a white zigzag margin.
    - **repeat_radius**: no tile twice within this many hexagon steps (each
      hexagon has six neighbours); 0 is pure nearest neighbour.
    - **descriptor**, **n_cols**, **collection**: as in Task 1.
    """)
    return


@app.cell
def _(collections):
    AXES = {
        "shape": ["rectangle", "hexagon"],
        "mask_tiles": [True, False],
        "boundary": ["crop", "whole"],
        "repeat_radius": [0, 1, 2],
        "descriptor": ["mean_bgr", "mean_lab"],
        "n_cols": [40, 60, 90],
        "collection": collections,
    }
    return (AXES,)


@app.cell
def _(AXES, collections, mo, photos):
    photo_pick = mo.ui.dropdown(list(photos), value=list(photos)[-1], label="Photo")
    rows_axis = mo.ui.dropdown(list(AXES), value="boundary", label="Rows vary")
    cols_axis = mo.ui.dropdown(list(AXES), value="mask_tiles", label="Columns vary")
    zoom = mo.ui.checkbox(value=True, label="Show close-ups")
    fixed = mo.ui.dictionary(
        {
            "shape": mo.ui.dropdown(AXES["shape"], value="hexagon"),
            "mask_tiles": mo.ui.dropdown(["True", "False"], value="True"),
            "boundary": mo.ui.dropdown(AXES["boundary"], value="crop"),
            "repeat_radius": mo.ui.dropdown(["0", "1", "2"], value="2"),
            "descriptor": mo.ui.dropdown(AXES["descriptor"], value="mean_lab"),
            "n_cols": mo.ui.dropdown(["40", "60", "90"], value="90"),
            "collection": mo.ui.dropdown(collections, value=collections[0]),
        },
        label="Other settings",
    )
    mo.hstack(
        [mo.vstack([photo_pick, rows_axis, cols_axis, zoom]), fixed],
        justify="start",
        gap=4,
    )
    return cols_axis, fixed, photo_pick, rows_axis, zoom


@app.cell
def _(
    COLLECTIONS_DIR,
    PREVIEW_W,
    ZOOM_TILES,
    build_hex_mosaic,
    build_mosaic,
    cv2,
    data_url,
    load_library,
    lru_cache,
    photos,
):
    @lru_cache(maxsize=8)
    def library(name):
        return load_library(COLLECTIONS_DIR / name)

    def png(image):
        return data_url(image)

    def fit(image):
        h, w = image.shape[:2]
        return cv2.resize(
            image, (PREVIEW_W, round(PREVIEW_W * h / w)), interpolation=cv2.INTER_AREA
        )

    @lru_cache(maxsize=96)
    def render(
        photo,
        shape,
        mask_tiles,
        boundary,
        repeat_radius,
        descriptor,
        n_cols,
        collection,
    ):
        """Build one mosaic; keep only the three views and the numbers."""
        tiles = library(collection)
        if shape == "rectangle":
            result = build_mosaic(
                photos[photo],
                tiles,
                n_cols=n_cols,
                descriptor=descriptor,
                repeat_radius=repeat_radius,
                reuse_penalty=0.0,
            )
            partial, placed = 0, result.index_map.size
        else:
            result = build_hex_mosaic(
                photos[photo],
                tiles,
                n_cols=n_cols,
                descriptor=descriptor,
                repeat_radius=repeat_radius,
                mask_tiles=mask_tiles,
                boundary=boundary,
            )
            partial, placed = result.n_partial, len(result.chosen)
        image = result.image
        height, width = image.shape[:2]
        tile_w, tile_h = tiles.tile_size
        zw, zh = ZOOM_TILES[0] * tile_w, ZOOM_TILES[1] * tile_h
        top, left = max(0, (height - zh) // 2), max(0, (width - zw) // 2)
        return {
            "small": png(fit(image)),
            "close": png(fit(image[top : top + zh, left : left + zw])),
            "edge": png(fit(image[:zh, :zw])),
            "ssim": result.scores.ssim,
            "delta_e": result.scores.delta_e,
            "distinct": result.usage.distinct,
            "max_uses": result.usage.max_uses,
            "placed": placed,
            "partial": partial,
        }

    return library, render


@app.cell
def _(AXES, cols_axis, fixed, mo, photo_pick, render, rows_axis, zoom):
    base = {
        "shape": fixed.value["shape"],
        "mask_tiles": fixed.value["mask_tiles"] == "True",
        "boundary": fixed.value["boundary"],
        "repeat_radius": int(fixed.value["repeat_radius"]),
        "descriptor": fixed.value["descriptor"],
        "n_cols": int(fixed.value["n_cols"]),
        "collection": fixed.value["collection"],
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
                    views += [
                        mo.md("centre"),
                        mo.image(out["close"]),
                        mo.md("top-left corner"),
                        mo.image(out["edge"]),
                    ]
                cells.append(mo.vstack([mo.md(caption), *views]))
                table.append(
                    params
                    | {
                        "ssim": round(out["ssim"], 3),
                        "delta_e": round(out["delta_e"], 2),
                        "max_uses": out["max_uses"],
                        "distinct": out["distinct"],
                        "tiles placed": out["placed"],
                        "cut at edge": out["partial"],
                    }
                )
            grid_rows.append(
                mo.hstack(cells, justify="start", align="start", wrap=True)
            )
    mo.vstack(grid_rows, gap=2)
    return (table,)


@app.cell
def _(mo, table):
    mo.vstack(
        [mo.md("**All numbers in this grid**"), mo.ui.table(table, selection=None)]
    )
    return


if __name__ == "__main__":
    app.run()
