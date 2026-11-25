"""Turn raw crops into a standardised, deduplicated tile collection (D22, D23).

Steps, each logged per crop in ``filter_log.csv``:

1. **size**: drop crops whose fitted region (after `fitted_size`) is smaller
   than ``MIN_W`` x ``MIN_H``; they would need more than 1.5x enlargement.
2. **confidence** (automatic only): drop crops below ``--min-conf``.
   0.14 (the extraction threshold) keeps all; 0.58 gives the precision-first
   version (D21).
3. **review** (automatic only): drop crops marked ``exclude`` in
   ``data/tile_review.csv``, with the reviewer's rule (T1 to T5).
4. **standardise**: `fit_to_aspect` (crop to at least ``--min-keep`` of the
   area, pad with the border colour) and `resize_tile` to ``TILE_W`` x
   ``TILE_H``.
5. **dedupe**: drop a tile within Hamming distance ``MAX_HAMMING`` of the
   pHash of an earlier kept tile (sorted tile ids).

Writes into ``data/processed/tiles/collections/<name>/``:

- ``tiles/<tile_id>.png`` and ``tiles.csv`` (kept tiles with their kept area
  share and mean colour);
- ``filter_log.csv``: every raw crop with the step that removed it, or kept;
- ``summary.json``: parameters, counts per step and per catalog;
- ``dedupe_pairs/``: side-by-side examples at distances 4 to 10, to check the
  dedupe cut-off by eye (L06);
- ``contact_sheet.png``: 100 kept tiles in a seeded random order, for the
  report.

The fit strategy is a parameter, so both D23 variants can be built and
compared: ``--min-keep 0.7`` (crop to 70 %, then pad; the default) and
``--min-keep 0`` (pure centre crop). The folder name records it
(``keep070`` or ``keep000``).

Usage::

    uv run python scripts/build_tile_collection.py --collection manual
    uv run python scripts/build_tile_collection.py --collection manual --min-keep 0
    uv run python scripts/build_tile_collection.py --collection auto --min-conf 0.58
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from computer_vision.tiles.contact import contact_sheet
from computer_vision.tiles.dedupe import greedy_unique, hamming, phash
from computer_vision.tiles.standardise import (
    TILE_ASPECT,
    fit_to_aspect,
    fitted_size,
    kept_share,
    resize_tile,
)

TILES = Path("data/processed/tiles")
REVIEW = Path("data/tile_review.csv")
TILE_W, TILE_H = 78, 90  # master size, W/H = 0.867 ~ sqrt(3)/2 (D22, D23)
MIN_W, MIN_H = 52, 60  # smallest fitted region kept: at most 1.5x enlargement
MAX_HAMMING = 6  # pHash bits that may differ for two tiles to count as copies
SEED = 2026
PAIR_DISTANCES = range(4, 11)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load(path: Path) -> np.ndarray:
    """Read a BGR image or fail loudly."""
    loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if loaded is None:
        raise FileNotFoundError(f"cannot read {path}")
    return np.asarray(loaded, dtype=np.uint8)


def main() -> None:
    """Filter, standardise and dedupe one collection."""
    parser = argparse.ArgumentParser(description="Build one tile collection.")
    parser.add_argument("--collection", choices=["manual", "auto"], required=True)
    parser.add_argument("--min-conf", dest="min_confidence", type=float, default=None)
    parser.add_argument(
        "--min-keep",
        type=float,
        default=0.7,
        help="share of each crop's area kept before padding; 0 = pure centre crop",
    )
    args = parser.parse_args()
    if (args.collection == "auto") != (args.min_confidence is not None):
        raise SystemExit("--min-conf is required for auto and not used for manual")

    fit = f"keep{round(args.min_keep * 100):03d}"  # keep000 = pure centre crop
    name = f"{args.collection}_{fit}_w{TILE_W}h{TILE_H}"
    if args.collection == "auto":
        name = f"auto_c{args.min_confidence:.2f}_{fit}_w{TILE_W}h{TILE_H}"
    out = TILES / "collections" / name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"{out} exists; delete it to rebuild (stale tiles would remain)"
        )
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    (out / "dedupe_pairs").mkdir(exist_ok=True)
    raw = TILES / args.collection / "raw"
    rows = sorted(
        read_csv(TILES / args.collection / "index.csv"), key=lambda r: r["tile_id"]
    )

    excluded: dict[str, str] = {}
    if args.collection == "auto":
        if not REVIEW.exists():
            raise SystemExit(
                f"{REVIEW} missing; run make_review_sheets.py and review first"
            )
        excluded = {
            r["tile_id"]: r["rule"] or "T5"
            for r in read_csv(REVIEW)
            if r["decision"].strip() == "exclude"
        }

    log: dict[str, dict[str, object]] = {}
    survivors: list[dict[str, str]] = []
    for r in rows:
        w, h = int(r["width"]), int(r["height"])
        fw, fh = fitted_size(w, h, TILE_ASPECT, args.min_keep)
        entry: dict[str, object] = {
            "tile_id": r["tile_id"],
            "step": "kept",
            "reason": "",
        }
        if fw < MIN_W or fh < MIN_H:
            entry |= {"step": "size", "reason": f"fitted {fw}x{fh}"}
        elif (
            args.min_confidence is not None
            and float(r["confidence"]) < args.min_confidence
        ):
            entry |= {"step": "confidence", "reason": r["confidence"]}
        elif r["tile_id"] in excluded:
            entry |= {"step": "review", "reason": excluded[r["tile_id"]]}
        else:
            survivors.append(r)
        log[r["tile_id"]] = entry

    tiles = {
        r["tile_id"]: resize_tile(
            fit_to_aspect(
                load(raw / f"{r['tile_id']}.png"), TILE_ASPECT, args.min_keep
            ),
            TILE_W,
            TILE_H,
        )
        for r in survivors
    }
    names = [r["tile_id"] for r in survivors]
    hashes = [phash(tiles[n]) for n in names]
    kept_idx, dropped = greedy_unique(hashes, MAX_HAMMING)
    for i, k in dropped.items():
        log[names[i]] |= {"step": "dedupe", "reason": f"copy of {names[k]}"}

    # examples near the cut-off: one pair per distance, first found in order
    shown: set[int] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = hamming(hashes[i], hashes[j])
            if d in PAIR_DISTANCES and d not in shown:
                pair = np.hstack([tiles[names[i]], tiles[names[j]]])
                cv2.imwrite(str(out / "dedupe_pairs" / f"d{d:02d}.png"), pair)
                shown.add(d)
        if len(shown) == len(PAIR_DISTANCES):
            break

    by_id = {r["tile_id"]: r for r in rows}
    tile_rows = []
    for i in kept_idx:
        n = names[i]
        cv2.imwrite(str(out / "tiles" / f"{n}.png"), tiles[n])
        b, g, rr = tiles[n].reshape(-1, 3).mean(axis=0)
        src = by_id[n]
        tile_rows.append(
            {
                "tile_id": n,
                "flyer_slug": src["flyer_slug"],
                "kept_share": round(
                    kept_share(
                        int(src["width"]),
                        int(src["height"]),
                        TILE_ASPECT,
                        args.min_keep,
                    ),
                    3,
                ),
                "mean_b": round(float(b), 2),
                "mean_g": round(float(g), 2),
                "mean_r": round(float(rr), 2),
                "phash": f"{hashes[i]:016x}",
            }
        )

    with (out / "tiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(tile_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(tile_rows)
    with (out / "filter_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["tile_id", "step", "reason"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(log.values())

    sample = [names[i] for i in kept_idx]
    random.Random(SEED).shuffle(sample)
    sample = sorted(sample[:100])
    sheet = contact_sheet(
        [tiles[n] for n in sample], [str(i + 1) for i in range(len(sample))], cell=100
    )
    cv2.imwrite(str(out / "contact_sheet.png"), sheet)

    steps = Counter(str(e["step"]) for e in log.values())
    review_rules = Counter(
        str(e["reason"]) for e in log.values() if e["step"] == "review"
    )
    summary = {
        "collection": args.collection,
        "min_confidence": args.min_confidence,
        "tile_size": [TILE_W, TILE_H],
        "aspect": TILE_ASPECT,
        "min_keep": args.min_keep,
        "min_fitted_size": [MIN_W, MIN_H],
        "max_hamming": MAX_HAMMING,
        "raw_crops": len(rows),
        "removed": {s: steps[s] for s in ("size", "confidence", "review", "dedupe")},
        "review_rules": dict(sorted(review_rules.items())),
        "tiles": len(tile_rows),
        "tiles_per_flyer": dict(
            sorted(Counter(t["flyer_slug"] for t in tile_rows).items())
        ),
        "median_kept_share": float(np.median([t["kept_share"] for t in tile_rows])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
