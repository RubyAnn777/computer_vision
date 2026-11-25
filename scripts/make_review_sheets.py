"""Prepare the manual review of the automatic crops (D23).

Numbers every automatic crop (sorted by tile id) and writes

- ``data/processed/tiles/auto/review/sheet_NNN.png``: contact sheets of 60
  crops, each labelled only with its number. The confidence is not shown, so
  the reviewer's judgement is not anchored on the detector's score;
- ``data/tile_review.csv``: one row per crop, ``decision`` pre-filled with
  ``keep``. The reviewer changes ``decision`` to ``exclude`` and fills in
  ``rule`` (and ``note`` if useful) for every crop that should not become a
  tile. This file is committed: it is the record of manual filtering that the
  brief asks for, like ``page_screening.csv`` for pages.

Review rules (``rule`` column):

- ``T1`` not a product: text, price tag, logo, badge, empty background;
- ``T2`` product cut off so badly that it is not recognisable;
- ``T3`` several separate products in one crop;
- ``T4`` styled scene or background dominates the crop;
- ``T5`` other, explained in ``note``.

The review covers all crops at the recall-first threshold, so it serves both
versions of the automatic collection (D21). An existing ``tile_review.csv``
is never overwritten.

Usage::

    uv run python scripts/make_review_sheets.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from computer_vision.tiles.contact import contact_sheet

INDEX = Path("data/processed/tiles/auto/index.csv")
RAW = Path("data/processed/tiles/auto/raw")
SHEETS = Path("data/processed/tiles/auto/review")
REVIEW = Path("data/tile_review.csv")
PER_SHEET = 60
COLUMNS = 10


def main() -> None:
    """Write the numbered contact sheets and the review file."""
    if REVIEW.exists():
        raise SystemExit(f"{REVIEW} exists; it may hold your review, so it is kept")
    with INDEX.open(newline="", encoding="utf-8") as handle:
        tile_ids = sorted(r["tile_id"] for r in csv.DictReader(handle))
    SHEETS.mkdir(parents=True, exist_ok=True)

    rows = []
    for start in range(0, len(tile_ids), PER_SHEET):
        sheet_no = start // PER_SHEET + 1
        batch = tile_ids[start : start + PER_SHEET]
        images = []
        for name in batch:
            loaded = cv2.imread(str(RAW / f"{name}.png"), cv2.IMREAD_COLOR)
            if loaded is None:
                raise FileNotFoundError(f"cannot read crop {name}.png")
            images.append(np.asarray(loaded, dtype=np.uint8))
        numbers = [start + i + 1 for i in range(len(batch))]
        sheet = contact_sheet(images, [str(n) for n in numbers], columns=COLUMNS)
        cv2.imwrite(str(SHEETS / f"sheet_{sheet_no:03d}.png"), sheet)
        rows += [
            {"number": n, "tile_id": t, "sheet": sheet_no, "decision": "keep"}
            for n, t in zip(numbers, batch, strict=True)
        ]

    with REVIEW.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["number", "tile_id", "sheet", "decision", "rule", "note"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} crops on {-(-len(rows) // PER_SHEET)} sheets in {SHEETS}")
    print(f"review file: {REVIEW}")


if __name__ == "__main__":
    main()
