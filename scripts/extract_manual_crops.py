"""Cut the manual tile collection out of the hand-drawn boxes (D20).

For every Müller page in ``data/splits/split.csv`` (train, val and test), reads
its pass-1 box file, numbers the boxes in reading order and saves each box as
a PNG crop of the page image the box was drawn on. Writes

- ``data/processed/tiles/manual/raw/<tile_id>.png``: one crop per box, native
  pixels, lossless;
- ``data/processed/tiles/manual/index.csv``: one row per crop with its page,
  split, pixel box, size and the SHA-256 of the page image.

The supermarket pages (split ``unseen``) are left out: the collection is the
Müller collection. Cropping test pages does not touch the detector's
evaluation, so it leaks nothing. These are raw crops: filtering, resizing and
deduplication come later and never change this folder.

Usage::

    uv run python scripts/extract_manual_crops.py
    uv run python scripts/extract_manual_crops.py --overwrite   # rebuild
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from computer_vision.data.ids import parse_page_id
from computer_vision.data.yolo import read_boxes
from computer_vision.detection.run_info import sha256_of
from computer_vision.tiles.crops import crop, reading_order, tile_id

SPLIT = Path("data/splits/split.csv")
IMAGES = Path("data/interim/pages/annotate")
BOXES = Path("data/annotations/pass1")
OUT = Path("data/processed/tiles/manual")
RETAILER = "mueller"
FIELDS = [
    "tile_id",
    "page_id",
    "flyer_slug",
    "split",
    "index",
    "x0",
    "y0",
    "x1",
    "y1",
    "width",
    "height",
    "page_sha256",
]


def main() -> None:
    """Crop every hand-drawn box on the Müller pages and write the index."""
    parser = argparse.ArgumentParser(description="Build the raw manual crops.")
    parser.add_argument(
        "--overwrite", action="store_true", help="write into a non-empty folder"
    )
    args = parser.parse_args()

    raw = OUT / "raw"
    if raw.exists() and any(raw.iterdir()) and not args.overwrite:
        raise SystemExit(f"{raw} is not empty; pass --overwrite to rebuild it")
    raw.mkdir(parents=True, exist_ok=True)

    with SPLIT.open(newline="", encoding="utf-8") as handle:
        pages = sorted(
            (r["page_id"], r["split"])
            for r in csv.DictReader(handle)
            if parse_page_id(r["page_id"]).retailer == RETAILER
        )

    rows: list[dict[str, object]] = []
    for page_id, split in pages:
        image_path = IMAGES / f"{page_id}.png"
        loaded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if loaded is None:
            raise FileNotFoundError(f"cannot read page image {image_path}")
        image = np.asarray(loaded, dtype=np.uint8)
        page_sha = sha256_of(image_path)
        boxes = read_boxes(BOXES / f"{page_id}.txt")
        for index, i in enumerate(reading_order(boxes)):
            box = boxes[i]
            name = tile_id(page_id, "b", index)
            if not cv2.imwrite(str(raw / f"{name}.png"), crop(image, box)):
                raise OSError(f"could not write {name}.png")
            x0, y0, x1, y1 = box
            rows.append(
                {
                    "tile_id": name,
                    "page_id": page_id,
                    "flyer_slug": parse_page_id(page_id).catalog,
                    "split": split,
                    "index": index,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "page_sha256": page_sha,
                }
            )

    with (OUT / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    per_split = Counter(str(row["split"]) for row in rows)
    summary = ", ".join(f"{s} {n}" for s, n in sorted(per_split.items()))
    print(f"{len(rows)} crops from {len(pages)} pages written to {raw} ({summary})")


if __name__ == "__main__":
    main()
