"""Render the Müller extraction pages for the automatic tile collection.

Lists the pages of every Müller flyer marked ``split_role=extraction`` in
``data/raw/flyers.csv`` (`extraction_pages`, D20) and writes
``data/interim/pages/extract/<page_id>.png`` with the long side at
``LONG_SIDE`` pixels, using the same `render_pdf_page` and `standardise` as
the annotation pages. The detector was trained and its thresholds chosen at
that scale, so rendering differently would change product sizes in pixels
and move the operating point. Existing images are skipped.

Usage::

    uv run python scripts/render_extraction_pages.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from computer_vision.data.ingest import render_pdf_page, standardise
from computer_vision.tiles.sources import extraction_pages

FLYERS = Path("data/raw/flyers.csv")
PAGES = Path("data/raw/pages.csv")
OUT_DIR = Path("data/interim/pages/extract")
RETAILER = "mueller"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    """Render every extraction page that is not there yet."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = extraction_pages(read_csv(FLYERS), read_csv(PAGES), RETAILER)
    rendered = 0
    for page in pages:
        target = OUT_DIR / f"{page.page_id}.png"
        if target.exists():
            continue
        if page.pdf_page_index is not None:
            image = render_pdf_page(page.source_path, page.pdf_page_index)
        else:
            loaded = cv2.imread(str(page.source_path), cv2.IMREAD_COLOR)
            if loaded is None:
                raise FileNotFoundError(f"cannot read image {page.source_path}")
            image = standardise(np.asarray(loaded, dtype=np.uint8))
        if not cv2.imwrite(str(target), image):
            raise OSError(f"could not write {target}")
        rendered += 1
    print(f"{rendered} rendered, {len(pages) - rendered} already there, in {OUT_DIR}")


if __name__ == "__main__":
    main()
