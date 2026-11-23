"""Render the pages listed in ``data/annotation_sample.csv`` for bbox_tool.html.

Writes ``data/interim/pages/annotate/<page_id>.png`` with the long side at
``LONG_SIDE`` pixels, using `standardise` and `render_pdf_page` from
`computer_vision.data.ingest`, so the box coordinates drawn on these images are
valid for training. Pages excluded in ``data/page_screening.csv`` are written
as ``D_<page_id>.png``, so a rebuilt folder keeps the screening marks.
Existing images (under either name) are skipped; delete the folder to rebuild
it.

Usage::

    uv run python scripts/render_annotation_pages.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from computer_vision.data.ingest import render_pdf_page, standardise

SAMPLE = Path("data/annotation_sample.csv")
SCREENING = Path("data/page_screening.csv")
OUT_DIR = Path("data/interim/pages/annotate")


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    """Render every listed page that is not there yet."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(SAMPLE)
    excluded = {r["page_id"] for r in read_csv(SCREENING) if r["decision"] == "exclude"}
    rendered = 0
    for row in rows:
        page_id = row["page_id"]
        name = f"D_{page_id}.png" if page_id in excluded else f"{page_id}.png"
        if (OUT_DIR / f"{page_id}.png").exists() or (
            OUT_DIR / f"D_{page_id}.png"
        ).exists():
            continue
        source = Path(row["source_path"])
        if row["pdf_page_index"]:
            image = render_pdf_page(source, int(row["pdf_page_index"]))
        else:
            loaded = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if loaded is None:
                raise FileNotFoundError(f"cannot read image {source}")
            image = standardise(np.asarray(loaded, dtype=np.uint8))
        cv2.imwrite(str(OUT_DIR / name), image)
        rendered += 1
    print(f"{rendered} rendered, {len(rows) - rendered} already there, in {OUT_DIR}")


if __name__ == "__main__":
    main()
