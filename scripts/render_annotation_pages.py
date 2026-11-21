"""Render the pages listed in ``data/annotation_sample.csv`` for bbox_tool.html.

Writes ``data/interim/pages/annotate/<page_id>.png`` with the long side at
``LONG_SIDE`` pixels, using the same functions as the full ingest step, so the
box coordinates drawn on these images are valid for training. Existing images
are skipped; delete the folder to rebuild it.

Usage::

    uv run python scripts/render_annotation_pages.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from computer_vision.data.ingest import render_pdf_page, standardise

SAMPLE = Path("data/annotation_sample.csv")
OUT_DIR = Path("data/interim/pages/annotate")


def main() -> None:
    """Render every listed page that is not there yet."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SAMPLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rendered = 0
    for row in rows:
        output = OUT_DIR / f"{row['page_id']}.png"
        if output.exists():
            continue
        source = Path(row["source_path"])
        if row["pdf_page_index"]:
            image = render_pdf_page(source, int(row["pdf_page_index"]))
        else:
            loaded = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if loaded is None:
                raise FileNotFoundError(f"cannot read image {source}")
            image = standardise(np.asarray(loaded, dtype=np.uint8))
        cv2.imwrite(str(output), image)
        rendered += 1
    print(f"{rendered} rendered, {len(rows) - rendered} already there, in {OUT_DIR}")


if __name__ == "__main__":
    main()
