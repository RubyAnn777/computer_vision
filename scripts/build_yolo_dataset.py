"""Build ``data/processed/yolo/`` from the split and the hand-drawn boxes.

For every page in ``data/splits/split.csv`` this writes

- ``images/<split>/<page_id>.png``: a link to the annotated page image in
  ``data/interim/pages/annotate/`` (a link, not a copy, so the model trains on
  exactly the pixels the boxes were drawn on);
- ``labels/<split>/<page_id>.txt``: the boxes in YOLO format;

plus ``dataset.yaml`` for Ultralytics. YOLO finds each label by replacing
``images`` with ``labels`` in the image path, so the two trees mirror each
other. The folder is generated and gitignored: it is deleted and rebuilt on
every run, and the committed inputs are the only source of truth.

Usage::

    uv run python scripts/build_yolo_dataset.py
"""

import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2

from computer_vision.data.yolo import dataset_yaml, read_boxes, to_yolo_line

SPLIT = Path("data/splits/split.csv")
IMAGES = Path("data/interim/pages/annotate")
BOXES = Path("data/annotations/pass1")
OUT = Path("data/processed/yolo")


def main() -> None:
    """Rebuild the YOLO dataset folder."""
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if OUT.exists():
        shutil.rmtree(OUT)

    pages: Counter[str] = Counter()
    boxes: Counter[str] = Counter()
    for row in rows:
        page_id, split = row["page_id"], row["split"]
        image_path = IMAGES / f"{page_id}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"cannot read image {image_path}")
        height, width = image.shape[:2]
        page_boxes = read_boxes(BOXES / f"{page_id}.txt")

        image_dir = OUT / "images" / split
        label_dir = OUT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / image_path.name).symlink_to(image_path.resolve())
        lines = [to_yolo_line(b, width, height) for b in page_boxes]
        (label_dir / f"{page_id}.txt").write_text("\n".join(lines) + "\n")
        pages[split] += 1
        boxes[split] += len(page_boxes)

    (OUT / "dataset.yaml").write_text(dataset_yaml(OUT))
    for split in sorted(pages):
        print(f"{split:<7} {pages[split]:>4} pages {boxes[split]:>6} boxes")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
