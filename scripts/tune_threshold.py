"""Pick the confidence threshold on the val pages: find many products.

The method is explained step by step in
``src/computer_vision/detection/threshold.py``. This script runs the trained
model on every val page, compares its boxes with the hand-drawn boxes in
``data/annotations/pass1/`` and writes into the run folder

- ``threshold.csv``: hits, wrong boxes, misses, precision, recall and F1 for
  every threshold from 0.01 to 0.99;
- ``threshold.json``: the chosen threshold and its scores.

Rule (D16): finding many products matters more than avoiding wrong boxes,
because wrong crops can be filtered out by hand before building the tile
collection (the brief asks to document such filtering). So the script takes
the threshold with the highest recall whose precision is still at least
``MIN_PRECISION``. The best-F1 threshold is printed for comparison.

Only val pages are used, never test or unseen pages: the threshold is a
setting, and settings are chosen on val.

Usage::

    uv run python scripts/tune_threshold.py --run runs/detect/yolo11n_full
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from computer_vision.data.yolo import read_boxes
from computer_vision.detection.threshold import (
    Match,
    best_threshold,
    match,
    precision_first,
    score_at,
)

SPLIT = Path("data/splits/split.csv")
IMAGES = Path("data/interim/pages/annotate")
BOXES = Path("data/annotations/pass1")
MIN_IOU = 0.5  # overlap needed to count as finding a box (the mAP50 rule)
LOWEST = 0.001  # predict with this, so every candidate box is kept
THRESHOLDS = [i / 100 for i in range(1, 100)]
MIN_PRECISION = 0.80  # at most 1 wrong box in 5


def main() -> None:
    """Score every threshold on the val pages and save the best one."""
    parser = argparse.ArgumentParser(description="Tune the confidence threshold.")
    parser.add_argument("--run", type=Path, required=True, help="training run folder")
    args = parser.parse_args()

    imgsz = json.loads((args.run / "run_info.json").read_text())["settings"]["imgsz"]
    model = YOLO(str(args.run / "weights" / "best.pt"))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with SPLIT.open() as f:
        pages = sorted(r["page_id"] for r in csv.DictReader(f) if r["split"] == "val")

    matched: list[Match] = []
    n_truths = 0
    for page_id in pages:
        truths = read_boxes(BOXES / f"{page_id}.txt")
        (result,) = model.predict(
            str(IMAGES / f"{page_id}.png"),
            imgsz=imgsz,
            conf=LOWEST,
            device=device,
            verbose=False,
        )
        boxes = result.boxes if isinstance(result, Results) else None
        if boxes is None:
            raise RuntimeError(f"no box output for {page_id}")
        # xyxy is in pixels of the original image, like the hand-drawn boxes
        predictions = [
            (float(c), (x0, y0, x1, y1))
            for c, (x0, y0, x1, y1) in zip(
                boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
            )
        ]
        matched += match(predictions, truths, MIN_IOU)
        n_truths += len(truths)

    scores = [score_at(matched, n_truths, t) for t in THRESHOLDS]
    chosen = precision_first(scores, MIN_PRECISION)
    best_f1 = best_threshold(scores)

    with (args.run / "threshold.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(scores[0])))
        writer.writeheader()
        writer.writerows(asdict(s) for s in scores)
    summary = {
        "pages": len(pages),
        "hand_drawn_boxes": n_truths,
        "min_iou": MIN_IOU,
        "rule": f"highest recall with precision >= {MIN_PRECISION}",
        "chosen": asdict(chosen),
        "best_f1_for_comparison": asdict(best_f1),
    }
    (args.run / "threshold.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{len(pages)} val pages, {n_truths} hand-drawn boxes")
    print("threshold  hits  wrong  missed  precision  recall    F1  clean  complete")
    for sc in scores[4::5]:  # every 0.05
        print(
            f"{sc.threshold:9.2f}  {sc.hits:4d}  {sc.wrong:5d}  {sc.missed:6d}"
            f"  {sc.precision:9.3f}  {sc.recall:6.3f}  {sc.f1:.3f}"
            f"  {sc.median_clean or 0:5.3f}  {sc.median_complete or 0:8.3f}"
        )
    for label, sc in (("chosen", chosen), ("best F1", best_f1)):
        print(
            f"{label}: {sc.threshold:.2f} (precision {sc.precision:.3f},"
            f" recall {sc.recall:.3f}, F1 {sc.f1:.3f})"
        )


if __name__ == "__main__":
    main()
