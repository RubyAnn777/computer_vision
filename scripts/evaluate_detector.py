"""Evaluate a frozen model on the test or unseen pages (run once, at the end).

Only run this after the model and threshold are frozen in a tagged commit
(``git tag frozen-before-test``): test and unseen pages must not influence
any setting.

For the chosen split this writes ``runs/detect/<run>/eval_<split>/`` with

- ``metrics.json``: mAP50 and mAP50-95 (computed by Ultralytics over all
  confidences) and, per threshold, hits, wrong boxes, misses, precision,
  recall, F1 and the crop scores, all at IoU 0.5;
- ``pages/<page_id>_t<threshold>.jpg``: every page with green hits, orange
  localization errors (box touches a product but IoU < 0.5), red false
  positives and blue missed products, for the visual examples in the report.

Usage (``R`` is the frozen run folder, e.g. ``runs/detect/tune_n_1280``)::

    uv run python scripts/evaluate_detector.py --run R --split test
    uv run python scripts/evaluate_detector.py --run R --split unseen --thresholds 0.30
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from computer_vision.data.yolo import read_boxes
from computer_vision.detection.threshold import (
    BoxF,
    Match,
    match,
    page_errors,
    score_at,
)

SPLIT = Path("data/splits/split.csv")
IMAGES = Path("data/interim/pages/annotate")
BOXES = Path("data/annotations/pass1")
YOLO_ROOT = Path("data/processed/yolo")
MIN_IOU = 0.5
COLOURS = {  # BGR
    "hits": (0, 160, 0),
    "localization": (0, 140, 255),
    "false_positives": (0, 0, 220),
    "missed": (220, 80, 0),
}


def main() -> None:
    """Score one split and draw the error pictures."""
    parser = argparse.ArgumentParser(description="Evaluate on test or unseen.")
    parser.add_argument("--run", type=Path, required=True, help="training run folder")
    parser.add_argument("--split", choices=["test", "unseen"], required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[],
        help="extra thresholds; the frozen one from threshold.json is always used",
    )
    args = parser.parse_args()

    chosen = json.loads((args.run / "threshold.json").read_text())["chosen"]
    thresholds = sorted({chosen["threshold"], *args.thresholds})
    imgsz = json.loads((args.run / "run_info.json").read_text())["settings"]["imgsz"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = args.run / f"eval_{args.split}"
    (out / "pages").mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.run / "weights" / "best.pt"))

    # mAP over all confidences, by Ultralytics, on the YOLO copy of the split
    data = out / "dataset.yaml"
    data.write_text(
        f"path: {YOLO_ROOT.resolve()}\n"
        "train: images/train\nval: images/val\n"
        f"test: images/{args.split}\nnames:\n  0: product\n"
    )
    stats = model.val(
        data=str(data),
        split="test",
        imgsz=imgsz,
        batch=8,
        device=device,
        plots=False,
        project=str(out.resolve()),
        name="ultralytics",
        exist_ok=True,
    )

    with SPLIT.open() as f:
        pages = sorted(
            r["page_id"] for r in csv.DictReader(f) if r["split"] == args.split
        )
    matched: list[Match] = []
    n_truths = 0
    for page_id in pages:
        image_path = IMAGES / f"{page_id}.png"
        truths: list[BoxF] = list(read_boxes(BOXES / f"{page_id}.txt"))
        (result,) = model.predict(
            str(image_path), imgsz=imgsz, conf=0.001, device=device, verbose=False
        )
        boxes = result.boxes if isinstance(result, Results) else None
        if boxes is None:
            raise RuntimeError(f"no box output for {page_id}")
        predictions = [
            (float(c), (x0, y0, x1, y1))
            for c, (x0, y0, x1, y1) in zip(
                boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
            )
        ]
        matched += match(predictions, truths, MIN_IOU)
        n_truths += len(truths)
        for t in thresholds:
            errors = page_errors(predictions, truths, t, MIN_IOU)
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            for kind, colour in COLOURS.items():
                for x0, y0, x1, y1 in getattr(errors, kind):
                    cv2.rectangle(
                        image, (int(x0), int(y0)), (int(x1), int(y1)), colour, 4
                    )
            cv2.imwrite(str(out / "pages" / f"{page_id}_t{t:.2f}.jpg"), image)

    summary = {
        "split": args.split,
        "pages": len(pages),
        "hand_drawn_boxes": n_truths,
        "min_iou": MIN_IOU,
        "frozen_threshold": chosen["threshold"],
        "mAP50": float(stats.box.map50),
        "mAP50-95": float(stats.box.map),
        "at_thresholds": [asdict(score_at(matched, n_truths, t)) for t in thresholds],
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{args.split}: {len(pages)} pages, {n_truths} hand-drawn boxes")
    print(f"mAP50 {summary['mAP50']:.3f}, mAP50-95 {summary['mAP50-95']:.3f}")
    for s in summary["at_thresholds"]:
        print(
            f"threshold {s['threshold']:.2f}: precision {s['precision']:.3f},"
            f" recall {s['recall']:.3f}, F1 {s['f1']:.3f}"
        )
    print(f"pictures in {out / 'pages'}")


if __name__ == "__main__":
    main()
