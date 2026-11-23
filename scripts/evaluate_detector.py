"""Evaluate the frozen model on val, test or unseen pages (run once, at the end).

Only run this after the model and threshold are frozen in a tagged commit
(``git tag frozen-before-test``): test and unseen pages must not influence
any setting. Val is included so the notebook can compare all splits with the
same code.

For the chosen split this writes ``runs/detect/<run>/eval_<split>/`` with

- ``metrics.json``: mAP50 and mAP50-95 (computed by Ultralytics over all
  confidences) and, per threshold, hits, wrong boxes, misses, precision,
  recall, F1 and the crop scores, all at IoU 0.5;
- ``predictions.csv``: every predicted box down to confidence 0.001, with the
  hand-drawn box it matched (if any) and the overlap;
- ``truths.csv``: every hand-drawn box, with the confidence of the
  prediction that found it (empty if none). A box counts as found at a
  threshold ``t`` exactly when that confidence is at least ``t``, because
  matching runs from the most confident prediction down;
- ``pages/<page_id>_t<threshold>.jpg``: every page with green hits, orange
  localization errors (box touches a product but IoU < 0.5), red false
  positives and blue missed products, for the visual examples in the report.

Usage (``R`` is the frozen run folder, e.g. ``runs/detect/tune_n_1280``)::

    uv run python scripts/evaluate_detector.py --run R --split val
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
    iou,
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
    parser.add_argument("--split", choices=["val", "test", "unseen"], required=True)
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
    prediction_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
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
        page_matches = match(predictions, truths, MIN_IOU)
        matched += page_matches
        n_truths += len(truths)
        found_by: dict[int, float] = {}
        ordered = sorted(predictions, key=lambda p: -p[0])  # the order of match()
        for (confidence, box), m in zip(ordered, page_matches, strict=True):
            overlaps = [iou(box, t) for t in truths]
            if m.truth is not None:
                found_by[m.truth] = confidence
            prediction_rows.append(
                {
                    "page_id": page_id,
                    "xmin": round(box[0], 1),
                    "ymin": round(box[1], 1),
                    "xmax": round(box[2], 1),
                    "ymax": round(box[3], 1),
                    "confidence": round(confidence, 5),
                    "truth_index": "" if m.truth is None else m.truth,
                    "iou": "" if m.truth is None else round(overlaps[m.truth], 4),
                    "best_iou_any_truth": round(max(overlaps, default=0.0), 4),
                    "clean": "" if m.clean is None else round(m.clean, 4),
                    "complete": "" if m.complete is None else round(m.complete, 4),
                }
            )
        for index, (x0, y0, x1, y1) in enumerate(truths):
            truth_rows.append(
                {
                    "page_id": page_id,
                    "truth_index": index,  # line order in the box file
                    "xmin": x0,
                    "ymin": y0,
                    "xmax": x1,
                    "ymax": y1,
                    "found_by_confidence": round(found_by[index], 5)
                    if index in found_by
                    else "",
                }
            )
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
    for name, rows in (("predictions", prediction_rows), ("truths", truth_rows)):
        with (out / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
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
