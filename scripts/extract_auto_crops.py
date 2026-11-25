"""Cut the automatic tile collection out of the detector's boxes (D20, D21).

Runs the frozen detector on every rendered extraction page
(``data/interim/pages/extract/``, see ``render_extraction_pages.py``) and saves
every box at or above the recall-first threshold from ``threshold.json``
(0.14 for s2_final) as a PNG crop. Writes

- ``data/processed/tiles/auto/raw/<tile_id>.png``: one crop per detection;
- ``data/processed/tiles/auto/index.csv``: one row per crop with its page,
  confidence, pixel box, size and the SHA-256 of the page image;
- ``data/processed/tiles/auto/run.json``: model, weights hash, both
  thresholds, inference settings, git state and the crop counts.

Extracting once at the low threshold keeps the confidence of every crop, so
the precision-first version of the collection (D21) is a filter on
``index.csv`` (confidence at or above ``precision_first`` with
``MIN_PRECISION``, chosen on val), not a second model run. Which version is
used is decided later by the rule written in D21, before counting.

Usage::

    uv run python scripts/extract_auto_crops.py --run runs/detect/s2_final
    uv run python scripts/extract_auto_crops.py --run runs/detect/s2_final --overwrite
"""

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from computer_vision.data.ids import parse_page_id
from computer_vision.detection.predict import pick_device, predict_page
from computer_vision.detection.run_info import collect_run_info, sha256_of
from computer_vision.detection.threshold import Score, precision_first
from computer_vision.tiles.crops import crop, reading_order, tile_id, to_pixel_box

PAGES = Path("data/interim/pages/extract")
OUT = Path("data/processed/tiles/auto")
MIN_PRECISION = 0.95  # precision-first version (D21): at most 1 wrong box in 20
FIELDS = [
    "tile_id",
    "page_id",
    "flyer_slug",
    "index",
    "confidence",
    "x0",
    "y0",
    "x1",
    "y1",
    "width",
    "height",
    "page_sha256",
]


def read_scores(path: Path) -> list[Score]:
    """Read the per-threshold scores that ``tune_threshold.py`` saved."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    def optional(value: str) -> float | None:
        return float(value) if value else None

    return [
        Score(
            threshold=float(r["threshold"]),
            hits=int(r["hits"]),
            wrong=int(r["wrong"]),
            missed=int(r["missed"]),
            precision=float(r["precision"]),
            recall=float(r["recall"]),
            f1=float(r["f1"]),
            median_clean=optional(r["median_clean"]),
            median_complete=optional(r["median_complete"]),
        )
        for r in rows
    ]


def main() -> None:
    """Detect, crop and index every extraction page."""
    parser = argparse.ArgumentParser(description="Build the raw automatic crops.")
    parser.add_argument("--run", type=Path, required=True, help="training run folder")
    parser.add_argument(
        "--overwrite", action="store_true", help="write into a non-empty folder"
    )
    args = parser.parse_args()

    raw = OUT / "raw"
    if raw.exists() and any(raw.iterdir()) and not args.overwrite:
        raise SystemExit(f"{raw} is not empty; pass --overwrite to rebuild it")
    pages = sorted(PAGES.glob("*.png"))
    if not pages:
        raise SystemExit(f"no pages in {PAGES}; run render_extraction_pages.py first")
    raw.mkdir(parents=True, exist_ok=True)

    recall_first = json.loads((args.run / "threshold.json").read_text())["chosen"]
    strict = precision_first(read_scores(args.run / "threshold.csv"), MIN_PRECISION)
    conf = float(recall_first["threshold"])
    imgsz = json.loads((args.run / "run_info.json").read_text())["settings"]["imgsz"]
    weights = args.run / "weights" / "best.pt"
    device = pick_device()
    model = YOLO(str(weights))

    rows: list[dict[str, object]] = []
    for image_path in pages:
        page_id = image_path.stem
        loaded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if loaded is None:
            raise FileNotFoundError(f"cannot read page image {image_path}")
        image = np.asarray(loaded, dtype=np.uint8)
        height, width = image.shape[:2]
        page_sha = sha256_of(image_path)
        detections = predict_page(model, image_path, imgsz, conf, device)
        boxes = [to_pixel_box(b, width, height) for _, b in detections]
        for index, i in enumerate(reading_order(boxes)):
            box = boxes[i]
            name = tile_id(page_id, "d", index)
            if not cv2.imwrite(str(raw / f"{name}.png"), crop(image, box)):
                raise OSError(f"could not write {name}.png")
            x0, y0, x1, y1 = box
            rows.append(
                {
                    "tile_id": name,
                    "page_id": page_id,
                    "flyer_slug": parse_page_id(page_id).catalog,
                    "index": index,
                    "confidence": round(detections[i][0], 5),
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

    per_flyer = Counter(str(r["flyer_slug"]) for r in rows)
    n_strict = sum(1 for r in rows if float(str(r["confidence"])) >= strict.threshold)
    record = collect_run_info(
        Path(),
        {"weights": weights, "threshold_csv": args.run / "threshold.csv"},
        {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "run": str(args.run),
            "device": device,
            "imgsz": imgsz,
            "extraction_threshold": conf,
            "extraction_rule": "recall-first threshold from threshold.json (D16)",
            "precision_first": {"min_precision": MIN_PRECISION, **asdict(strict)},
            "pages": len(pages),
            "crops": len(rows),
            "crops_at_precision_first": n_strict,
            "crops_per_flyer": dict(sorted(per_flyer.items())),
        },
    )
    (OUT / "run.json").write_text(json.dumps(record, indent=2) + "\n")

    print(f"{len(rows)} crops from {len(pages)} pages at confidence >= {conf:.2f}")
    print(
        f"{n_strict} of them at the precision-first threshold {strict.threshold:.2f}"
        f" (val precision {strict.precision:.3f}, recall {strict.recall:.3f})"
    )
    for flyer, n in sorted(per_flyer.items()):
        print(f"  {flyer}: {n}")


if __name__ == "__main__":
    main()
