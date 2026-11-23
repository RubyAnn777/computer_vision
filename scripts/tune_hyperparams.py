"""Compare a few model settings on the val pages and report the best.

Each candidate in ``GRID`` is trained with the fixed settings of
``src/computer_vision/detection/training.py``, changing only what the
candidate names. The candidates are compared by their best val mAP50-95, the
same score Ultralytics uses to keep ``best.pt``. Only val pages are used.

With 10 val pages the scores are noisy, so a difference below ``TIE`` counts
as a tie, and among tied candidates the smaller, faster one wins (it is
listed first in ``GRID``).

The summary goes to ``runs/detect/tuning_summary.csv``. A run folder that
already has ``run_info.json`` is not trained again, so the script can be
restarted after an interruption.

Usage::

    uv run python scripts/tune_hyperparams.py
"""

import csv
from pathlib import Path

from computer_vision.detection.results import best_epoch
from computer_vision.detection.training import train

RUNS = Path("runs/detect")
TIE = 0.02  # val mAP50-95 differences below this count as noise
# (run name, pretrained model, image size), smallest and fastest first.
# yolo11n at 1280 is the current model.
GRID = [
    ("tune_n_1280", "yolo11n.pt", 1280),
    ("tune_n_1600", "yolo11n.pt", 1600),
    ("tune_s_1280", "yolo11s.pt", 1280),
    ("tune_s_1600", "yolo11s.pt", 1600),
]


def main() -> None:
    """Train every candidate that has no finished run yet, then compare."""
    rows: list[dict[str, object]] = []
    scores: list[float] = []
    for name, weights, imgsz in GRID:
        run = RUNS / name
        if not (run / "run_info.json").exists():
            train(name, weights, imgsz=imgsz)
        best = best_epoch(run / "results.csv")
        rows.append({"name": name, "weights": weights, "imgsz": imgsz} | best)
        scores.append(best["mAP50-95"])

    top = max(scores)
    winner = next(r for r, s in zip(rows, scores, strict=True) if s >= top - TIE)

    with (RUNS / "tuning_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("name          epoch  precision  recall  mAP50  mAP50-95")
    for r in rows:
        print(
            f"{r['name']:<12}  {r['epoch']:5d}  {r['precision']:9.3f}"
            f"  {r['recall']:6.3f}  {r['mAP50']:5.3f}  {r['mAP50-95']:8.3f}"
        )
    print(f"chosen: {winner['name']} (ties within {TIE} go to the smaller model)")


if __name__ == "__main__":
    main()
