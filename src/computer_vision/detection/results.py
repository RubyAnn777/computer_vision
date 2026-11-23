"""Read the best epoch of a training run from Ultralytics' ``results.csv``."""

import csv
from pathlib import Path

# Ultralytics keeps the epoch with the highest val mAP50-95 as best.pt
# (its "fitness"), so the same column picks the best epoch here.
FITNESS = "metrics/mAP50-95(B)"
COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": FITNESS,
}


def best_epoch(results_csv: Path) -> dict[str, float]:
    """Return the epoch and val scores of the best epoch of one run.

    Parameters
    ----------
    results_csv : Path
        ``results.csv`` from a run folder.

    Returns
    -------
    dict[str, float]
        ``epoch`` plus precision, recall, mAP50 and mAP50-95 of the epoch
        with the highest mAP50-95; on a tie, the earliest such epoch.
    """
    with results_csv.open(newline="") as handle:
        # Ultralytics pads the column names with spaces
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"{results_csv} has no epochs")
    best = max(rows, key=lambda r: (float(r[FITNESS]), -int(r["epoch"])))
    return {"epoch": int(best["epoch"])} | {
        name: float(best[column]) for name, column in COLUMNS.items()
    }
