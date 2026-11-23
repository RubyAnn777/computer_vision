"""Tune model settings on the val pages in two stages (D18).

Stage 1, structure: every model size x image size in ``STAGE1``.
Stage 2, training: starting from the stage-1 winner, change one setting at a
time (``STAGE2``). Every change that beats the winner by more than ``TIE``
is then combined in one final run, ``s2_final``.

All other settings stay as in ``src/computer_vision/detection/training.py``.
Candidates are compared by best val mAP50-95, the score Ultralytics uses to
keep ``best.pt``. With only 10 val pages the scores are noisy, so a
difference of ``TIE`` or less counts as a tie, and a tie goes to the simpler
candidate (listed first). Only val pages are used.

Logging, for inspection later. ``runs/`` is gitignored, so everything small
is also copied to ``docs/tuning/`` to be committed:

- ``log.jsonl``: one line per event (run started, finished, failed,
  skipped, decision), with time, stage, all settings and scores;
- ``stage<N>.csv``: one row per candidate with its settings, best epoch and
  its scores, the scores of the last epoch, training minutes, the
  difference to the baseline and the decision;
- ``report.md``: the same tables in readable form, with the rule applied;
- ``runs/<name>/``: ``results.csv`` (every epoch), ``args.yaml`` (every
  Ultralytics setting) and ``run_info.json`` (git commit, input hashes,
  versions, machine, minutes) of each run.

A run folder that already has ``run_info.json`` is not trained again, so an
interrupted stage can simply be restarted. A run that fails (for example out
of memory) is logged as failed and skipped; the stage continues.

Usage::

    uv run python scripts/tune_hyperparams.py --stage 1   # 8 runs
    uv run python scripts/tune_hyperparams.py --stage 2   # 5 runs (+ final)
"""

import argparse
import csv
import json
import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path

from computer_vision.detection.results import best_epoch, last_epoch
from computer_vision.detection.training import train

RUNS = Path("runs/detect")
DOCS = Path("docs/tuning")
TIE = 0.02  # val mAP50-95 differences up to this count as noise
SCORE = "mAP50-95"

# (model, image size), smallest and fastest first. yolo11m at 1920 px is left
# out: it would not fit in the Mac's memory at batch 8.
STAGE1 = [
    (model, imgsz)
    for model in ("yolo11n", "yolo11s", "yolo11m")
    for imgsz in (1280, 1600, 1920)
    if (model, imgsz) != ("yolo11m", 1920)
]

# One change at a time, applied to the stage-1 winner.
STAGE2: dict[str, dict[str, object]] = {
    "epochs300": {"epochs": 300, "patience": 300},  # longer, still no early stop
    "lr0.001": {"lr0": 0.001},
    "lr0.005": {"lr0": 0.005},
    # Keep the pretrained backbone (layers 0-9 of YOLO11) fixed; with 29
    # training pages this may generalise better.
    "freeze10": {"freeze": 10},
    # No mirrored text, milder colour changes: flyers are printed artwork.
    "flyer_aug": {"fliplr": 0.0, "hsv_h": 0.0, "hsv_s": 0.3, "hsv_v": 0.2},
}

Row = dict[str, object]


def log(event: str, **fields: object) -> None:
    """Append one event to ``docs/tuning/log.jsonl``."""
    DOCS.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now(UTC).isoformat(timespec="seconds"), "event": event}
    with (DOCS / "log.jsonl").open("a") as handle:
        handle.write(json.dumps(entry | fields) + "\n")


def run(stage: int, name: str, weights: str, overrides: dict[str, object]) -> Row:
    """Train ``name`` unless finished; return its settings and scores."""
    folder = RUNS / name
    row: Row = {"stage": stage, "name": name, "weights": weights, **overrides}
    if (folder / "run_info.json").exists():
        log("skipped, already trained", stage=stage, name=name)
    else:
        log("started", stage=stage, name=name, weights=weights, overrides=overrides)
        try:
            train(name, weights, **overrides)
        except Exception as error:  # log any failure, then go on
            log("failed", stage=stage, name=name, error=traceback.format_exc())
            print(f"{name} failed: {error}; continuing")
            return row | {"status": "failed", "error": str(error)}
    info = json.loads((folder / "run_info.json").read_text())
    best = best_epoch(folder / "results.csv")
    last = last_epoch(folder / "results.csv")
    row |= {
        "status": "ok",
        "best_epoch": best.pop("epoch"),
        **best,
        **{f"last_{k}": v for k, v in last.items()},
        "train_minutes": info["train_minutes"],
        "git_commit": info["git"]["commit"],
        "git_dirty": info["git"]["dirty"],
    }
    log("finished", **row)
    copy = DOCS / "runs" / name
    copy.mkdir(parents=True, exist_ok=True)
    for file in ("results.csv", "args.yaml", "run_info.json"):
        shutil.copy2(folder / file, copy / file)
    return row


def score(row: Row) -> float:
    """Return the comparison score of a row (failed runs score -1)."""
    return float(str(row[SCORE])) if row.get("status") == "ok" else -1.0


def pick(rows: list[Row]) -> Row:
    """Return the first finished row within ``TIE`` of the best score."""
    top = max(score(r) for r in rows)
    if top < 0:
        raise RuntimeError("every run failed; see docs/tuning/log.jsonl")
    return next(r for r in rows if score(r) >= top - TIE)


def write(rows: list[Row], stage: int, rule: str, summary: list[str]) -> None:
    """Save one stage as CSV and as a readable section of report.md."""
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (DOCS / f"stage{stage}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def cell(row: Row, key: str) -> str:
        value = row.get(key, "")
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    cols = ["name", "status", "best_epoch", "precision", "recall", "mAP50", SCORE,
            f"delta_{SCORE}", "train_minutes", "decision"]  # fmt: skip
    lines = [
        f"## Stage {stage}",
        "",
        f"Rule: {rule}",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
        *("| " + " | ".join(cell(r, c) for c in cols) + " |" for r in rows),
        "",
        *summary,
        "",
    ]
    report = DOCS / "report.md"
    old = report.read_text() if report.exists() else "# Hyperparameter tuning (D18)\n\n"
    marker = f"## Stage {stage}"
    if marker in old:  # replace this stage's section, keep the other
        start = old.index(marker)
        end = old.find("## Stage", start + len(marker))
        old = old[:start] + (old[end:] if end >= 0 else "")
    report.write_text(old.rstrip() + "\n\n" + "\n".join(lines))
    print("\n".join(lines))


def stage1() -> None:
    """Compare model sizes and image sizes."""
    rows = [run(1, f"s1_{m}_{size}", f"{m}.pt", {"imgsz": size}) for m, size in STAGE1]
    winner = pick(rows)
    for r in rows:
        if r.get("status") == "ok":
            r[f"delta_{SCORE}"] = score(r) - score(winner)
        r["decision"] = "winner" if r is winner else ""
    rule = (
        f"highest val {SCORE}; differences up to {TIE} are a tie, and a tie goes "
        "to the smaller model or image size (listed first)."
    )
    summary = [f"Winner: **{winner['name']}**."]
    write(rows, 1, rule, summary)
    log("decision", stage=1, winner=winner["name"], rule=rule)


def stage2() -> None:
    """Try one training change at a time on the stage-1 winner, then combine."""
    with (DOCS / "stage1.csv").open(newline="") as handle:
        stage1_rows: list[Row] = list(csv.DictReader(handle))
    winner = next(r for r in stage1_rows if r["decision"] == "winner")
    weights, imgsz = str(winner["weights"]), int(str(winner["imgsz"]))
    base = run(2, str(winner["name"]), weights, {"imgsz": imgsz})
    base["decision"] = "baseline (stage-1 winner)"

    rows = [base]
    better: dict[str, object] = {}
    for label, change in STAGE2.items():
        row = run(2, f"s2_{label}", weights, {"imgsz": imgsz, **change})
        if row.get("status") == "ok":
            row[f"delta_{SCORE}"] = score(row) - score(base)
            adopted = score(row) > score(base) + TIE
            row["decision"] = "adopted" if adopted else "not adopted"
            if adopted:
                better |= change
        rows.append(row)
    final = base
    if better:
        combined = run(2, "s2_final", weights, {"imgsz": imgsz, **better})
        combined[f"delta_{SCORE}"] = score(combined) - score(base)
        rows.append(combined)
        final = pick([base, combined]) if combined.get("status") == "ok" else base
        combined["decision"] = (
            "final" if final is combined else "not better than baseline"
        )
    if final is base:
        base["decision"] = "baseline (stage-1 winner), final"
    rule = (
        f"a change is adopted if it beats the baseline's val {SCORE} by more than "
        f"{TIE}; all adopted changes are combined in s2_final, which must again "
        "not be worse than the baseline by more than the tie margin."
    )
    summary = [
        f"Adopted changes: {better or 'none'}.",
        f"Final model: **{final['name']}** (next: tune_threshold.py).",
    ]
    write(rows, 2, rule, summary)
    log("decision", stage=2, adopted=better, final=final["name"], rule=rule)


def main() -> None:
    """Run one stage."""
    parser = argparse.ArgumentParser(description="Tune model settings on val.")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    stage1() if args.stage == 1 else stage2()


if __name__ == "__main__":
    main()
