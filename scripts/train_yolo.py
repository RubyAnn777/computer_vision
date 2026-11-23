"""Fine-tune YOLO11n on the training pages and validate on the val flyer.

All training settings are in `SETTINGS` below, so the script is the full
record of how a model was trained. Ultralytics also
saves them as ``args.yaml`` in the run folder, and this script adds
``run_info.json`` with the git commit, input hashes, library versions,
device and training time.

Only the train and val folders of ``data/processed/yolo`` are used here; the
test and unseen pages are never touched during training.

Usage::

    uv run python scripts/build_yolo_dataset.py           # once, if not done
    uv run python scripts/train_yolo.py --epochs 1 --name smoke   # quick check
    uv run python scripts/train_yolo.py                  # the real run
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from computer_vision.detection.run_info import collect_run_info

DATASET = Path("data/processed/yolo/dataset.yaml")
SPLIT = Path("data/splits/split.csv")
WEIGHTS = "yolo11n.pt"  # COCO-pretrained; Ultralytics downloads it on first use

SETTINGS: dict[str, object] = {
    "imgsz": 1280,  # small products stay visible; archive and test both shrink
    "epochs": 150,  # one epoch is only 4 steps, so all 150 take about 13 min
    # No early stopping: the val score dips after warm-up and needs more than
    # 30 epochs to recover, so patience 30 stopped the first run too soon.
    "patience": 150,
    "batch": 8,  # lower to 4 if the Mac runs out of memory
    # Update the weights after every batch. The default (nbs 64) sums 8 batches
    # per update; with 4 batches per epoch that is one update every 2 epochs.
    "nbs": 8,
    "seed": 2026,
    "deterministic": True,
    # Absolute, because Ultralytics nests a relative project inside runs/detect.
    "project": str(Path("runs/detect").resolve()),
    "plots": True,
}


def main() -> None:
    """Train one model and write its run record."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11n.")
    parser.add_argument("--epochs", type=int, help="override SETTINGS['epochs']")
    parser.add_argument("--name", default="yolo11n", help="run folder name")
    args = parser.parse_args()

    if not DATASET.exists():
        raise FileNotFoundError(f"{DATASET} missing: run build_yolo_dataset.py first")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    settings = {**SETTINGS, "name": args.name, "device": device}
    if args.epochs is not None:
        settings["epochs"] = args.epochs

    model = YOLO(WEIGHTS)
    weights_path = Path(str(model.ckpt_path))  # where the download landed
    start = time.time()
    model.train(data=str(DATASET.resolve()), **settings)
    minutes = (time.time() - start) / 60

    trainer = model.trainer
    if trainer is None or trainer.save_dir is None:
        raise RuntimeError("training finished without a results folder")
    save_dir = Path(trainer.save_dir)
    info = collect_run_info(
        repo=Path.cwd(),
        inputs={"split": SPLIT, "weights": weights_path, "dataset": DATASET},
        extra={
            "settings": settings,
            "weights": WEIGHTS,
            "train_minutes": round(minutes, 1),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
            },
            "machine": platform.platform(),
        },
    )
    (save_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"done in {minutes:.1f} min; results in {save_dir}")


if __name__ == "__main__":
    main()
