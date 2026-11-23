"""Train one YOLO model with the project's fixed settings.

All training settings live in :data:`SETTINGS`, so this module is the full
record of how a model was trained. Ultralytics also saves them as
``args.yaml`` in the run folder, and :func:`train` adds ``run_info.json``
with the git commit, input hashes, library versions, device and training
time. Only the train and val folders of ``data/processed/yolo`` are used;
test and unseen pages are never touched during training.
"""

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
    # Optimizer written out: these four values are exactly what Ultralytics'
    # "auto" picks for this data (AdamW, lr 0.002 for one class, beta1 0.9,
    # no bias warm-up), so the learning rate can be tuned.
    "optimizer": "AdamW",
    "lr0": 0.002,
    "momentum": 0.9,
    "warmup_bias_lr": 0.0,
    "freeze": None,  # number of first layers kept fixed; None trains all
    # Augmentation defaults, written out so they can be tuned.
    "fliplr": 0.5,  # left-right flip probability (mirrors the text)
    "hsv_h": 0.015,  # colour jitter: hue, saturation, brightness
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "seed": 2026,
    "deterministic": True,
    "plots": True,
}


def train(name: str, weights: str = WEIGHTS, **overrides: object) -> Path:
    """Train one model and write its run record.

    Parameters
    ----------
    name : str
        Run folder name inside ``runs/detect``.
    weights : str
        Pretrained starting point, e.g. ``yolo11n.pt`` or ``yolo11s.pt``.
    **overrides
        Settings that replace the ones in :data:`SETTINGS` for this run.

    Returns
    -------
    Path
        The run folder.
    """
    if not DATASET.exists():
        raise FileNotFoundError(f"{DATASET} missing: run build_yolo_dataset.py first")
    unknown = set(overrides) - set(SETTINGS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    settings = {
        **SETTINGS,
        **overrides,
        "name": name,
        "device": device,
        # Absolute, because Ultralytics nests a relative project inside runs/detect.
        "project": str(Path("runs/detect").resolve()),
        # Reuse the folder of an interrupted run instead of creating name2.
        "exist_ok": True,
    }

    model = YOLO(weights)
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
            "weights": weights,
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
    return save_dir
