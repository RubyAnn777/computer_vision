"""Fine-tune a pretrained YOLO model on the training pages.

The settings are in ``src/computer_vision/detection/training.py``. Command
line options change single settings for one run; every run records what it
used in ``args.yaml`` and ``run_info.json``.

Usage::

    uv run python scripts/build_yolo_dataset.py           # once, if not done
    uv run python scripts/train_yolo.py --epochs 1 --name smoke   # quick check
    uv run python scripts/train_yolo.py --name yolo11n_full       # the real run
"""

import argparse

from computer_vision.detection.training import WEIGHTS, train


def main() -> None:
    """Train one model."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLO.")
    parser.add_argument("--name", default="yolo11n", help="run folder name")
    parser.add_argument("--weights", default=WEIGHTS, help="pretrained model")
    parser.add_argument("--epochs", type=int, help="override the epoch count")
    parser.add_argument("--imgsz", type=int, help="override the image size")
    args = parser.parse_args()

    overrides = {
        key: value
        for key, value in {"epochs": args.epochs, "imgsz": args.imgsz}.items()
        if value is not None
    }
    train(args.name, args.weights, **overrides)


if __name__ == "__main__":
    main()
