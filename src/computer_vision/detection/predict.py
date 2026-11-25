"""Run the trained detector on one page and return plain Python boxes.

The tuning and evaluation scripts each call Ultralytics inline; they are
frozen under the ``frozen-before-test`` tag and stay as they are. Building the
automatic tile collection uses this one function instead, with the same
call pattern (same ``imgsz``, same device rule), so extraction and evaluation
see the model in the same way. ``tests/detection/test_predict.py`` checks
that on a val page against the saved evaluation predictions.
"""

from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from computer_vision.detection.threshold import BoxF


def pick_device() -> str:
    """Return ``"mps"`` on an Apple-silicon Mac, otherwise ``"cpu"``."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def predict_page(
    model: YOLO, image_path: Path, imgsz: int, conf: float, device: str
) -> list[tuple[float, BoxF]]:
    """Detect products on one page image.

    Parameters
    ----------
    model : YOLO
        The loaded detector, e.g. ``YOLO("runs/detect/s2_final/weights/best.pt")``.
    image_path : Path
        The page image, rendered like the training pages (long side 2400).
    imgsz : int
        Inference size; use the training run's own ``imgsz``.
    conf : float
        Keep boxes with at least this confidence.
    device : str
        ``"mps"``, ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    list[tuple[float, BoxF]]
        ``(confidence, (xmin, ymin, xmax, ymax))`` per box, in pixels of the
        original page image, in the order Ultralytics returns them (most
        confident first).

    Raises
    ------
    RuntimeError
        If Ultralytics returns no box output for the page.
    """
    (result,) = model.predict(
        str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False
    )
    boxes = result.boxes if isinstance(result, Results) else None
    if boxes is None:
        raise RuntimeError(f"no box output for {image_path}")
    return [
        (float(c), (x0, y0, x1, y1))
        for c, (x0, y0, x1, y1) in zip(
            boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
        )
    ]
