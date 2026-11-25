"""Extraction must see the model exactly as the frozen evaluation did."""

import csv
import json
from pathlib import Path

import pytest

RUN = Path("runs/detect/s2_final")
WEIGHTS = RUN / "weights" / "best.pt"
SAVED = RUN / "eval_val" / "predictions.csv"
PAGE = "mueller_prospekt-2024-kw49_p001"
IMAGE = Path("data/interim/pages/annotate") / f"{PAGE}.png"
CONF = 0.5


@pytest.mark.slow
@pytest.mark.skipif(
    not (WEIGHTS.exists() and SAVED.exists() and IMAGE.exists()),
    reason="needs the trained model, its val evaluation and the page image",
)
def test_predict_page_matches_the_saved_val_predictions() -> None:
    from ultralytics import YOLO

    from computer_vision.detection.predict import pick_device, predict_page

    imgsz = json.loads((RUN / "run_info.json").read_text())["settings"]["imgsz"]
    got = predict_page(YOLO(str(WEIGHTS)), IMAGE, imgsz, CONF, pick_device())

    with SAVED.open(newline="") as handle:
        saved = [
            (
                float(r["confidence"]),
                tuple(float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")),
            )
            for r in csv.DictReader(handle)
            if r["page_id"] == PAGE and float(r["confidence"]) >= CONF
        ]

    assert len(got) == len(saved)
    # predictions.csv rounds to 0.1 px and 5 digits of confidence
    for (c1, b1), (c2, b2) in zip(
        sorted(got, reverse=True), sorted(saved, reverse=True), strict=True
    ):
        assert c1 == pytest.approx(c2, abs=1e-3)
        assert b1 == pytest.approx(b2, abs=0.5)
