from pathlib import Path

import pytest

from computer_vision.detection.results import best_epoch, last_epoch

HEADER = (
    "                  epoch,  metrics/precision(B),  metrics/recall(B),"
    "  metrics/mAP50(B),  metrics/mAP50-95(B)\n"
)


def test_best_epoch_picks_highest_map50_95_then_earliest(tmp_path: Path):
    path = tmp_path / "results.csv"
    path.write_text(
        HEADER
        + "1,0.5,0.5,0.7,0.50\n"
        + "2,0.9,0.8,0.9,0.70\n"
        + "3,0.8,0.9,0.95,0.70\n"
    )
    assert best_epoch(path) == {
        "epoch": 2,
        "precision": 0.9,
        "recall": 0.8,
        "mAP50": 0.9,
        "mAP50-95": 0.7,
    }
    assert last_epoch(path)["epoch"] == 3


def test_best_epoch_rejects_empty_file(tmp_path: Path):
    path = tmp_path / "results.csv"
    path.write_text(HEADER)
    with pytest.raises(ValueError, match="no epochs"):
        best_epoch(path)
