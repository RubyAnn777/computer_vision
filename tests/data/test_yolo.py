from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from computer_vision.data.yolo import dataset_yaml, read_boxes, to_yolo_line


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "page.txt"
    path.write_text(text)
    return path


def test_read_boxes(tmp_path: Path) -> None:
    path = write(tmp_path, "10 20 30 40\n5 5 6 9\n")
    assert read_boxes(path) == [(10, 20, 30, 40), (5, 5, 6, 9)]


def test_read_boxes_skips_blank_lines(tmp_path: Path) -> None:
    assert read_boxes(write(tmp_path, "\n10 20 30 40\n\n")) == [(10, 20, 30, 40)]


def test_read_boxes_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_boxes(tmp_path / "missing.txt")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "no boxes"),
        ("\n\n", "no boxes"),
        ("10 20 30\n", "expected 4 numbers"),
        ("30 20 10 40\n", "no area"),
        ("10 20 30 20\n", "no area"),
    ],
)
def test_read_boxes_rejects_bad_files(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_boxes(write(tmp_path, text))


def test_to_yolo_line_hand_example() -> None:
    # Box 200 x 400 px centred at (200, 400) on a 1000 x 2000 page.
    assert to_yolo_line((100, 200, 300, 600), 1000, 2000) == (
        "0 0.200000 0.200000 0.200000 0.200000"
    )


def test_to_yolo_line_full_image() -> None:
    line = to_yolo_line((0, 0, 50, 80), 50, 80)
    assert line == "0 0.500000 0.500000 1.000000 1.000000"


def test_to_yolo_line_rejects_box_outside_image() -> None:
    with pytest.raises(ValueError, match="outside"):
        to_yolo_line((0, 0, 51, 80), 50, 80)


def two_distinct(upper: int) -> st.SearchStrategy[list[int]]:
    return st.lists(st.integers(0, upper), min_size=2, max_size=2, unique=True)


@st.composite
def box_on_image(draw: st.DrawFn) -> tuple[tuple[int, int, int, int], int, int]:
    width = draw(st.integers(2, 4000))
    height = draw(st.integers(2, 4000))
    x0, x1 = sorted(draw(two_distinct(width)))
    y0, y1 = sorted(draw(two_distinct(height)))
    return (x0, y0, x1, y1), width, height


@given(box_on_image())
def test_to_yolo_line_round_trips(
    case: tuple[tuple[int, int, int, int], int, int],
) -> None:
    (xmin, ymin, xmax, ymax), width, height = case
    cls, cx, cy, w, h = to_yolo_line((xmin, ymin, xmax, ymax), width, height).split()
    cx_, cy_, w_, h_ = float(cx), float(cy), float(w), float(h)

    assert cls == "0"
    assert all(0 <= v <= 1 for v in (cx_, cy_, w_, h_))
    # Six decimals keep every corner within 0.5 px on pages up to 4000 px.
    assert abs((cx_ - w_ / 2) * width - xmin) < 0.5
    assert abs((cx_ + w_ / 2) * width - xmax) < 0.5
    assert abs((cy_ - h_ / 2) * height - ymin) < 0.5
    assert abs((cy_ + h_ / 2) * height - ymax) < 0.5


def test_dataset_yaml_names_one_class(tmp_path: Path) -> None:
    text = dataset_yaml(tmp_path)
    assert f"path: {tmp_path.resolve()}" in text
    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "test: images/test" in text
    assert text.endswith("names:\n  0: product\n")
