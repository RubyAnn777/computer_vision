"""Turn the hand-drawn boxes into the dataset layout that YOLO reads.

The labelling tool writes one ``.txt`` per page with one box per line as
``xmin ymin xmax ymax`` in pixels of that page image. YOLO wants one line per
box as ``class cx cy w h``: the class number (always 0, ``product``), then
the box centre, width and height, each divided by the image width or height
so that every number lies between 0 and 1::

    cx = (xmin + xmax) / (2 W)    cy = (ymin + ymax) / (2 H)
    w  = (xmax - xmin) / W        h  = (ymax - ymin) / H

Because the numbers are relative to the image size, YOLO can resize a page
for training without touching the label files.

Every eligible page shows at least one product (no-product pages are
excluded at screening), so a missing or empty box file means the page has not
been labelled. That is an error here, never "a page with zero products".
"""

from pathlib import Path

Box = tuple[int, int, int, int]


def read_boxes(path: Path) -> list[Box]:
    """Read one box file written by the labelling tool.

    Parameters
    ----------
    path : Path
        ``<page_id>.txt`` with one ``xmin ymin xmax ymax`` line per box.

    Returns
    -------
    list[Box]
        The boxes in file order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist (the page is not labelled yet).
    ValueError
        If the file has no boxes, a line does not hold four integers, or a
        box has zero or negative width or height.
    """
    boxes: list[Box] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"{path}:{number}: expected 4 numbers, got {line!r}")
        xmin, ymin, xmax, ymax = (int(p) for p in parts)
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"{path}:{number}: box has no area: {line!r}")
        boxes.append((xmin, ymin, xmax, ymax))
    if not boxes:
        raise ValueError(f"{path}: no boxes; is the page labelled?")
    return boxes


def to_yolo_line(box: Box, width: int, height: int) -> str:
    """Convert one pixel box into a YOLO label line.

    Parameters
    ----------
    box : Box
        ``(xmin, ymin, xmax, ymax)`` in pixels; pixel edges, so the width is
        ``xmax - xmin`` with no ``+1``.
    width, height : int
        Size of the page image in pixels.

    Returns
    -------
    str
        ``"0 cx cy w h"`` with six decimals, all four values in [0, 1].

    Raises
    ------
    ValueError
        If the box reaches outside the image.
    """
    xmin, ymin, xmax, ymax = box
    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
        raise ValueError(f"box {box} lies outside a {width} x {height} image")
    cx = (xmin + xmax) / (2 * width)
    cy = (ymin + ymax) / (2 * height)
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def dataset_yaml(root: Path) -> str:
    """Return the dataset description that Ultralytics reads.

    Parameters
    ----------
    root : Path
        The dataset folder; written as an absolute path, so training works
        from any working directory.

    Returns
    -------
    str
        YAML text naming the train, val and test image folders and the one
        class.
    """
    return (
        f"path: {root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: product\n"
    )
