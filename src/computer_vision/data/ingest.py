"""Bring every flyer page to one common resolution.

Flyers arrive as JPEG pages of different sizes (Aldi Süd 1807 x 3189,
Kaufland 1458 x 2400) and as PDFs (Penny, Norma, Edeka). Annotations are pixel
coordinates, and the policy's minimum box size is in pixels, so every page is
stored with the same long side before anyone draws a box. This plays the role
that a fixed DPI plays for PDF-only datasets.

Downscaling uses area interpolation, which averages source pixels and avoids
aliasing; PDFs are rasterised directly at the target size.
"""

from pathlib import Path

import cv2
import numpy as np
import pymupdf
from numpy.typing import NDArray

LONG_SIDE = 2400


def target_size(width: int, height: int, long_side: int = LONG_SIDE) -> tuple[int, int]:
    """Scale a size so that its longer side equals ``long_side``.

    Parameters
    ----------
    width, height : int
        Original size in pixels.
    long_side : int, optional
        Length of the longer side after scaling.

    Returns
    -------
    tuple[int, int]
        New ``(width, height)``; the shorter side is rounded to the nearest
        pixel, so the aspect ratio is kept to within half a pixel.

    Raises
    ------
    ValueError
        If any argument is not positive.
    """
    if min(width, height, long_side) <= 0:
        raise ValueError(f"sizes must be positive, got {width}, {height}, {long_side}")
    scale = long_side / max(width, height)
    if width >= height:
        return long_side, max(1, round(height * scale))
    return max(1, round(width * scale)), long_side


def standardise(
    image: NDArray[np.uint8], long_side: int = LONG_SIDE
) -> NDArray[np.uint8]:
    """Resize an image so that its longer side equals ``long_side``.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Image of shape (H, W, 3).
    long_side : int, optional
        Length of the longer side after resizing.

    Returns
    -------
    NDArray[np.uint8]
        The resized image, or the input itself if it already has that size.
    """
    height, width = image.shape[:2]
    new_width, new_height = target_size(width, height, long_side)
    if (new_width, new_height) == (width, height):
        return image
    shrinking = new_width < width
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    return np.asarray(resized, dtype=np.uint8)


def render_pdf_page(
    pdf_path: Path, page_index: int, long_side: int = LONG_SIDE
) -> NDArray[np.uint8]:
    """Rasterise one PDF page so that its longer side equals ``long_side``.

    Parameters
    ----------
    pdf_path : Path
        The PDF file.
    page_index : int
        0-based page index within the PDF.
    long_side : int, optional
        Length of the longer side of the rendered page.

    Returns
    -------
    NDArray[np.uint8]
        BGR image of shape (H, W, 3), matching OpenCV's channel order.
    """
    with pymupdf.open(pdf_path) as document:
        page = document[page_index]
        zoom = long_side / max(page.rect.width, page.rect.height)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, 3
        )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return standardise(np.asarray(bgr, dtype=np.uint8), long_side)
