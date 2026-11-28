"""Score how much a mosaic looks like its photo from a distance (D25, D27).

Seen up close a mosaic is products, not the photo, so a pixel-by-pixel
comparison would call every mosaic bad. From a distance the eye blurs the
image (a low-pass filter, Lecture 2). We imitate that:

1. bring the photo and the mosaic to the same working size, ``WORK_WIDTH``
   pixels wide, with area interpolation (an average, so no aliasing);
2. blur both with the same Gaussian, ``sigma_frac`` of the image width;
3. compare the blurred images with two standard measures:

   - SSIM (structural similarity, Wang et al. 2004) on lightness L*:
     compares local brightness, contrast and pattern; 1 means identical;
   - mean colour difference Delta E (CIEDE2000): below about 2 is hard to see,
     above 10 clearly different.

The blur is tied to the image, not to the tile, so it stands for one fixed
viewing distance. Mosaics with different numbers of columns are then judged
from the same distance and can be compared (D27). The default, 1/120 of the
width, is half a tile at 60 columns. It is an assumption, so conclusions are
checked at half and at twice that width.

Matching uses Delta E 1976 (plain distance in CIELAB); scoring with CIEDE2000,
a different formula, avoids grading the CIELAB matcher on its own exam. Some
bias towards CIELAB remains because CIEDE2000 is also built on CIELAB.
"""

from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.color import deltaE_ciede2000
from skimage.metrics import structural_similarity

from computer_vision.mosaic.descriptors import to_lab

WORK_WIDTH = 960  # pixels; far finer than the blur, so the result barely depends on it
SIGMA_FRAC = 1 / 120  # blur width as a share of the image width


class Scores(NamedTuple):
    """Viewing-distance similarity of a mosaic to its photo."""

    ssim: float  # structural similarity of lightness, 1 = identical
    delta_e: float  # mean CIEDE2000 colour difference, 0 = identical


def distant_view(
    image: NDArray[np.uint8],
    size: tuple[int, int],
    sigma_frac: float = SIGMA_FRAC,
) -> NDArray[np.float32]:
    """Return the image as seen from a distance, in CIELAB.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Photo or mosaic, BGR.
    size : tuple[int, int]
        Working ``(width, height)`` in pixels; the same for both images that
        are compared.
    sigma_frac : float, optional
        Gaussian blur width as a share of the image width.

    Returns
    -------
    NDArray[np.float32]
        CIELAB image of shape ``(height, width, 3)``.
    """
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(
        np.asarray(small, dtype=np.uint8), (0, 0), sigma_frac * size[0]
    )
    return to_lab(np.asarray(blurred, dtype=np.uint8))


def viewing_scores(
    photo: NDArray[np.uint8],
    mosaic: NDArray[np.uint8],
    sigma_frac: float = SIGMA_FRAC,
) -> Scores:
    """Compare photo and mosaic after the same viewing-distance blur.

    Both are brought to ``WORK_WIDTH`` pixels wide, with the height of the
    mosaic's aspect ratio (the photo was resized to that ratio to build it).
    """
    height, width = mosaic.shape[:2]
    size = (WORK_WIDTH, max(1, round(WORK_WIDTH * height / width)))
    a = distant_view(photo, size, sigma_frac)
    b = distant_view(mosaic, size, sigma_frac)
    ssim = structural_similarity(a[..., 0], b[..., 0], data_range=100.0)
    delta_e = deltaE_ciede2000(a, b).mean()
    return Scores(float(ssim), float(delta_e))
