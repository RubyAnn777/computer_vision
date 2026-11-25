"""Find near-duplicate tiles with a perceptual hash (pHash).

Catalogs show the same packshot many times (same product on several pages,
the same brand every year), and a mosaic built from copies looks repetitive
while the brief asks for diversity over count. A perceptual hash summarises
an image's coarse structure in 64 bits, so two tiles that look alike get
hashes that differ in only a few bits:

1. convert to grey and shrink to 32 x 32 (only coarse structure survives,
   Lecture 2's low-pass idea);
2. take the 2-D discrete cosine transform and keep the 8 x 8 lowest
   frequencies;
3. set each bit to 1 if that coefficient is above the median of the 64.

The Hamming distance (number of differing bits) is then the dissimilarity:
0 for identical tiles, about 32 for unrelated ones. `greedy_unique` keeps a
tile unless it lies within ``max_distance`` of one already kept.
"""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

HASH_SIDE = 8  # 8 x 8 = 64 bits
DCT_SIDE = 32


def phash(image: NDArray[np.uint8]) -> int:
    """Return the 64-bit perceptual hash of a BGR image as an integer."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (DCT_SIDE, DCT_SIDE), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(np.asarray(small, dtype=np.float32))
    low = coefficients[:HASH_SIDE, :HASH_SIDE].flatten()
    bits = low > np.median(low)
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    """Return the number of bits in which two hashes differ."""
    return (a ^ b).bit_count()


def greedy_unique(
    hashes: Sequence[int], max_distance: int
) -> tuple[list[int], dict[int, int]]:
    """Keep tiles in order, dropping any within ``max_distance`` of a kept one.

    Parameters
    ----------
    hashes : Sequence[int]
        One hash per tile, in a fixed order (sorted tile ids), so the result
        is deterministic.
    max_distance : int
        Largest Hamming distance that still counts as a duplicate.

    Returns
    -------
    tuple[list[int], dict[int, int]]
        Indices of the kept tiles, and for every dropped tile the index of
        the kept tile it duplicates (for the filter log).
    """
    kept: list[int] = []
    dropped: dict[int, int] = {}
    for i, h in enumerate(hashes):
        match = next((k for k in kept if hamming(h, hashes[k]) <= max_distance), None)
        if match is None:
            kept.append(i)
        else:
            dropped[i] = match
    return kept, dropped
