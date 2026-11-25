import cv2
import numpy as np

from computer_vision.tiles.contact import contact_sheet
from computer_vision.tiles.dedupe import greedy_unique, hamming, phash


def textured(seed: int) -> np.ndarray:
    """A smooth random image, so pHash has structure to describe."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
    return np.asarray(cv2.resize(small, (78, 90), interpolation=cv2.INTER_CUBIC))


def jpeg_copy(image: np.ndarray, quality: int = 60) -> np.ndarray:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return np.asarray(cv2.imdecode(buffer, cv2.IMREAD_COLOR))


def test_hamming() -> None:
    assert hamming(0b1011, 0b1011) == 0
    assert hamming(0b1011, 0b0010) == 2


def test_phash_is_64_bits_and_identical_for_identical_images() -> None:
    image = textured(1)
    assert 0 <= phash(image) < 2**64
    assert phash(image) == phash(image.copy())


def test_phash_survives_jpeg_recompression_and_separates_different_images() -> None:
    a, b = textured(1), textured(2)
    assert hamming(phash(a), phash(jpeg_copy(a))) <= 6
    assert hamming(phash(a), phash(b)) > 6


def test_greedy_unique_keeps_the_first_of_each_group() -> None:
    a, b = textured(1), textured(2)
    hashes = [phash(a), phash(b), phash(jpeg_copy(a)), phash(a)]
    kept, dropped = greedy_unique(hashes, max_distance=6)
    assert kept == [0, 1]
    assert dropped == {2: 0, 3: 0}


def test_greedy_unique_with_distance_minus_one_keeps_everything() -> None:
    kept, dropped = greedy_unique([5, 5, 5], max_distance=-1)
    assert kept == [0, 1, 2]
    assert dropped == {}


def test_contact_sheet_size() -> None:
    images = [textured(i) for i in range(12)]
    sheet = contact_sheet(images, [str(i) for i in range(12)], columns=5, cell=50)
    assert sheet.shape == (3 * (50 + 22), 5 * 50, 3)
