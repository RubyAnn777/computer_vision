"""Choose the confidence threshold: which predicted boxes count as products.

The detector gives every box a confidence between 0 and 1. A threshold keeps
the boxes at or above it. A low threshold finds more products but also draws
boxes on text and prices; a high one draws fewer wrong boxes but misses
products. We pick the threshold on the val pages like this:

1. Predict with a very low threshold, so that every candidate box is kept.
2. Match predictions to the hand-drawn boxes, page by page. Going from the
   most to the least confident prediction, each prediction takes the
   hand-drawn box it overlaps most, if that overlap (IoU, see :func:`iou`) is
   at least 0.5 and the box is not taken yet. A matched prediction is a hit
   (true positive), an unmatched one a wrong box (false positive).
3. For every candidate threshold, keep only the predictions at or above it
   and count hits, wrong boxes and missed hand-drawn boxes (false negatives).
4. Pick a threshold. :func:`best_threshold` takes the highest F1 (misses
   and wrong boxes weigh the same); :func:`precision_first` takes the
   highest recall among thresholds whose precision reaches a minimum (a
   wrong crop costs more than a missed one).

Box matching says only whether a product was found. For extracting product
images it also matters how good the crop is, so every hit gets two crop
scores (:func:`crop_scores`): how clean it is (share of the predicted box that
lies inside the hand-drawn box, so not on text or price) and how complete it
is (share of the hand-drawn box that the prediction covers).

Step 3 can reuse the matches of step 2 because matching runs from the most
confident prediction down: the matches among the boxes above a threshold are
the same as if only those boxes had been predicted.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

BoxF = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax in pixels


def iou(a: BoxF, b: BoxF) -> float:
    """Return the overlap of two boxes: shared area divided by joint area.

    1 means the boxes are identical, 0 that they do not touch.
    """
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return shared / (area_a + area_b - shared)


def crop_scores(predicted: BoxF, truth: BoxF) -> tuple[float, float]:
    """Return how clean and how complete a predicted crop is.

    Returns
    -------
    tuple[float, float]
        ``(clean, complete)``: the shared area divided by the predicted box's
        area, and the shared area divided by the hand-drawn box's area. Both
        are 1 for a perfect crop.
    """
    width = min(predicted[2], truth[2]) - max(predicted[0], truth[0])
    height = min(predicted[3], truth[3]) - max(predicted[1], truth[1])
    shared = max(width, 0.0) * max(height, 0.0)
    area_p = (predicted[2] - predicted[0]) * (predicted[3] - predicted[1])
    area_t = (truth[2] - truth[0]) * (truth[3] - truth[1])
    return shared / area_p, shared / area_t


@dataclass(frozen=True)
class Match:
    """One prediction after matching."""

    confidence: float
    hit: bool
    clean: float | None = None  # crop scores, only for hits
    complete: float | None = None


def match(
    predictions: Sequence[tuple[float, BoxF]],
    truths: Sequence[BoxF],
    min_iou: float = 0.5,
) -> list[Match]:
    """Match the predictions of one page to its hand-drawn boxes.

    Parameters
    ----------
    predictions : Sequence[tuple[float, BoxF]]
        ``(confidence, box)`` for every predicted box on the page.
    truths : Sequence[BoxF]
        The hand-drawn boxes of the same page.
    min_iou : float
        Smallest overlap that counts as finding a box.

    Returns
    -------
    list[Match]
        One entry per prediction, most confident first.
    """
    taken = [False] * len(truths)
    result: list[Match] = []
    for confidence, box in sorted(predictions, key=lambda p: -p[0]):
        best, best_iou = -1, min_iou
        for i, truth in enumerate(truths):
            overlap = iou(box, truth)
            if not taken[i] and overlap >= best_iou:
                best, best_iou = i, overlap
        if best < 0:
            result.append(Match(confidence, hit=False))
            continue
        taken[best] = True
        clean, complete = crop_scores(box, truths[best])
        result.append(Match(confidence, True, clean, complete))
    return result


@dataclass(frozen=True)
class Score:
    """Counts and rates for one threshold."""

    threshold: float
    hits: int  # true positives
    wrong: int  # false positives: boxes on something that is not a product
    missed: int  # false negatives: hand-drawn boxes nobody found
    precision: float  # share of drawn boxes that are products
    recall: float  # share of products that were found
    f1: float  # 2 P R / (P + R)
    median_clean: float | None  # crop scores over the kept hits
    median_complete: float | None


def score_at(matched: Sequence[Match], n_truths: int, threshold: float) -> Score:
    """Count hits, wrong boxes and misses when keeping boxes >= ``threshold``.

    Parameters
    ----------
    matched : Sequence[Match]
        Output of :func:`match`, concatenated over all pages.
    n_truths : int
        Number of hand-drawn boxes on those pages.
    threshold : float
        Keep predictions with at least this confidence.
    """
    kept = [m for m in matched if m.confidence >= threshold]
    kept_hits = [m for m in kept if m.hit]
    hits = len(kept_hits)
    wrong = len(kept) - hits
    missed = n_truths - hits
    precision = hits / len(kept) if kept else 0.0
    recall = hits / n_truths if n_truths else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    clean = [m.clean for m in kept_hits if m.clean is not None]
    complete = [m.complete for m in kept_hits if m.complete is not None]
    return Score(
        threshold,
        hits,
        wrong,
        missed,
        precision,
        recall,
        f1,
        median(clean) if clean else None,
        median(complete) if complete else None,
    )


def best_threshold(scores: Sequence[Score]) -> Score:
    """Return the score with the highest F1; on a tie, the lowest threshold."""
    return max(scores, key=lambda s: (s.f1, -s.threshold))


def precision_first(scores: Sequence[Score], min_precision: float) -> Score:
    """Return the highest-recall score with at least ``min_precision``.

    On a tie in recall, the lowest threshold. Raises ``ValueError`` if no
    threshold reaches ``min_precision``.
    """
    good = [s for s in scores if s.precision >= min_precision]
    if not good:
        raise ValueError(f"no threshold reaches precision {min_precision}")
    return max(good, key=lambda s: (s.recall, -s.threshold))
