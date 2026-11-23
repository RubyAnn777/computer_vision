import pytest

from computer_vision.detection.threshold import (
    Match,
    Score,
    best_threshold,
    crop_scores,
    iou,
    match,
    page_errors,
    precision_first,
    score_at,
)

BOX = (0.0, 0.0, 10.0, 10.0)


def test_iou_identical_disjoint_and_half():
    assert iou(BOX, BOX) == 1.0
    assert iou(BOX, (20.0, 20.0, 30.0, 30.0)) == 0.0
    # shares 50 of 150 area units
    assert iou(BOX, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(1 / 3)


def test_match_most_confident_prediction_takes_the_box():
    predictions = [(0.4, BOX), (0.9, BOX)]
    assert match(predictions, [BOX]) == [
        Match(0.9, True, 1.0, 1.0, 0),
        Match(0.4, False),
    ]


def test_match_needs_enough_overlap():
    shifted = (5.0, 0.0, 15.0, 10.0)  # IoU 1/3
    assert not match([(0.8, shifted)], [BOX])[0].hit
    assert match([(0.8, shifted)], [BOX], min_iou=0.3)[0].hit


def test_match_takes_the_box_it_overlaps_most():
    other = (6.0, 0.0, 16.0, 10.0)
    prediction = (5.0, 0.0, 15.0, 10.0)  # IoU 1/3 with BOX, 9/11 with other
    (first,) = match([(0.8, prediction)], [BOX, other], min_iou=0.3)
    assert first.clean == pytest.approx(0.9)  # 9 of its 10 columns lie in other
    # BOX is still free for a second prediction
    both = match([(0.8, prediction), (0.7, BOX)], [BOX, other], min_iou=0.3)
    assert [m.hit for m in both] == [True, True]


def test_crop_scores_box_with_price_tag_is_complete_but_not_clean():
    with_tag = (0.0, 0.0, 10.0, 20.0)  # the product plus the same area again
    assert crop_scores(with_tag, BOX) == (0.5, 1.0)
    assert crop_scores(BOX, with_tag) == (1.0, 0.5)


def test_score_at_counts():
    matched = [Match(0.9, True, 0.8, 1.0), Match(0.6, False), Match(0.3, True)]
    high = score_at(matched, n_truths=4, threshold=0.5)
    assert (high.hits, high.wrong, high.missed) == (1, 1, 3)
    assert high.precision == 0.5
    assert high.recall == 0.25
    assert high.f1 == pytest.approx(2 * 0.5 * 0.25 / 0.75)
    assert (high.median_clean, high.median_complete) == (0.8, 1.0)
    nothing = score_at(matched, n_truths=4, threshold=0.95)
    assert (nothing.precision, nothing.recall, nothing.f1) == (0.0, 0.0, 0.0)
    assert nothing.median_clean is None


def score(t: float, precision: float, recall: float, f1: float) -> Score:
    return Score(t, 0, 0, 0, precision, recall, f1, None, None)


def test_best_threshold_highest_f1_then_lowest_threshold():
    scores = [score(0.3, 0, 0, 0.7), score(0.5, 0, 0, 0.8), score(0.7, 0, 0, 0.8)]
    assert best_threshold(scores).threshold == 0.5


def test_precision_first_highest_recall_above_minimum():
    scores = [
        score(0.3, 0.88, 0.90, 0),
        score(0.8, 0.95, 0.77, 0),
        score(0.9, 1, 0.5, 0),
    ]
    assert precision_first(scores, 0.95).threshold == 0.8
    # same recall: the higher threshold only drops wrong boxes, so it wins
    same_recall = [score(0.06, 0.81, 0.986, 0), score(0.10, 0.84, 0.986, 0)]
    assert precision_first(same_recall, 0.80).threshold == 0.10
    with pytest.raises(ValueError, match="no threshold"):
        precision_first(scores[:1], 0.95)


def test_page_errors_sorts_every_box_into_one_kind():
    other = (20.0, 0.0, 30.0, 10.0)
    loose = (0.0, 0.0, 10.0, 30.0)  # IoU 1/3 with BOX
    far = (50.0, 50.0, 60.0, 60.0)
    predictions = [(0.9, BOX), (0.8, loose), (0.7, far), (0.1, other)]
    errors = page_errors(predictions, [BOX, other], threshold=0.5)
    assert errors.hits == [BOX]
    assert errors.localization == [loose]
    assert errors.false_positives == [far]
    assert errors.missed == [other]  # its prediction is below the threshold
