import csv
import random
from pathlib import Path

import pytest

from computer_vision.data.split import (
    VAL_FLYER_GROUP,
    Split,
    assign_split,
    build_split,
)

REPO = Path(__file__).resolve().parents[2]


def sample_row(page_id: str, subset: str, group: str) -> dict[str, str]:
    return {"page_id": page_id, "set": subset, "flyer_group": group}


SAMPLE = [
    sample_row("m_a_p001", "pool", "mueller_a"),
    sample_row("m_a_p002", "pool", "mueller_a"),
    sample_row("m_v_p001", "pool", VAL_FLYER_GROUP),
    sample_row("m_t_p001", "test", "mueller_t"),
    sample_row("s_u_p001", "unseen", "shop_u"),
    sample_row("s_u_p002", "unseen", "shop_u"),
]
# Every page is eligible except the last unseen page.
SCREENING = [{"page_id": r["page_id"], "decision": "eligible"} for r in SAMPLE]
SCREENING[-1] = {"page_id": "s_u_p002", "decision": "exclude"}


@pytest.mark.parametrize(
    ("subset", "group", "expected"),
    [
        ("pool", "mueller_a", Split.TRAIN),
        ("pool", VAL_FLYER_GROUP, Split.VAL),
        ("test", "mueller_t", Split.TEST),
        ("unseen", "shop_u", Split.UNSEEN),
    ],
)
def test_assign_split(subset: str, group: str, expected: Split) -> None:
    assert assign_split(subset, group) == expected


def test_assign_split_rejects_unknown_set() -> None:
    with pytest.raises(ValueError, match="unknown sample set"):
        assign_split("fresh", "x")


def test_build_split_keeps_only_eligible_pages() -> None:
    assert build_split(SAMPLE, SCREENING) == [
        ("m_a_p001", Split.TRAIN),
        ("m_a_p002", Split.TRAIN),
        ("m_t_p001", Split.TEST),
        ("m_v_p001", Split.VAL),
        ("s_u_p001", Split.UNSEEN),
    ]


def test_build_split_ignores_input_order() -> None:
    shuffled = SAMPLE.copy()
    random.Random(0).shuffle(shuffled)
    assert build_split(shuffled, SCREENING) == build_split(SAMPLE, SCREENING)


def test_build_split_rejects_page_without_decision() -> None:
    with pytest.raises(ValueError, match="without a screening decision"):
        build_split(SAMPLE, SCREENING[1:])


def test_build_split_rejects_unknown_decision() -> None:
    screening = [*SCREENING[:-1], {"page_id": "s_u_p002", "decision": "maybe"}]
    with pytest.raises(ValueError, match="unknown screening decisions"):
        build_split(SAMPLE, screening)


def test_build_split_rejects_duplicate_page_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_split([*SAMPLE, SAMPLE[0]], SCREENING)


def test_build_split_rejects_missing_val_flyer() -> None:
    with pytest.raises(ValueError, match="not in the pool"):
        build_split(SAMPLE, SCREENING, val_flyer_group="mueller_missing")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.skipif(
    not (REPO / "data/splits/split.csv").exists(), reason="split not written yet"
)
def test_committed_split_matches_inputs() -> None:
    """The committed split is exactly what the code builds from the inputs."""
    expected = build_split(
        read_csv(REPO / "data/annotation_sample.csv"),
        read_csv(REPO / "data/page_screening.csv"),
    )
    committed = [
        (r["page_id"], Split(r["split"]))
        for r in read_csv(REPO / "data/splits/split.csv")
    ]
    assert committed == expected
