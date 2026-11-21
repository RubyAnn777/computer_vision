import random
from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st

from computer_vision.data.sample import (
    Candidate,
    select_pages,
    selection_key,
    split_quota,
)


def make_pool(groups: dict[str, dict[str, int]]) -> list[Candidate]:
    """Build candidates: {retailer: {group: n_pages}}, every 7th page ineligible."""
    return [
        Candidate(f"{r}_{g}_p{i:03d}", r, f"{r}_{g}", eligible=i % 7 != 0)
        for r, gs in groups.items()
        for g, n in gs.items()
        for i in range(1, n + 1)
    ]


POOL = make_pool(
    {
        "aldi-sued": {"w46": 32, "w47": 37, "w48": 24, "w49": 25},
        "kaufland": {"w46": 100, "w47": 60, "w48": 86},
    }
)


def test_selection_key_is_fixed_sha256() -> None:
    # Pinned value: guards against any change to the key definition.
    assert selection_key(2026, "penny_x_p001") == (
        "b622f6c838f90a6534c6df04a79878ac32f99e4823d764461bea9d0b3a0c0518"
    )
    assert selection_key(1, "a") != selection_key(2, "a")


@pytest.mark.parametrize(
    ("total", "groups", "expected"),
    [
        (24, ["b", "a", "c"], {"a": 8, "b": 8, "c": 8}),
        (25, ["b", "a", "c"], {"a": 9, "b": 8, "c": 8}),
        (2, ["a", "b", "c"], {"a": 1, "b": 1, "c": 0}),
    ],
)
def test_split_quota(total: int, groups: list[str], expected: dict[str, int]) -> None:
    assert split_quota(total, groups) == expected


def test_split_quota_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        split_quota(-1, ["a"])
    with pytest.raises(ValueError, match="group"):
        split_quota(3, [])


def test_select_pages_gives_quota_per_group_and_retailer() -> None:
    chosen = select_pages(POOL, per_retailer=24, seed=2026)
    assert Counter(c.retailer for c in chosen) == {"aldi-sued": 24, "kaufland": 24}
    per_group = Counter(c.flyer_group for c in chosen)
    assert all(per_group[f"aldi-sued_{w}"] == 6 for w in ("w46", "w47", "w48", "w49"))
    assert all(per_group[f"kaufland_{w}"] == 8 for w in ("w46", "w47", "w48"))


def test_select_pages_only_eligible_sorted_unique() -> None:
    chosen = select_pages(POOL, per_retailer=24, seed=2026)
    assert all(c.eligible for c in chosen)
    ids = [c.page_id for c in chosen]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_select_pages_is_deterministic_and_order_independent() -> None:
    shuffled = POOL.copy()
    random.Random(0).shuffle(shuffled)
    assert select_pages(POOL, 24, 2026) == select_pages(shuffled, 24, 2026)


def test_seed_changes_the_draw() -> None:
    assert select_pages(POOL, 24, 2026) != select_pages(POOL, 24, 2027)


def test_select_pages_rejects_duplicates_and_small_groups() -> None:
    with pytest.raises(ValueError, match="unique"):
        select_pages([*POOL, POOL[0]], 24, 2026)
    tiny = make_pool({"penny": {"w46": 3}})
    with pytest.raises(ValueError, match="quota"):
        select_pages(tiny, 10, 2026)


@given(seed=st.integers(min_value=0, max_value=2**32), k=st.integers(0, 24))
def test_select_pages_property(seed: int, k: int) -> None:
    chosen = select_pages(POOL, per_retailer=k, seed=seed)
    assert len(chosen) == 2 * k
    assert all(c.eligible for c in chosen)
