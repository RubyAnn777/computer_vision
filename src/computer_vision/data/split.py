"""Assign every eligible page to train, val, test or unseen.

The split unit is the whole flyer, so the pages of one flyer never end up in
two different sets (design decision D01). The four older Müller flyers in the
``pool`` are ordered by time: the three oldest are training data and the
newest, 2024 KW49, is the validation set (D13). Test (the Müller 2026 flyer)
and unseen (the supermarket flyers) are already fixed by the ``set`` column
of ``data/annotation_sample.csv``. Pages excluded in
``data/page_screening.csv`` are left out.

No random numbers are involved, so the same input files always give the same
split.
"""

from collections.abc import Iterable, Mapping
from enum import StrEnum

VAL_FLYER_GROUP = "mueller_prospekt-2024-kw49"


class Split(StrEnum):
    """The four sets a page can belong to."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    UNSEEN = "unseen"


def assign_split(
    sample_set: str, flyer_group: str, val_flyer_group: str = VAL_FLYER_GROUP
) -> Split:
    """Return the split of one page.

    Parameters
    ----------
    sample_set : str
        The page's ``set`` in the annotation sample: ``pool``, ``test`` or
        ``unseen``.
    flyer_group : str
        The page's flyer group, used only for ``pool`` pages.
    val_flyer_group : str, optional
        The pool flyer that becomes the validation set.

    Returns
    -------
    Split
        ``val`` for pages of the validation flyer, ``train`` for the other
        pool pages, otherwise ``test`` or ``unseen`` as in the sample.

    Raises
    ------
    ValueError
        If ``sample_set`` is not one of the three known sets.
    """
    if sample_set == "pool":
        return Split.VAL if flyer_group == val_flyer_group else Split.TRAIN
    if sample_set == "test":
        return Split.TEST
    if sample_set == "unseen":
        return Split.UNSEEN
    raise ValueError(f"unknown sample set {sample_set!r}")


def build_split(
    sample_rows: Iterable[Mapping[str, str]],
    screening_rows: Iterable[Mapping[str, str]],
    val_flyer_group: str = VAL_FLYER_GROUP,
) -> list[tuple[str, Split]]:
    """Assign a split to every eligible page of the annotation sample.

    Parameters
    ----------
    sample_rows : Iterable[Mapping[str, str]]
        Rows of ``annotation_sample.csv``; needs ``page_id``, ``set`` and
        ``flyer_group``.
    screening_rows : Iterable[Mapping[str, str]]
        Rows of ``page_screening.csv``; needs ``page_id`` and ``decision``
        (``eligible`` or ``exclude``).
    val_flyer_group : str, optional
        The pool flyer that becomes the validation set.

    Returns
    -------
    list[tuple[str, Split]]
        ``(page_id, split)`` for every eligible page, sorted by page id.

    Raises
    ------
    ValueError
        If a page id repeats, a sample page has no screening decision, a
        decision is neither ``eligible`` nor ``exclude``, or the validation
        flyer is not in the pool.
    """
    sample = list(sample_rows)
    ids = [row["page_id"] for row in sample]
    if len(ids) != len(set(ids)):
        raise ValueError("page ids in the sample must be unique")

    decisions = {row["page_id"]: row["decision"] for row in screening_rows}
    missing = sorted(set(ids) - decisions.keys())
    if missing:
        raise ValueError(f"pages without a screening decision: {missing}")
    unknown = sorted({decisions[i] for i in ids} - {"eligible", "exclude"})
    if unknown:
        raise ValueError(f"unknown screening decisions: {unknown}")

    pool_groups = {row["flyer_group"] for row in sample if row["set"] == "pool"}
    if val_flyer_group not in pool_groups:
        raise ValueError(f"validation flyer {val_flyer_group!r} is not in the pool")

    split = [
        (row["page_id"], assign_split(row["set"], row["flyer_group"], val_flyer_group))
        for row in sample
        if decisions[row["page_id"]] == "eligible"
    ]
    return sorted(split)
