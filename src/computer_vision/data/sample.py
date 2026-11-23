"""Deterministic selection of the pages to annotate.

A fixed number of pages per retailer is drawn, spread evenly over that retailer's flyer
groups (weeks). The same function draws the Müller pool and the unseen supermarket
pages.
The draw must give the same pages on every machine and every Python version, so it
does not use a random number generator. Instead every page gets a selection key,
the SHA-256 of ``"<seed>:<page_id>"``, and each flyer group keeps the pages with
the smallest keys. This is a seeded random permutation that depends only on the
seed and the page ids, not on file order, library versions or the platform.
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """One page that may be drawn for annotation.

    Attributes
    ----------
    page_id : str
        Canonical page id, see `computer_vision.data.ids.make_page_id`.
    retailer : str
        Retailer slug, the stratum the quota is defined on.
    flyer_group : str
        Weekly flyer group; the quota is spread evenly over these.
    eligible : bool
        Whether the page may be drawn at all. The sampling script sets it from
        the automatic ``suggested_eligible`` flag in ``pages.csv``; the human
        screening against the page rules (P1 to P7) comes after the draw.
    """

    page_id: str
    retailer: str
    flyer_group: str
    eligible: bool


def selection_key(seed: int, page_id: str) -> str:
    """Return the hash that orders pages within a flyer group.

    Parameters
    ----------
    seed : int
        The sampling seed.
    page_id : str
        Canonical page id.

    Returns
    -------
    str
        Hex SHA-256 digest of ``"<seed>:<page_id>"``.
    """
    return hashlib.sha256(f"{seed}:{page_id}".encode()).hexdigest()


def split_quota(total: int, groups: list[str]) -> dict[str, int]:
    """Spread ``total`` pages as evenly as possible over sorted groups.

    Parameters
    ----------
    total : int
        Pages to draw for one retailer.
    groups : list[str]
        The retailer's flyer groups.

    Returns
    -------
    dict[str, int]
        Pages per group. When ``total`` does not divide evenly, the first
        groups in sorted order get one page more.

    Raises
    ------
    ValueError
        If ``total`` is negative or there are no groups.
    """
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if not groups:
        raise ValueError("at least one flyer group is needed")
    ordered = sorted(groups)
    base, extra = divmod(total, len(ordered))
    return {g: base + (1 if i < extra else 0) for i, g in enumerate(ordered)}


def select_pages(
    candidates: Iterable[Candidate], per_retailer: int, seed: int
) -> list[Candidate]:
    """Draw ``per_retailer`` eligible pages per retailer, spread over its groups.

    Parameters
    ----------
    candidates : Iterable[Candidate]
        All pages of one set (pool or unseen), in any order.
    per_retailer : int
        Pages to draw for each retailer.
    seed : int
        Sampling seed; a different seed gives a different, equally valid draw.

    Returns
    -------
    list[Candidate]
        The drawn pages, sorted by page id.

    Raises
    ------
    ValueError
        If page ids repeat, or a flyer group has fewer eligible pages than its
        quota.
    """
    pages = list(candidates)
    ids = [c.page_id for c in pages]
    if len(ids) != len(set(ids)):
        raise ValueError("page ids must be unique")

    groups: defaultdict[str, defaultdict[str, list[Candidate]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for c in pages:
        if c.eligible:
            groups[c.retailer][c.flyer_group].append(c)

    chosen: list[Candidate] = []
    for retailer in sorted(groups):
        quota = split_quota(per_retailer, list(groups[retailer]))
        for group, k in quota.items():
            pool = groups[retailer][group]
            if len(pool) < k:
                raise ValueError(
                    f"{group} has {len(pool)} eligible pages, fewer than its quota {k}"
                )
            pool.sort(key=lambda c: selection_key(seed, c.page_id))
            chosen.extend(pool[:k])
    return sorted(chosen, key=lambda c: c.page_id)
