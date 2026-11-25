"""Which catalog pages feed the automatic tile collection (D20).

The automatic collection is built only from flyers marked
``split_role=extraction`` in ``data/raw/flyers.csv``: they were set aside
before the detector existed, were never annotated, and were used neither for
training nor for choosing thresholds. Their pages are disjoint from the pages
behind the manual collection, so the two collections are independent.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from computer_vision.data.ids import make_page_id

EXTRACTION_ROLE = "extraction"


@dataclass(frozen=True)
class SourcePage:
    """One page to run the detector on."""

    page_id: str
    flyer_slug: str
    source_path: Path
    pdf_page_index: int | None  # None for flyers saved as page images


def extraction_pages(
    flyers: Iterable[Mapping[str, str]],
    pages: Iterable[Mapping[str, str]],
    retailer: str,
) -> list[SourcePage]:
    """List the pages of one retailer's extraction flyers.

    Parameters
    ----------
    flyers : Iterable[Mapping[str, str]]
        Rows of ``flyers.csv`` (``retailer``, ``flyer_slug``, ``split_role``).
    pages : Iterable[Mapping[str, str]]
        Rows of ``pages.csv`` (``retailer``, ``flyer_slug``, ``page_number``,
        ``path``, ``pdf_page_index``).
    retailer : str
        Retailer slug, e.g. ``"mueller"``.

    Returns
    -------
    list[SourcePage]
        The pages, sorted by page id.

    Raises
    ------
    ValueError
        If an extraction flyer has no pages in ``pages.csv``.
    """
    slugs = {
        f["flyer_slug"]
        for f in flyers
        if f["retailer"] == retailer and f["split_role"] == EXTRACTION_ROLE
    }
    selected = [
        SourcePage(
            page_id=make_page_id(retailer, p["flyer_slug"], int(p["page_number"])),
            flyer_slug=p["flyer_slug"],
            source_path=Path(p["path"]),
            pdf_page_index=int(p["pdf_page_index"]) if p["pdf_page_index"] else None,
        )
        for p in pages
        if p["retailer"] == retailer and p["flyer_slug"] in slugs
    ]
    missing = slugs - {s.flyer_slug for s in selected}
    if missing:
        raise ValueError(f"extraction flyers without pages: {sorted(missing)}")
    return sorted(selected, key=lambda s: s.page_id)
