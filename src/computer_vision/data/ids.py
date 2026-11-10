"""Stable identifiers for catalog pages.

Every page is named ``<retailer>_<catalog>_p<NNN>``, for example
``aldi-sued_kw46-26-op-mp_p014``. The id is the file stem of the page image
and of its annotation file, and the join key between the manifest, the split
file and the annotation log, so it is only ever produced by `make_page_id`.

Design notes
------------
- Underscores separate the three fields and never occur inside a field;
  words inside a field are joined by hyphens. This makes ids parseable.
- Page numbers are 1-based (as printed in the PDF viewer) and zero-padded,
  so lexicographic order equals page order. This keeps ``sorted()`` in Python
  and the natural sort in ``bbox_tool.html`` consistent.
"""

import re
import unicodedata
from typing import NamedTuple

PAGE_NUMBER_WIDTH = 3
MAX_PAGE_NUMBER = 10**PAGE_NUMBER_WIDTH - 1

_GERMAN = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_NON_SLUG = re.compile(r"[^a-z0-9]+")
_SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_PAGE_ID = re.compile(rf"({_SLUG})_({_SLUG})_p(\d{{{PAGE_NUMBER_WIDTH}}})")


class PageKey(NamedTuple):
    """The three fields encoded in a page id."""

    retailer: str
    catalog: str
    page_number: int


def slugify(text: str) -> str:
    """Convert free text to a lower-case ASCII slug joined by hyphens.

    Parameters
    ----------
    text : str
        Human-readable name, e.g. ``"ALDI SÜD"``.

    Returns
    -------
    str
        The slug, e.g. ``"aldi-sued"``.

    Raises
    ------
    ValueError
        If no letters or digits remain after normalisation.
    """
    lowered = text.lower().translate(_GERMAN)
    ascii_only = (
        unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode()
    )
    slug = _NON_SLUG.sub("-", ascii_only).strip("-")
    if not slug:
        raise ValueError(f"cannot build a slug from {text!r}")
    return slug


def make_page_id(retailer: str, catalog: str, page_number: int) -> str:
    """Build the canonical id of one catalog page.

    Parameters
    ----------
    retailer : str
        Retailer name; slugified.
    catalog : str
        Catalog name or id; slugified.
    page_number : int
        1-based page number within the catalog.

    Returns
    -------
    str
        Page id of the form ``<retailer>_<catalog>_p<NNN>``.

    Raises
    ------
    ValueError
        If the page number is outside ``1 .. MAX_PAGE_NUMBER`` or a name has
        no letters or digits.
    """
    if not 1 <= page_number <= MAX_PAGE_NUMBER:
        raise ValueError(
            f"page_number must be in 1..{MAX_PAGE_NUMBER}, got {page_number}"
        )
    number = str(page_number).zfill(PAGE_NUMBER_WIDTH)
    return f"{slugify(retailer)}_{slugify(catalog)}_p{number}"


def parse_page_id(page_id: str) -> PageKey:
    """Split a page id back into its fields.

    Parameters
    ----------
    page_id : str
        An id produced by `make_page_id`.

    Returns
    -------
    PageKey
        Retailer slug, catalog slug and page number.

    Raises
    ------
    ValueError
        If the string is not a well-formed page id.
    """
    match = _PAGE_ID.fullmatch(page_id)
    if match is None:
        raise ValueError(f"not a valid page id: {page_id!r}")
    retailer, catalog, number = match.groups()
    return PageKey(retailer, catalog, int(number))
