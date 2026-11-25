from pathlib import Path

import pytest

from computer_vision.tiles.sources import SourcePage, extraction_pages

FLYERS = [
    {
        "retailer": "mueller",
        "flyer_slug": "spielwaren-kw47-2026",
        "split_role": "extraction",
    },
    {"retailer": "mueller", "flyer_slug": "prospekt-2024-kw46", "split_role": "pool"},
    {"retailer": "aldi-sued", "flyer_slug": "kw46", "split_role": "extraction"},
]
PDF = "data/raw/mueller/spielwaren-kw47-2026/flyer.pdf"
PAGES = [
    {
        "retailer": "mueller",
        "flyer_slug": "spielwaren-kw47-2026",
        "page_number": "2",
        "path": PDF,
        "pdf_page_index": "1",
    },
    {
        "retailer": "mueller",
        "flyer_slug": "spielwaren-kw47-2026",
        "page_number": "1",
        "path": PDF,
        "pdf_page_index": "0",
    },
    {
        "retailer": "mueller",
        "flyer_slug": "prospekt-2024-kw46",
        "page_number": "1",
        "path": "x.jpg",
        "pdf_page_index": "",
    },
    {
        "retailer": "aldi-sued",
        "flyer_slug": "kw46",
        "page_number": "1",
        "path": "y.jpg",
        "pdf_page_index": "",
    },
]


def test_keeps_only_the_retailers_extraction_flyers_sorted() -> None:
    assert extraction_pages(FLYERS, PAGES, "mueller") == [
        SourcePage(
            "mueller_spielwaren-kw47-2026_p001", "spielwaren-kw47-2026", Path(PDF), 0
        ),
        SourcePage(
            "mueller_spielwaren-kw47-2026_p002", "spielwaren-kw47-2026", Path(PDF), 1
        ),
    ]


def test_image_pages_have_no_pdf_index() -> None:
    (page,) = extraction_pages(FLYERS, PAGES, "aldi-sued")
    assert page.pdf_page_index is None


def test_flyer_without_pages_is_an_error() -> None:
    with pytest.raises(ValueError, match="without pages"):
        extraction_pages(FLYERS, PAGES[2:], "mueller")
