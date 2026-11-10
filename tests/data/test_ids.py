import pytest
from hypothesis import given
from hypothesis import strategies as st

from computer_vision.data.ids import (
    MAX_PAGE_NUMBER,
    PageKey,
    make_page_id,
    parse_page_id,
    slugify,
)

slugs = st.from_regex(r"[a-z0-9]+(-[a-z0-9]+)*", fullmatch=True)
page_numbers = st.integers(min_value=1, max_value=MAX_PAGE_NUMBER)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ALDI SÜD", "aldi-sued"),
        ("Kaufland", "kaufland"),
        ("PENNY", "penny"),
        ("Edeka Märkte", "edeka-maerkte"),
        ("2026-44_EDe", "2026-44-ede"),
        ("DE_de_KDZ1_3000_D46", "de-de-kdz1-3000-d46"),
        ("  Weihnachten 2026! ", "weihnachten-2026"),
        ("Straße", "strasse"),
        ("Café", "cafe"),
    ],
)
def test_slugify_examples(text: str, expected: str) -> None:
    assert slugify(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "!!!", "_"])
def test_slugify_rejects_text_without_letters_or_digits(text: str) -> None:
    with pytest.raises(ValueError, match="slug"):
        slugify(text)


def test_make_page_id_example() -> None:
    assert make_page_id("ALDI SÜD", "kw46-26-op-mp", 14) == (
        "aldi-sued_kw46-26-op-mp_p014"
    )


@pytest.mark.parametrize("page_number", [0, -1, MAX_PAGE_NUMBER + 1])
def test_make_page_id_rejects_out_of_range_pages(page_number: int) -> None:
    with pytest.raises(ValueError, match="page_number"):
        make_page_id("penny", "kw46", page_number)


def test_lexicographic_order_equals_page_order() -> None:
    ids = [make_page_id("penny", "kw46", n) for n in range(1, 121)]
    assert sorted(ids) == ids


@given(retailer=slugs, catalog=slugs, page_number=page_numbers)
def test_parse_inverts_make(retailer: str, catalog: str, page_number: int) -> None:
    page_id = make_page_id(retailer, catalog, page_number)
    assert parse_page_id(page_id) == PageKey(retailer, catalog, page_number)


@pytest.mark.parametrize(
    "page_id",
    [
        "penny_kw46_p14",
        "penny_kw46_p0014",
        "penny_p014",
        "PENNY_kw46_p014",
        "aldi_sued_kw46_p001",
    ],
)
def test_parse_rejects_malformed_ids(page_id: str) -> None:
    with pytest.raises(ValueError, match="page id"):
        parse_page_id(page_id)
