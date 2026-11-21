"""Select the pages to annotate and write ``data/annotation_sample.csv``.

Reads the download manifests in ``data/raw/`` and writes one row per page to
annotate:

- ``set = pool``: ``PER_RETAILER`` eligible pages per main-pool retailer, spread
  evenly over its flyer groups, drawn with `select_pages` and ``SEED``;
- ``set = unseen``: every page of the unseen-retailer flyers in ``UNSEEN_FLYERS``.

The output is committed. Running the script again on the same ``data/raw``
gives the same file byte for byte; the SHA-256 of ``pages.csv`` is stored in
every row so a changed input is visible.

Usage::

    uv run python scripts/select_annotation_pages.py
"""

import csv
import hashlib
from pathlib import Path

from computer_vision.data.ids import make_page_id
from computer_vision.data.sample import Candidate, select_pages

RAW = Path("data/raw")
OUTPUT = Path("data/annotation_sample.csv")
SEED = 2026
PER_RETAILER = 24
UNSEEN_FLYERS = {("mueller", "drogerie-kw47-2026")}
FIELDS = [
    "page_id",
    "set",
    "retailer",
    "flyer_slug",
    "flyer_group",
    "page_number",
    "source_path",
    "pdf_page_index",
    "seed",
    "pages_csv_sha256",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    """Draw the sample and write it."""
    flyers = {(f["retailer"], f["flyer_slug"]): f for f in read_csv(RAW / "flyers.csv")}
    pages = read_csv(RAW / "pages.csv")
    with (RAW / "pages.csv").open("rb") as handle:
        pages_sha = hashlib.file_digest(handle, "sha256").hexdigest()

    by_id: dict[str, dict[str, str]] = {}
    candidates = []
    for page in pages:
        key = (page["retailer"], page["flyer_slug"])
        page_id = make_page_id(*key, int(page["page_number"]))
        by_id[page_id] = page
        if flyers[key]["split_role"] == "annotation":
            candidates.append(
                Candidate(
                    page_id=page_id,
                    retailer=page["retailer"],
                    flyer_group=flyers[key]["flyer_group"],
                    eligible=page["suggested_eligible"] == "yes",
                )
            )

    chosen = [("pool", c.page_id) for c in select_pages(candidates, PER_RETAILER, SEED)]
    chosen += sorted(
        ("unseen", pid)
        for pid, p in by_id.items()
        if (p["retailer"], p["flyer_slug"]) in UNSEEN_FLYERS
    )

    rows = []
    for subset, page_id in chosen:
        page = by_id[page_id]
        rows.append(
            {
                "page_id": page_id,
                "set": subset,
                "retailer": page["retailer"],
                "flyer_slug": page["flyer_slug"],
                "flyer_group": flyers[(page["retailer"], page["flyer_slug"])][
                    "flyer_group"
                ],
                "page_number": page["page_number"],
                "source_path": page["path"],
                "pdf_page_index": page["pdf_page_index"],
                "seed": SEED,
                "pages_csv_sha256": pages_sha,
            }
        )
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    n_pool = sum(r["set"] == "pool" for r in rows)
    n_unseen = len(rows) - n_pool
    print(f"{n_pool} pool and {n_unseen} unseen pages written to {OUTPUT}")


if __name__ == "__main__":
    main()
