"""Select the pages to annotate and write ``data/annotation_sample.csv``.

This is stage 1 of the data design: it decides which pages are annotated.
Every flyer in ``data/raw/flyers.csv`` has a ``split_role``:

- ``pool``: weekly Müller flyers. ``POOL_PER_FLYER`` pages per flyer are drawn
  with `select_pages` and ``SEED``. These pages are later split into train and
  val by whole flyer (stage 2, the split script).
- ``test``: the Müller flyer used as the in-distribution test set. Every page
  is annotated.
- ``unseen``: supermarket flyers (retailers never used for training or tuning).
  ``UNSEEN_PER_RETAILER`` pages per retailer are drawn, spread evenly over its
  flyer weeks.
- ``extraction``: Müller catalogues that are never annotated. Together with
  the pool pages that were not drawn, they are the input for the automatically
  extracted tile collection.

The output is committed. Running the script again on the same ``data/raw``
gives the same file byte for byte; the SHA-256 of ``pages.csv`` is stored in
every row so that a changed input is visible.

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
POOL_PER_FLYER = 10
UNSEEN_PER_RETAILER = 24
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
    pages_sha = hashlib.sha256((RAW / "pages.csv").read_bytes()).hexdigest()

    by_id: dict[str, dict[str, str]] = {}
    candidates: dict[str, list[Candidate]] = {"pool": [], "unseen": []}
    test_ids: list[str] = []
    for page in pages:
        key = (page["retailer"], page["flyer_slug"])
        role = flyers[key]["split_role"]
        page_id = make_page_id(*key, int(page["page_number"]))
        by_id[page_id] = page
        if role == "test":
            test_ids.append(page_id)
        elif role in candidates:
            candidates[role].append(
                Candidate(
                    page_id=page_id,
                    retailer=page["retailer"],
                    flyer_group=flyers[key]["flyer_group"],
                    eligible=page["suggested_eligible"] == "yes",
                )
            )

    n_pool_flyers = len({c.flyer_group for c in candidates["pool"]})
    pool = select_pages(candidates["pool"], POOL_PER_FLYER * n_pool_flyers, SEED)
    unseen = select_pages(candidates["unseen"], UNSEEN_PER_RETAILER, SEED)
    chosen = (
        [("pool", c.page_id) for c in pool]
        + [("test", pid) for pid in sorted(test_ids)]
        + [("unseen", c.page_id) for c in unseen]
    )

    rows = []
    for subset, page_id in chosen:
        page = by_id[page_id]
        key = (page["retailer"], page["flyer_slug"])
        rows.append(
            {
                "page_id": page_id,
                "set": subset,
                "retailer": page["retailer"],
                "flyer_slug": page["flyer_slug"],
                "flyer_group": flyers[key]["flyer_group"],
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
    counts = {s: sum(r["set"] == s for r in rows) for s in ("pool", "test", "unseen")}
    print(f"{counts} pages written to {OUTPUT}")


if __name__ == "__main__":
    main()
