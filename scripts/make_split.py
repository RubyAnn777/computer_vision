"""Write ``data/splits/split.csv``: one row per eligible page with its split.

Reads ``data/annotation_sample.csv`` (which set and flyer each page belongs
to) and ``data/page_screening.csv`` (whether the page is eligible), and
assigns train, val, test or unseen with `build_split`. The output is
committed and must not be regenerated after training has started.

Usage::

    uv run python scripts/make_split.py
"""

import csv
from collections import Counter
from pathlib import Path

from computer_vision.data.split import Split, build_split

SAMPLE = Path("data/annotation_sample.csv")
SCREENING = Path("data/page_screening.csv")
OUTPUT = Path("data/splits/split.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    """Build the split and write it."""
    split = build_split(read_csv(SAMPLE), read_csv(SCREENING))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["page_id", "split"])
        writer.writerows(split)
    counts = Counter(s for _, s in split)
    summary = ", ".join(f"{s.value} {counts[s]}" for s in Split)
    print(f"{len(split)} pages written to {OUTPUT} ({summary})")


if __name__ == "__main__":
    main()
