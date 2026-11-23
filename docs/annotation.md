# Annotation policy

> **Frozen.** Version 1.3, locked on 2026-11-21 after the 5-page pilot.
> These rules apply unchanged to every page in pass 1 and pass 2. Any later
> change requires re-checking every page already annotated and a new entry
> in the change log.

The boxes are drawn with the provided bounding box tool (`bbox_tool.html`) and
used in two ways: first, as hand-labelled ground truth for a single-class
detector ("product"), and second, cropped, as tiles for the manual image
collection used in later tasks. The flyers were collected with a Claude agent
and then reviewed manually. Rules are numbered so that the report can cite
them.

## Scope

- Source: German supermarket flyers. Main pool: Kaufland, Aldi Süd and
  Penny. Unseen retailer: Müller (drugstore), used only for evaluation and
  never for training, model selection or threshold tuning.
- One class only: `product`.

## Coordinates

- Boxes are stored as `xmin ymin xmax ymax` in integer pixels of the exact
  page image file. The origin is the top-left corner, x points right and
  y points down.
- Coordinates are pixel edges: a box's width is `xmax - xmin`, with no `+1`.
- Page images are never resized or re-encoded after annotation, because the
  boxes are only valid for that exact file.

## Page eligibility (P)

- **P1** A page is excluded if its products appear only inside a styled
  scene (a laid table, a kitchen, a living room, people shopping or cooking).
  A styled scene next to separate packshots does not exclude the page: the
  packshots are boxed and the scene is not (R5).
- **P2** Every sampled page is screened before annotation and recorded in
  `data/page_screening.csv`, eligible pages included, so that the exclusion
  rate can be reported per rule and retailer.
- **P3** Pages to annotate are drawn by the seeded function `select_pages`
  in `scripts/select_annotation_pages.py` (seed 2026). Within each retailer,
  the quota of 24 pages is spread evenly over the flyer weeks. Within a week,
  the eligible pages with the smallest SHA-256 keys of `"<seed>:<page_id>"` are
  taken, which is uniform sampling without replacement. The unseen set is
  every page of one Müller flyer. Pages are never chosen by hand. Discarded
  pages are not replaced.
- **P4** A page is excluded if it depicts no item for sale at all: app,
  loyalty, newsletter or messenger adverts, service tariffs, icon-only pages.
  A page with at least one depicted product stays eligible, however few or
  large its products.
- **P5** A page is excluded if it repeats, with the same layout and the same
  products, a page from an earlier flyer week in the sample. The earlier copy
  is kept and the excluded page names it in `duplicate_of`.
- **P6** A page is excluded if its image is defective (blank, cut, corrupted
  or the wrong page).

## What is a product (R)

- **R1** A product is one item offered for sale, food or non-food, in the
  form in which it is sold: a pack, bottle, can, jar, tray, box, a single
  object, or a loose pile.
- **R2** Each physically separate pack gets its own box, even when several
  packs or flavours share one price tag.
- **R3** A multipack sold as one unit (a shrink-wrapped six-pack, a gift
  box) gets one box around the whole unit.
- **R4** Loose food (fruit, vegetables, counter meat, cheese, bread) gets one
  box around the visible pile that belongs to one offer. If the same offer
  appears as two separate piles, each pile gets its own box.
- **R5** Prepared or cooked food shown as a serving suggestion is not a
  product. Only the pack next to it is boxed. A plated dish with no pack is not
  boxed.
- **R6** Contents shown spilling out in front of a pack (nuts, pasta) are not
  part of the box, unless the loose food itself is the offer (R4). if non food items
  are shown in front of the box, they are included.
- **R7** Never boxed: price tags and badges, logos, text, quality seals,
  icons, people, props (plates, boards, cutlery).

## Box geometry (G)

- **G1** The box is the smallest rectangle that contains all visible pixels
  of the product (a modal box). Shadows and reflections are excluded.
- **G2** If a price badge or another product covers part of a product, the
  box still spans the product's visible extent. Occluders inside the box are
  tolerated.
- **G3** A product more than about 50% hidden, by occlusion or by the page
  edge, is not boxed. The estimate is made by eye and not measured.
- **G4** Overlapping products each get their own box. Boxes may overlap.

## Process

1. **Screening.** Every sampled page is screened against the rules and
   logged in `data/page_screening.csv`. The image of an excluded page is
   renamed with a `D_` prefix in `data/interim/pages/annotate/`, so it is
   visibly skipped in the tool.
2. **Pass 1 (ground truth).** Every eligible page is annotated in
   `bbox_tool.html`, with images from `data/interim/pages/annotate/` and
   annotations saved to `data/annotations/pass1/`. Every eligible page
   contains at least one product (P4), so a page without a `.txt` file is
   not yet annotated.
3. **Pilot.** After the first 5 pages, this policy was reviewed and frozen as
   v1.3 on 2026-11-21 (see the note at the top).
4. **Pass 2 (consistency check).** At least 3 days after pass 1, 10 seeded
   pages (2 per retailer, Müller included) are re-annotated into
   `data/annotations/pass2/` without looking at pass 1. Pass 2 only measures
   annotation consistency (mean IoU of matched boxes and agreement F1 at
   IoU 0.5).

## Log files

`data/page_screening.csv`, one row per sampled page to track exclusion:

| Column | Meaning |
| --- | --- |
| `page_id` | Canonical page id |
| `set` | `pool` or `unseen` |
| `decision` | `eligible` or `exclude` |
| `rule` | `P1`, `P4`, `P5` or `P6`; empty if eligible |
| `duplicate_of` | Page id of the kept copy; only for `P5` |
| `note` | What is on the page, one phrase |
| `screened_on` | Date |
| `policy_version` | Policy version used |

## Change log

| Version | Date | Change |
| --- | --- | --- |
| 1.0 | 2026-11-10 | First draft. |
| 1.1 | 2026-11-10 | Non-food included; three-offer page rule removed. |
| 1.2 | 2026-11-21 | Added P4 (no product), P5 (duplicate), P6 (defect) and the "never a reason" list; P3 rewritten to match the code; no replacement of discarded pages; R1 restored to food or non-food (needed for Müller); added Process, log files and change log. No page had been annotated, so nothing needed re-checking. |
| 1.3 | 2026-11-21 | G5 (minimum box size of 20 px) removed: every product that passes G3 is boxed, whatever its size. No separate overlap rule: G2 covers occluders. Annotation log and edge-case log dropped; `page_screening.csv` is the only log. Pass 2 gap set to at least 3 days. Policy frozen after the 5-page pilot. |
