# Annotation policy

The boxes are drawn using the provided bounding box tool and used in two ways.
Firstly as a hand-labelled ground-truth for single-class detector ("product").
Secondly as cropped tiles for the manual image collection used in later tasks.
Flyers have been collected using a Claude Agent, then they were reviewed manually.

## Scope

- Source: German supermarket flyers. Main pool: Kaufland, Aldi Süd, Penny
  and Norma. Unseen retailer: Edeka, used only for evaluation.
- One class only: `product`.

## Coordinates

- Boxes are stored as `xmin ymin xmax ymax` in integer pixels of the exact
  page image file. The origin is the top-left corner, x points right and
  y points down.
- Coordinates are pixel edges: a box's width is `xmax - xmin`, with no `+1`.
- Page images are never resized or re-encoded after annotation.

## Page eligibility (P)

- **P1** A page is excluded if its products appear only inside a styled
  scene (a laid table, a kitchen).
- **P2** Every excluded page is recorded in the screening file with the rule
  that excluded it, so that the exclusion rate can be reported.
- **P3** Pages to annotate are drawn from the eligible pages by a seeded
  sampling function, never chosen by hand.

## What is a product (R)

- **R1** A product is one food item offered for sale, in the form in which
  it is sold: a pack, bottle, can, jar, tray, or a loose pile.
- **R2** Each physically separate pack gets its own box, even when several
  packs or flavours share one price tag.
- **R3** A multipack sold as one unit (a shrink-wrapped six-pack, a gift
  box) gets one box around the whole unit.
- **R4** Loose food (fruit, vegetables, counter meat, cheese, bread) gets one
  box around the visible pile that belongs to one offer. If the same offer
  appears as two separate piles, each pile gets its own box.
- **R5** Prepared or cooked food shown as a serving suggestion is not a
  product. Box only the pack next to it. A plated dish with no pack is not
  boxed.
- **R6** Contents shown spilling out in front of a pack (nuts, pasta) are not
  part of the box, unless the loose food itself is the offer (R4).
- **R7** Never boxed: price tags and badges, logos, text,
  quality seals, icons, people, props (plates, boards, cutlery).

## Box geometry (G)

- **G1** The box is the smallest rectangle that contains all visible pixels
  of the product (a modal box). Shadows and reflections are excluded.
- **G2** If a price badge or another product covers part of a product, the
  box still spans the product's visible extent. Occluders inside the box are
  tolerated.
- **G3** A product more than about 50% hidden, by occlusion or by the page
  edge, is not boxed. The estimate is made by eye and not measured.
- **G4** Overlapping products each get their own box. Boxes may overlap.
- **G5** A product whose shorter box side would be under 20 px on the
  standardised page image is not boxed.
