# Label meanings and numbering

## Fruit names

Use these 13 lowercase names in this exact order. Detection class IDs are zero-based:

0 apple, 1 banana, 2 grape, 3 guava, 4 jujube, 5 lemon, 6 lulo, 7 mango,
8 orange, 9 pomegranate, 10 strawberry, 11 tamarillo, 12 tomato.

The machine-readable list is `labels.json`. These are planned class IDs, not evidence
that a trained detector already recognizes them.

## Freshness labels

`fresh` means the source publisher placed the photo in its fresh folder.
`spoiled` combines the source words rotten and spoiled for inventory purposes.
This mapping is recorded in every row through `source_folder` and `label_origin`.
It still needs visual review before training. It is not a laboratory safety label.

The proposed classifier has 26 outputs, ordered as all 13 `fresh_` labels followed
by all 13 `spoiled_` labels. Fruit identity and condition are also separate columns
so a later multi-head model can reuse the inventory without parsing strings.

Do not map unripe, overripe, bruised, cut, moldy, and chemically treated fruit to the
same condition without a written annotation rule. The boxed GitHub source's
`normal` condition remains unchanged and is not mapped to `fresh`.

## What the labels cannot tell us

A photograph may show only one side of a fruit. A brown banana peel alone does
not establish the condition of its flesh. Image labels do not establish remaining
shelf life, internal freshness, contamination, or whether food is safe to eat.

## Unknowns and unsuitable photos

During review, mark uncertain cases as `uncertain`, and unsuitable photos as
`reject`; record a reason. Do not quietly force them into the nearest class.
An app-level abstention policy must later be evaluated using unsupported fruits,
non-fruit objects, bad lighting, and blur. A confidence threshold alone is not a
validated unknown-object detector.
