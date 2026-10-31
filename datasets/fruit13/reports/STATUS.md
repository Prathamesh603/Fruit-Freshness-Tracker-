# Collection status

This is a collected and inventoried staging dataset. It is NOT a training-ready
release and no 13-fruit model has been trained from it.

## Measured files

- Classification: **19,200 source images**, **13 fruit names**, **26 fruit-condition combinations**.
- Classification unreadable images: **0** (JPEG draft readability check).
- Exact duplicate copies in classification inventory: **1280**.
- Perceptual candidate label conflicts: **145 groups / 664 images**. These need review, not automatic relabelling.
- After withholding duplicate copies and candidate conflicting groups: **17,282 classification candidates**, with **1,918 images held for review**. This is automatic staging, not human approval.
- Images overlapping the prior three-fruit experiment by SHA-256: **1197**.
- Boxed GitHub source: **3,202 readable images / 7,087 boxes**, across three fruit types; **0 recorded geometry/readability issues**.
- Open Images: **413 downloaded images / 1,462 boxes**, across nine fruit types; **0 download/box failures**.
- Counts are source-file counts before deduplication, filtering, crop generation, or final split selection. They are not counts of independent physical fruit.
- No trained model, final project split, independent phone test, or all-fruit support is claimed.

## Coverage

| Fruit | Fresh images | Spoiled images | GitHub boxes | Open Images boxes |
|---|---:|---:|---:|---:|
| apple | 200 | 200 | 1987 | 128 |
| banana | 1200 | 1200 | 3041 | 26 |
| grape | 200 | 200 | 0 | 158 |
| guava | 200 | 200 | 0 | 0 |
| jujube | 200 | 200 | 0 | 0 |
| lemon | 1000 | 1000 | 0 | 190 |
| lulo | 1000 | 1000 | 0 | 0 |
| mango | 1000 | 1000 | 0 | 13 |
| orange | 1200 | 1200 | 2059 | 176 |
| pomegranate | 200 | 200 | 0 | 18 |
| strawberry | 1200 | 1200 | 0 | 330 |
| tamarillo | 1000 | 1000 | 0 | 0 |
| tomato | 1000 | 1000 | 0 | 423 |

Different sources can overlap. Do not add these columns together and call them
independent training examples. Box counts also include source group/depiction
flags that must be filtered according to a written annotation policy.

## Remaining work

1. Review condition labels and candidate conflict groups. Keep decisions separate from raw data.
2. Establish same-fruit/session groups where possible; pHash cannot prove independence.
3. Draw real fruit boxes for **guava, jujube, lulo, tamarillo**. The work queue contains 4,800 candidate source photos with blank box fields.
4. Add independent detection photos for sparse classes; the small Open Images subset is a seed collection.
5. Resolve GitHub boxed image provenance and review the original VOC coordinate convention.
6. Deduplicate across detection, classification, and any derived crops before making fixed splits.
7. Collect phone evaluation photos; that folder is currently a documented empty collection area.

See `classification_audit.json`, `detection_boxes_audit.json`, `openimages_audit.json`,
`downloads.json`, `coverage.csv`, and `image_dimensions.json` for machine-readable evidence.
