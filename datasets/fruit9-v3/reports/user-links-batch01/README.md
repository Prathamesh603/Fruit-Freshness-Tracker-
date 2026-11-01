# User dataset links — batch 1, 1 November 2026

All seven supplied projects are registered in `datasets/fruit9-v3/links.csv`. `review.json` preserves the original browse URLs, normalized project URLs, page findings and access limitations. No image archive was downloaded, exported or merged, and no training was started.

| Project | Page-level findings | Current decision |
|---|---|---|
| `emrah-fidan-pusay-uwwfc/apple-p92nm` | Page access blocked; apple suggested by title only | Register; verify class map, task, version and counts from authorized export |
| `thesis-ekum4/object-detection-for-thesis-rkr7v` | Object detection; 7,646 project images; six generic condition labels; CC BY 4.0 | Fruit species unverified; do not silently label everything banana |
| `object-detection-fmrts/citrus-object-detection-2` | Page access blocked | Citrus is not automatically orange or lemon; verify exported labels |
| `grid-air2x/guava-gvhzp` | Object detection; 1,594 images; Fresh_guava/Rotten_guava; publisher Public Domain | Useful extra-fruit candidate; guava is outside current nine-class map |
| `grid-air2x/grapes-mrlti` | Object detection; 1,360 images; Fresh_grapes/Rotten_grapes; publisher Public Domain | Grape candidate; export version/augmentation details still pending |
| `mango-ai/mango-project-cmofl` | Object detection; 3,370 images; fresh/rotten; CC BY 4.0 | Mango candidate; inspect actual images, boxes and dataset version |
| `test-51u0f/object-detection-study-ugj93` | Page access blocked; generic title | Fruit identity and annotation task unverified |

These are publisher website counts, not audited independent acquisitions or final training counts. Browse offsets such as `startingIndex=7600` are not image totals.

## Versions checked

- Thesis v1 page: https://universe.roboflow.com/thesis-ekum4/object-detection-for-thesis-rkr7v/dataset/1 — 7,546 files, split 5,264/1,525/757; 640 x 640 stretch; six classes remapped; no augmentation at this export. The seven listed version IDs are 1, 6, 13, 16, 20, 21, 22. The overview's seven versions does not mean version 7 exists. Source acquisition independence and fruit species remain unverified. Its v1 count/split resemble the previously found Banana Ripening Process project, so check for forks/overlap rather than counting both as new independent data.
- Guava v1 page: https://universe.roboflow.com/grid-air2x/guava-gvhzp/dataset/1 — 1,594 files, split 1,116/318/160; 640 x 640 stretch; no augmentation at this export. This does not establish that upstream source photos are unaugmented or independent.
- Grapes overview lists two versions; the linked version page could not be retrieved by the reader. No final version selected.
- Mango overview lists two dataset versions and references model/dataset number 5 in deployment text. The exact export must be selected from the dataset versions; do not infer it from the number of versions or the deployed model identifier.

## How these can be used

Fresh/rotten or ripe/unripe **bounding-box labels** can be mapped to a fruit identity for detector training. Keep original condition labels separately. The existence of those words does not replace checking that boxes tightly enclose real fruit, that all target fruit is labeled, and that the dataset is not merely classification photos with full-frame boxes.

Before merging, audit the actual YOLO ZIP: file pairing, class list, image readability, box geometry, original/augmented families and duplicates. Compare hashes and perceptual groups against our existing detection sources AND freshness datasets; the same source photos may have been reuploaded or relabeled. Rebuild group-aware splits rather than accepting publisher splits as independently validated.

Apple, citrus and the generic study project returned HTTP403 to a normal public-page request and were unavailable to the web reader. No restricted API or access-control workaround was used. Review their normal authorized YOLOv8 exports when available; do not guess labels or counts. There is no need to supply API keys in chat.

The current YOLO26s target map stays at nine fruits until an expanded scope is explicitly selected. Registering guava does not silently change that map, and the existing nine-fruit freshness classifier must not be applied to guava as if supported.
