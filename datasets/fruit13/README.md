# Fruit Lens: 13-fruit data collection

This folder collects data for the next version of Fruit Lens. It supports preparing
two different jobs: finding fruit in a photo, and assessing the visible condition
of a fruit crop. Downloading the data does not mean that either model has been trained.

The target fruits are apple, banana, grape, guava, jujube, lemon, lulo, mango,
orange, pomegranate, strawberry, tamarillo, and tomato. Tomato is included because
the publisher includes it. These are 13 fruit names, not 13 varieties of apple.

## Start here

1. Read `reports/STATUS.md` for actual counts and remaining gaps.
2. Read `SOURCES.md` for the origins, licenses, and limitations of each source.
3. Read `LABELS.md` before changing a label or drawing boxes.
4. Read `review/REVIEW_GUIDE.md` for the checks needed before training.
5. Read `HANDOFF.md` for reproducible commands and the next-agent checklist.
6. Read `reports/DETECTION_SOURCE_SEARCH.md` for newly found Roboflow and other
   detection candidates. They are documented separately from downloaded data.

## What each folder does

| Folder | Plain-language purpose |
|---|---|
| `raw/` | Original downloaded photos and archives. Keep these unchanged. |
| `sources/` | Publisher metadata, source folder names, original annotation tables, and attribution. These are the receipts for the data. |
| `classification/` | A table connecting each photo to its fruit and source condition label. It points to raw photos instead of making another image copy. |
| `detection/` | Tables describing real bounding boxes and the images they belong to. A box says where a fruit is, not where rot is. |
| `review/` | Sample sheets, conflicting labels, and work queues for a human reviewer. |
| `phone_eval/` | Reserved space for independently collected phone photos. No phone photos have been supplied yet. |
| `reports/` | Generated counts, hashes, quality findings, and the current completion status. |
| `scripts/` | Download, preparation, and verification programs. They do not train or modify an app. |

Every data folder has its own README. Image loaders should select image extensions;
they should not try to load README files as pictures.

## Current boundary

The freshness sources cover all 13 target fruit names. Detection boxes from the
collected sources cover only nine. Guava, jujube, lulo, and tamarillo still need
boxes. Even some of the nine covered fruits have very few detection examples.

The collection is a documented staging dataset, not a training-ready release.
Labels, repeated captures, and box quality need review. Final project train,
validation, and test splits are deliberately unassigned. No fabricated boxes,
synthetic freshness claims, or placeholder photos are included.

The preserved Classic app, its APKs, and the completed MobileNetV3-Small export
remain in their original folders. Training data is not bundled into a phone APK.
