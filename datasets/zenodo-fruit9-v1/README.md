# Zenodo nine-fruit detection dataset, version 1

This folder installs the nine-fruit subset of **Multi-Class Fruit and Vegetable Detection Dataset for Computer Vision**, Zenodo record 18904638, published 7 March 2026.

**Read `INSTALLATION_RESULTS.md` first.** Actual extraction found severe class imbalance. This collection is supplementary detection data, not a sufficient standalone nine-fruit training set.

Source: https://zenodo.org/records/18904638

This data teaches a detector **where a fruit is and which fruit it is**. It does not teach fresh/not-fresh classification. No freshness labels are invented here. Our separate freshness photos under `datasets/fruit13/raw` remain available for later classifier work.

## Labels

| New ID | Fruit | Original ID |
|---:|---|---:|
| 0 | Apple | 0 |
| 1 | Banana | 3 |
| 2 | Grape | 11 |
| 3 | Lemon | 12 |
| 4 | Mango | 15 |
| 5 | Orange | 16 |
| 6 | Pomegranate | 22 |
| 7 | Strawberry | 27 |
| 8 | Tomato | 28 |

The full source has 20,216 photos across 30 categories. We retain only photos with at least one valid box for these nine fruits, so the final subset has fewer images. Use `reports/summary.json` for actual installed counts, not the source headline count. Unsupported produce boxes are excluded from our nine-class labels; the original labels remain available separately for review.

## Folder guide

- `downloads/`: verified original ZIP. The full archive is retained to avoid repeated downloads, but the other 21 categories are not extracted as image folders.
- `prepared/images/train`, `val`, `test`: selected original image bytes, with short SHA256-derived filenames. Images are not resized or recompressed by this installer.
- `prepared/labels/`: corresponding five-field YOLO detection labels with the nine-class IDs above. Coordinates are normalized, not pixel units.
- `original-labels/`: original 30-class label text for the retained photos, to explain how each remapped annotation was derived.
- `quarantine/conflicting-annotations/`: photos whose duplicate copies disagree on target annotations. These are excluded from active train/validation/test folders until reviewed.
- `reports/download.json`: source URL, byte count, MD5/SHA256, and publisher checksum verification.
- `reports/images.csv`: original ZIP member, source split, output paths, image hash, dimensions, and selected box count for each retained photo.
- `reports/annotation-audit.json`: primary-pass class counts, rejected records, and exact duplicate/conflicting-label findings. Counts stop at the first invalid row in a rejected file; use `installation-verification.json` for the full archive's raw row counts.
- `reports/class-map.json`: original and new label maps.
- `reports/summary.json`: measured subset totals and checks still required before a trustworthy training experiment.
- `reports/status.json`: current download/preparation stage. This is the file to inspect while the installer is running.
- `source-data.yaml`: unmodified source class map.
- `data.yaml`: correct local dataset paths for Ultralytics. Regenerate if the project moves to another drive.
- `prepare_dataset.py`: reproducible, resumable downloader and subset preparation program.
- `quarantine_conflicts.py`: removes all conflicting exact-duplicate groups from active splits while retaining the photos separately.
- `verify_prepared.py`: checks installed image hashes, exact split separation, file/label inventories, box counts and class coverage.

## Installation

Requires Python with `requests`, `PyYAML`, and `Pillow`.

```powershell
python datasets/zenodo-fruit9-v1/prepare_dataset.py
python datasets/zenodo-fruit9-v1/verify_prepared.py
```

The archive must match **1,005,423,492 bytes** and publisher MD5 **394a7fb681366d22bd9e1c81d97884b3**. An interrupted download resumes from its `.partial` file. A wrong checksum stops preparation. ZIP member CRCs are checked when files are read. Only selected images are decoded/extracted; this is not a whole-archive CRC/image-quality audit.

Standard YOLOv11 export labels are ordinary detection labels usable by YOLOv8 too. The export name does not force a model architecture change. Source export preprocessing stretched photos to 512 x 512; this installer retains that source content. It does not undo the distortion or claim native high resolution.

## Quality boundaries

The installer rejects malformed/out-of-range boxes, missing images, unreadable photos, and exact matches to old validation/test source hashes where those hashes are recorded. It removes exact duplicate photos within this source, prioritizing the source test split, then validation, then training. Original annotations are retained for conflict review.

The publisher splits are retained for staging. They are **not automatically independent**: this collection combines many Roboflow projects. Before training, check augmented/source families and perceptual duplicates across these splits and against our earlier datasets and frozen evaluation photos. Exact hashes cannot identify resized/recompressed copies. Any duplicate-label conflicts and incomplete fruit boxes also need review. `training_ready: false` means these checks have not been completed; do not interpret a large source image count as proof of reliable accuracy.

Do not merge the larger 28,139-image collection without overlap checks. All 71 source URLs listed by this smaller collection are also listed by the larger one.

## License and model/app preservation

The Zenodo publication declares **CC BY 4.0** and names Mencia George Bayon and Lidia Sanchez Gonzalez, Universidad de Leon. Its upstream images come from multiple Roboflow collections; preserve the original source list and verify underlying permissions before redistribution. Publisher metadata and source URLs are preserved in `datasets/research/2026-10-31-user-dataset-review/`.

This installation does not train a model or replace any app. Classic, Classic Plus, Custom, all their weights and APKs, and previous measured results remain preserved. Training images are not bundled into the mobile application.
