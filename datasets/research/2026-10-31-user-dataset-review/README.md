# Review of the six datasets suggested by the user

Checked on 31 October 2026. This folder contains research evidence, not a new training dataset. No app, model weights, or training configuration was changed.

## Recommendation for our app

The 20,216-image Zenodo collection is the strongest candidate in this list for expanding our current nine-fruit detector. Its actual class file includes all nine fruits. That makes it a candidate for an experiment, not proof that a trained model will beat our current detector.

Keep the first experiment focused on apple, banana, grape, lemon, mango, orange, pomegranate, strawberry, and tomato. Add selected Open Images training photos and our own phone photos to improve backgrounds, lighting, crowded scenes, and confusing objects. Resolve apple/tomato confusion, orange-colored false detections, and missed bananas before expanding the label set.

These datasets do not provide a complete, consistently labeled freshness dataset for our nine fruits. Fruit identity, ripeness, bruising, and freshness are different labels. Do not create freshness labels just because a fruit looks normal in a detection photo.

## What was actually inspected

The original publication pages and repository documentation were checked. For the three ZIP archives, bounded HTTP byte-range requests read their ZIP file inventories and small class/configuration files without downloading the image collections. Extracted configuration files were checked through Python's ZIP reader, including their member CRCs. The saved audit JSON files contain byte counts and metadata hashes.

This confirms archive sizes, listed image/label file counts, and class maps. It does NOT confirm every box is correct, independently recount every annotation, inspect every image, prove image/label filename pairing across the whole archive, establish class balance in the Zenodo datasets, verify the full archive checksum, or establish freedom from duplicate images. Publisher annotation totals below are identified as publisher claims.

Approximately 16.3 MB of ZIP metadata was transferred across the three archives. No full image archive was downloaded. Website/API metadata adds a small amount beyond that figure. Counts describe the checked versions; repositories may change later.

## 1. Multi-Class Fruit and Vegetable Detection Dataset

Source: https://zenodo.org/records/18904638

- Published 7 March 2026, version 1, by Mencia George Bayon and Lidia Sanchez Gonzalez, Universidad de Leon.
- ZIP: `AgricultureDataset.zip`, 1,005,423,492 bytes, approximately 1.01 GB in decimal units.
- Uncompressed member sizes total 1,003,405,058 bytes, before filesystem overhead.
- Image files: 20,216. Train 15,944; validation 2,582; test 1,690. Each split has the same number of label files as image files.
- Publisher reports 222,301 annotations; this review did not recount those boxes.
- Actual class count: 30. Standard YOLO bounding-box format, exported as YOLOv11. Ordinary YOLO detection labels can be adapted for YOLOv8; the export name does not require training YOLOv11.
- Export README says 512 x 512 stretch resize and no augmentation at export. That does not prove upstream images were never augmented.
- Publication license: CC BY 4.0. It is a compilation of 71 listed Roboflow sources; preserve attribution and check the underlying image/source terms before distribution.

Actual zero-based class map from `zenodo20216/00_data.yaml`:

```text
0 Apple          10 Fig           20 Pineapple
1 Apricot        11 Grape         21 Plum
2 Aubegine       12 Lemon         22 Pomegranate
3 Banana         13 Lime          23 Quince
4 Bean           14 Lychee        24 Radish
5 Carrot         15 Mango         25 Rice
6 Cherry         16 Orange        26 Salad
7 Courgette      17 Peach         27 Strawberry
8 DragonFruit    18 Pear          28 Tomato
9 Durian         19 Pepper        29 Watermelon
```

`Aubegine` is the spelling in the source class map; normalize deliberately if importing it. Thirty categories does not mean thirty fruit species.

All nine current fruit classes are present. Guava, jujube, lulo, and tamarillo from our planned thirteen-fruit scope are absent. Other notable missing fruits include papaya, kiwi, coconut, chikoo, and custard apple. Pineapple is present.

The YAML contains relative paths such as `../train/images`; check and rewrite paths for the actual extracted layout. Do not launch training against unchecked paths.

## 2. Multi-Class Fruit, Vegetable and Agricultural Object Detection

Source: https://zenodo.org/records/22544419

- Published 6 September 2026, version 1, same authors.
- ZIP size: 7,648,225,279 bytes, approximately 7.65 GB.
- Uncompressed member sizes: 7,664,187,742 bytes, approximately 7.66 GB.
- Image inventory confirms 28,139 images: train 22,334; validation 3,672; test 2,133. Label-file counts match split image counts.
- Publisher reports 261,631 annotations, not independently recounted here.
- Actual class count: **36**, extracted from its YAML.
- YOLOv11 export; no augmentation at export according to its README.
- Publication license: CC BY 4.0, with the same upstream attribution/provenance review needed for compiled sources.

Actual class order:

```text
Apple, Apricot, Aubergine, Banana, Bean, Carrot, Cherry, Courgette,
DragonFruit, Durian, Fig, Grape, Lemon, Lime, Lychee, Mango, Orange,
Peach, Pear, Pepper, Person, Pineapple, Plum, Pomegranate, Post,
Pothole, Puddle, Quince, Rice, Row, Salad, Strawberry, Tomato,
Trace, Trunk, Watermelon
```

Compared with the smaller collection, it drops Radish, corrects Aubergine's spelling, and adds Person, Post, Pothole, Puddle, Row, Trace, and Trunk. These additional categories are not extra fruits.

It lists 109 Roboflow source URLs, including all 71 URLs listed by the smaller collection. This proves overlap in listed sources, not an exact number of shared photos. Do not merge the collections and trust their original train/test splits without deduplication.

The archive has three root `data.yaml` entries with identical content. The snapshots retain each occurrence. It contains the current nine fruits but still lacks guava, jujube, lulo, and tamarillo.

Verdict: not our first download. The extra size and irrelevant classes are not evidence of better mobile fruit performance.

## 3. Open Images V7

Sources:

- https://storage.googleapis.com/openimages/web/factsfigures_v7.html
- https://storage.googleapis.com/openimages/web/download_v7.html
- https://docs.voxel51.com/dataset_zoo/datasets/open_images_v7.html

The overall collection has roughly nine million images, but approximately 1.9 million have bounding-box annotations. Do not interpret nine million as nine million box-labeled photos. The commonly stated 600 box categories are a rounded figure; the official boxable class-description CSV contains 601 entries.

The downloaded official boxable class map confirms all nine current target fruits. It does not contain the four missing planned fruits under their names. The class CSV is stored as `openimages_boxclasses.txt`.

Use a capped, seeded download of the relevant TRAINING split rather than the whole collection. In the pasted FiftyOne example, `max_samples=5000` is a total sample limit, not 5,000 examples for each fruit. Filtering to the nine classes can discard labels for other visible objects; retain a record of all source annotations and explicitly define ignored objects.

Our existing detector data already includes 413 Open Images photos from an earlier V5 validation selection. Preserve their identities and existing holdouts; do not put the same images back into a new training set.

Annotations are CC BY 4.0; image licensing/attribution must be tracked per image. This is our most useful supplementary source in the list for varied real-world backgrounds. Add phone-camera failure cases and confusing nonfruit objects too.

## 4. Fruits-360

Current repository READMEs, rather than older organization-page summaries:

- https://github.com/fruits-360/fruits-360-100x100 — README version 2026.10.27.0: **200,201 images, 278 classes**, train 150,179, test 50,022.
- https://github.com/fruits-360/fruits-360-original-size — README version 2026.10.26.0: **119,807 images, 164 classes**, train 59,982, validation 30,001, test 29,824.
- https://github.com/fruits-360/fruits-360-multi — README version 2026.09.20.0: **919 images**.

These are publisher README counts, not a full image inventory independently downloaded in this review. The pasted 188,346/265, 107,952/150, and 724 figures refer to older releases.

Many categories are varieties, vegetables, nuts, seeds, or mushrooms. They are not 278 distinct fruit species. Main classification photos use controlled backgrounds and rotating specimens. Related views of the same physical fruit are highly correlated; split by physical specimen or capture group for independent evaluation where that information is available.

License: **CC BY-SA 4.0**, which differs from CC BY 4.0. The main collection is classification data, not a ready multi-object bounding-box dataset. The existence of multi-object photos alone does not establish ready detection labels; these were not verified here. It also does not supply a consistent freshness target for our nine-fruit classifier.

Verdict: optional fruit identity/variety classification research, not the main fix for our detector's background false positives or our freshness classifier's generalization.

## 5. deepNIR fruit/crop detection collection

Sources:

- https://inkyusa.github.io/deepNIR_dataset/overview/obj-detection/
- https://inkyusa.github.io/deepNIR_dataset/download/obj-detection/
- https://zenodo.org/records/6324489
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9269522/

The detection ZIP `yolov5.zip` is 719,274,932 bytes, approximately 0.72 GB; its uncompressed members total 726,219,678 bytes. License: CC BY 4.0.

The ZIP inventory contains 4,295 image files and corresponding label-file counts: train 3,434; validation 430; test 431. Images by source crop folder:

| Crop | Images |
|---|---:|
| Apple | 63 |
| Avocado | 54 |
| Blueberry | 78 |
| Capsicum | 122 |
| Cherry | 154 |
| Kiwi | 125 |
| Mango | 170 |
| Orange | 57 |
| Rockmelon | 58 |
| Strawberry | 41 |
| Wheat | 3,373 |

Wheat accounts for 78.5% of this archive. Eleven crops does not mean eleven balanced fruit classes. Only apple, mango, orange, and strawberry overlap our nine targets: **331 images total** in those four folders.

Each crop uses its own single-class YAML (`nc: 1`, class name `1`) and machine-specific absolute paths. Class zero means different things in different crop folders. Remap IDs using the enclosing crop folder and replace paths before merging.

The small detection download is separate from the much larger synthetic near-infrared collections. Our RGB phone detector does not need those large NIR archives. Verdict: selected orchard examples may help; downloading the whole wheat-dominated collection is not a priority.

## 6. FruitDet community annotation repository

Sources:

- https://huggingface.co/datasets/sirunchained/fruit-dataset-annotation
- https://huggingface.co/api/datasets/sirunchained/fruit-dataset-annotation/tree/main?recursive=true&limit=1000
- https://huggingface.co/datasets/sirunchained/fruit-dataset

The annotation repository card claims 19 categories with 30 images each, implying 570 images. The inspected repository tree actually contains **150 JPGs with 150 matching annotation paths**, across **five populated categories**: apple, avocado, banana, blackberry, and carrot, 30 images each.

The card's category list has 18 names, not 19. Thirteen named directories contain placeholders rather than photos: cherry, grape, kiwi, lemon, onion, orange, papaya, peach, pear, pepper, raspberry, strawberry, tomato. Only apple and banana currently overlap our nine targets in this annotation snapshot.

The Hugging Face viewer's 934 text rows are not 934 images. Each populated crop's `classes.txt` defines a local class map; remap folder-local zero IDs before combining.

Its linked original image repository is a different repository. Its web viewer reports 210 images and seven classes; this review did not inventory its complete original tree or verify boxes there. That does not establish the missing annotated 570-image release. Keep the annotation-repository finding separate from the original image collection.

The annotation card declares CC BY 4.0; original-photo provenance remains to be checked. Verdict: too small and incomplete in its present annotated form to be our main dataset. Preserve the saved API snapshot because the repository can change.

## Storage and an experiment that would be useful

At the end of this review, D: had approximately 2.04 billion free bytes, or 1.90 GiB. Free space changes as other programs run.

The smaller Zenodo ZIP plus its unpacked members alone needs approximately **2.009 billion bytes**, before filesystem overhead, training caches, checkpoints, or logs. That would almost fill D:. The 7.65 GB archive cannot fit. Do not start a full download/extraction on this drive in its current state. Use another drive or a deliberately bounded subset, and budget actual free space before starting.

Recommended next experiment:

1. Establish a storage budget and choose where images and checkpoints will live.
2. Inspect per-class box counts and visual examples in the 20,216-image collection. Audit missing boxes, grouped fruit vs individual-fruit labeling, blurry fruit, tiny fruit, and distorted stretch resizing. A nine-class subset will have fewer than 20,216 images unless every image contains a target fruit; its actual size is not known yet.
3. Deduplicate against Bohni, our existing detection data, Open Images IDs, and all held-out evaluation photos. Group augmented variants, source images, and phone capture sessions together before splitting. Preserve existing independent holdouts.
4. Build a nine-class training subset with one explicit label map. Audit images containing removed categories and partially annotated target fruit; do not silently treat unlabeled target fruit as background.
5. Add capped Open Images training examples and hard negatives addressing current failures. A phone-only holdout should include single fruit, piles, hands, packaging, shadows, colored squares, and similar-looking fruits.
6. Train a separate detector version, then compare per-fruit precision, recall, mAP, false positives on nonfruit scenes, crowded-scene misses, model size, and matched phone latency against the existing model. A larger dataset does not guarantee an improvement or change the runtime model size by itself.
7. Keep freshness training separate until there are suitable labels, source/session splits, and phone validation. A new fruit detector class must not be routed through the nine-fruit freshness head as if supported. Show freshness unavailable/uncertain for unsupported classes.

An extra variety classifier is optional. The existing detector-to-crop-to-freshness pipeline is sufficient for the current product scope; adding another model needs an actual measured benefit.

## Files in this folder

- `*_archive_audit.json`: verified archive sizes, entry counts, selected metadata hashes, and limits of verification.
- `zenodo20216/`, `zenodo28139/`, `deepnir/`: archive member inventories and the small original metadata files read remotely. These are not image datasets.
- `zenodo20216.json`, `zenodo28139.json`, `deepnir.json`: original Zenodo API responses for provenance.
- `zenodo*_sources.json`: listed Roboflow source URLs used for source-overlap checks.
- `fruitdet.json`, `fruitdet_metadata.txt`, `fruitdet_readme.txt`, `fruitdet_*_classes.txt`: annotation repository snapshots and class maps.
- `openimages_boxclasses.txt`: official boxable category descriptions.
- `inspect_remote_archives.py`: bounded byte-range inspection utility; refuses full responses and oversized reads.
- `archive-inspection.log`: raw inspection output.

Evidence remains metadata-level unless a specific content check is explicitly stated above. This review does not claim a trained accuracy or latency improvement.
