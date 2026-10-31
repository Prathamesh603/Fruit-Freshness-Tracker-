# Start here: Fruit Lens project

This guide is new. Existing README files and the Classic snapshot are preserved.

Quick links: [Classic](apps/01-fruit-lens-classic/README.md) ·
[Custom app](apps/02-fruit-lens-custom/README.md) ·
[Model catalog](models/README.md) ·
[Detector results](detector_training/yolov8n-fruit9-v2/MODEL_CARD.md) ·
[Classifier results](classifier_training/mobilenetv3small-fruit9-v1/MODEL_CARD.md).

Latest Custom update: [v1.3 APK](apps/02-fruit-lens-custom/apk/Fruit-Lens-Custom-v1.3-arm64.apk)
with matching fruit IDs, colored boxes and per-fruit prediction scores. [Update notes](apps/02-fruit-lens-custom/CHANGELOG.md).

Classic with the same interface features: [Classic Plus APK](apps/01-fruit-lens-classic/apk/Fruit-Lens-Classic-Plus-v1.1-arm64.apk)
and [Classic Plus source/setup](apps/01-fruit-lens-classic/editions/02-classic-plus/README.md).
It keeps the original pretrained models, supports apple/banana/orange, and adds
IDs, scores, green/red boxes and a separate amber Unripe state. Its app ID is
`com.fruitlens.classic.plus`; original Classic and Custom remain separate.

## What we built, in order

1. **Original prototype:** `android/`. Existing public models proved that offline scanning works. Its package is `com.fruitlens.app`.
2. **Classic demonstration:** `apps/01-fruit-lens-classic/README.md`. Frozen NanoDet + SmartFruit MobileNetV2 app, package `com.fruitlens.classic`. Source, original weights, APKs, hashes and measured evidence stay together. Do not replace its models.
3. **Training data collection:** `datasets/fruit13/README.md` and `datasets/bohni-v13/README.md`. Mendeley sources label condition; box datasets train detection. Original downloads remain local, not in Git. Source attribution and actual counts are recorded. Not every image is independent; read the audits.
4. **First classifier experiment:** `ml/README.md`. MobileNetV3Small for fresh/rotten apple, banana and orange. Preserved six-class export in `model_exports/`.
5. **Detector experiments:** `detector_training/yolov8n-fruit9-v1/README.md` and `detector_training/yolov8n-fruit9-v2/MODEL_CARD.md`. Nine classes. v2 adds Bohni data but regresses on the original frozen test. The test composition matters; 79.01% and 58.68% are not a fair direct comparison.
6. **Nine-fruit classifier:** `classifier_training/mobilenetv3small-fruit9-v1/README.md` and `MODEL_CARD.md`. Eighteen joint classes, two conditions for each of nine fruit identities. Dataset test accuracy is 99.65%; this is not camera or complete-app accuracy.
7. **Custom demonstration:** `apps/02-fruit-lens-custom/README.md`. Separate installable app using YOLOv8n v2 and the nine-fruit classifier. Its package is `com.fruitlens.custom`; it can coexist with Classic. App language: Fresh, Not fresh, Uncertain.

## Where to find things

| Folder | Purpose |
|---|---|
| `apps/` | Named app editions, APKs and verification |
| `models/` | Versioned model catalog, original/trained weights, architecture summaries, hashes |
| `detector_training/` | Detector data splits, code, configurations, loss CSVs and evaluation |
| `classifier_training/` | Nine-fruit classifier training, export, loss CSVs and evaluation |
| `ml/` | Earlier six-class experiment, kept intact |
| `datasets/` | Local original data and tracked inventory/audit documents |
| `tools/` | Model catalog, deployment preparation and integrity helpers |
| `Training/` | Existing user workspace, left unchanged |

## Model versions and honest comparisons

`models/catalog.json` links the six exact versions. Each model folder has a README,
weight hashes, executable input/output rules and architecture information. Optimized
graph operations are not the same as original training layers; unknown original
parameter counts are marked unknown. A new checkpoint belongs in a new version
folder. Never rename a new model over an older one.

Both Custom models run on CPU. The detector was trained on GTX1650 CUDA; the
classifier was trained with native Windows TensorFlow CPU. Training backend and
phone inference backend are different things. Model sizes do not guarantee speed.

## Git and GitHub

Repository: https://github.com/Prathamesh603/Fruit-Freshness-Tracker-

- `classic-baseline` and its Classic tag identify the preserved baseline.
- `main` carries the latest verified Custom edition once publication completes.
- `feat/fruit-lens-custom` contains the catalog and the Custom app.
- Tags identify completed editions; check `git tag` and `git log --oneline --all`.
- Binary weights/APKs use Git LFS. Install Git LFS before cloning, then run `git lfs pull`.
- Raw datasets, SDKs, environments, build caches and temporary arrays stay out of Git.
- A commit is local until a successful push; publishing status is reported separately.

Build/setup details are in the new app's README. Existing READMEs were not rewritten
to hide earlier behavior or test limitations.
