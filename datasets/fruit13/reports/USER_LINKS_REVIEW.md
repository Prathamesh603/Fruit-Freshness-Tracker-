# Review of the additional user-provided detection sources

Checked 2026-10-30. Only pages, class lists, configurations and small file metadata
were inspected. No model weights or dataset images were downloaded or merged.

Follow-up: the user subsequently supplied the Bohni YOLOv8 ZIP. Its actual audit
is documented in `../../bohni-v13/reports/STATUS.md`; the public-page access findings
below describe the earlier search, not a remaining request for that ZIP. A separate
YOLO11m reference benchmark is in `../../../detector_training/reference-yolo11m-bohni/README.md`.

## LVIS-derived GitHub dataset

https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset

The README reports 8,221 images, 63 categories, and another 180 manually annotated
test images. These are fruit/vegetable/food categories, not 63 different fruit types.
It supplies YOLO labels and YOLOv8 medium/large/extra-large baseline links.

The inspected class list contains seven of our thirteen targets: apple, banana,
grape, lemon, orange, strawberry and tomato. It does not list mango, pomegranate,
guava, jujube, lulo or tamarillo. No freshness-condition labels are supplied.

Critical preparation findings: the checked data.yaml sets both val and test to
images/test. Do not use that configuration unchanged for independent evaluation.
The README also acknowledges incorrect/duplicate uppercase Strawberry and Tomato
categories and strong class imbalance. Clean the label map and inspect annotations.
The repository lists MIT; retain and inspect the underlying LVIS/image provenance
rather than treating a repository software license as proof of every image's terms.

Class list: https://raw.githubusercontent.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset/main/fruits_vegetables_LVIS_dict.json
Configuration: https://raw.githubusercontent.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset/main/LVIS_Fruits_And_Vegetables/data.yaml

## Hugging Face YOLO11 models

| Model | Weight file bytes, checked via public file API | Model-card metrics | Listed license tag |
|---|---:|---|---|
| https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector | 40,565,868 | mAP50 0.925, mAP50-95 0.691 | AGPL-3.0 |
| https://huggingface.co/Aniket2003333333/fruit-veg-yolo11l-detector | 51,245,849 | mAP50 0.913, mAP50-95 0.689 | Apache-2.0 |

Both cards list 35 produce classes and a 640-pixel input. They identify the Bohni
Tech Fruits and Vegi dataset; the medium card explicitly names version 13. These
are reported metrics on the uploaders' protocols, not verified results on our
held-out split or evidence of Android latency. The public model tags differ;
upstream model/software and dataset terms also need checking before redistribution.

Both class lists cover eight of our thirteen target fruits: apple, banana, grape,
mango, orange, pomegranate, strawberry and tomato. They omit lemon, guava, jujube,
lulo and tamarillo. They predict identities and boxes, not freshness. Names are
model-card claims until a weight inspection/inference check confirms them.

## Bohni Tech Fruits and Vegi

https://universe.roboflow.com/bohni-tech/fruits-and-vegi/dataset/13

The project overview was accessible and lists 2,375 project images, 35 categories
and CC BY 4.0. The exact v13 page returned HTTP 403 to the command-line client and
could not be verified by the browsing tool. Do not call 2,375 the audited v13 export
size or claim it counts independent originals. Export size, split counts, class
balance, duplicates, augmentation lineage and per-class box counts remain to check.

The displayed class list matches the eight-target coverage above. Green_Orange is
another source category to inspect/remap deliberately; it is not a separate global
freshness label and is not a replacement for the lemon category.

## Recommendation

Prioritize auditing the Bohni export for mango/pomegranate and other sparse classes.
Use the medium pretrained detector as a reference candidate on a fixed independent
benchmark; its weight file is smaller than the large version and its card reports
similar metrics. Measure actual speed instead of inferring it from file size.
The LVIS-derived dataset can supplement the seven matching targets after cleanup.
All these sources combined still leave guava, jujube, lulo and tamarillo to source.
Keep new training samples out of the preserved first experiment's test set; detect
cross-source overlap before comparing a model already pretrained on related data.

For the small Android app, train/benchmark a compact detector using accepted new
data, or evaluate teacher-to-student distillation as a later experiment. Large and
medium checkpoints are not automatically the best latency choice for the phone.
