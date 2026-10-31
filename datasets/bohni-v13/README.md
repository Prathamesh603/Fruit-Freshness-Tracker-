# Bohni Fruits and Vegi, Roboflow version 13

This folder audits the ZIP supplied by the user and prepares a small detection
subset without extracting the entire archive. It does not retrain a model.

Original archive: `../Fruits and Vegi.v13-verison8.yolov8.zip`. It stays unchanged.
Read `reports/STATUS.md` for measured counts, annotation findings and readiness.

## Folder guide

- `metadata/`: unchanged publisher README files and original data.yaml.
- `prepared/images/`: compact RGB JPEG copies, long side at most 640 pixels, only
  for photos containing our matching fruit classes. Filenames are shortened hashes
  to work on Windows; reports/images.csv links them back to original ZIP members.
- `prepared/labels/`: matching YOLO fruit-identity labels, remapped to the previous
  experiment's nine-class order. Other produce classes are not included in this subset.
- `reports/`: per-image inventory, all 35 source class counts, duplicate and
  augmentation-family findings, checksums and readiness.
- `review/`: representative source photos with boxes for visual inspection.
- `audit_prepare.py`: reads the archive, validates every label/image and prepares
  the compact subset. Original image bytes remain available inside the ZIP.

## Class mapping

Use IDs 0 apple, 1 banana, 2 grape, 3 lemon, 4 mango, 5 orange, 6 pomegranate,
7 strawberry, 8 tomato, matching our completed YOLOv8n experiment. Source
Green_Orange and Orange both map to orange. Lemon is absent in this source.
No freshness-condition mapping is inferred from fruit identity or colour.

The source contains no guava, jujube, lulo or tamarillo classes. Together with our
older detection data it can improve nine-class coverage, not establish all 13.

## Preparation boundaries

Source files already include random crops and brightness augmentation. Several
files can come from one original photo. Original family names before `.rf.` are
recorded; all family members must stay in one project split. A cropped view may
contain different visible classes from another member of the same family.

`source_split` preserves the publisher train/valid/test folder. `project_split`
is deliberately unassigned. The supplied test split is too small and has no boxes
for our matching fruit classes, so it must not be used as a meaningful fruit test.
New group-aware splits and cross-source duplicate checks are needed before training.

The current source declares CC BY 4.0 and identifies version 13. Keep author,
source URL, license and changes with derived data. See metadata/README.dataset.txt.
The exporter README's headline count differs from the actual archive; use audited
counts instead of assuming all augmented files are independent original photos.

Images in this subset are classification-independent detection data. A box says
where a fruit is; it does not label its freshness, internal condition or safety.
