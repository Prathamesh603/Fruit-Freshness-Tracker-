# Detection data search: Roboflow and other platforms

Checked 2026-10-30. This search found candidates for the four missing detection
classes and stronger mango/pomegranate coverage. Publisher-page numbers below
are not audited original-image counts. No candidate image archive was merged into
the existing collection. The local verified coverage remains nine detection fruits.

## Priority candidates

| Target | Dataset and link | Publisher-page coverage | Listed license | What remains to check |
|---|---|---|---|---|
| Guava | https://universe.roboflow.com/aidataset-4fd8s/guava-lj2yt-gv05c | 1,463 images; overripe/ripe/unripe guava; object detection; version 1 | CC BY 4.0 | Export, original vs augmented counts, box placement and image provenance. Merge identity labels to guava only after inspection; ripeness is not fresh/spoiled. |
| Guava alternative | https://universe.roboflow.com/fruit-yevjb/guava-lhmgg | 352 project images; GuavaFresh class | CC BY 4.0 | Multiple versions; choose and pin an export. Do not assume all source permissions or condition labels are independently validated. |
| Guava alternative | https://universe.roboflow.com/experiment-oxztk/guava-fruit-detection-o5uh3 | 262 project images; Guava-fruit-detection class | CC BY 4.0 | Check for overlap with other guava projects before adding counts. |
| Jujube | https://universe.roboflow.com/project01-610ou/all_jujube | 113 project images; Jujube class; version 1 | CC BY 4.0 | Export and inspect species/variety, orchard bias, boxes and source permissions. |
| Jujube alternative | https://universe.roboflow.com/fresh-jujube-fruit/fresh-jujube-fruit | 429 project images; object detection; page shows no dataset version | CC BY 4.0 | A project image count is not proof of an exportable dataset. Check whether the publisher has generated a downloadable version. |
| Lulo | https://universe.roboflow.com/frutas2/lulo | 712 project images; class named 1; two dataset versions; model page lists 1,781 version images | CC BY 4.0 | Resolve project/version/augmentation counts. Inspect that class 1 really encloses lulo fruit, then map to lulo. |
| Tamarillo | https://github.com/omercruz-sys/YOLO11---Solanum-betaceum-Cav- and https://doi.org/10.5281/zenodo.21167173 | Linked 2026 study describes 200 original smartphone orchard images with green/ripe fruit annotations | Dataset license not verified | Obtain the actual Zenodo image/annotation files and inspect license. GitHub's small ZIP is results/cache only, not the image dataset. |
| Mango | https://data.mendeley.com/datasets/gcgrjvwmm2/1 | 1,025 images; Single, Proximal, Far-Field folders, with images and labels | CC BY 4.0 | Metadata confirms an 865,808,607-byte RAR. Download and inspect actual labels/geometry before counting locally. |
| Pomegranate | https://universe.roboflow.com/vani-pyq5t/pomegranate-detection | 492 project images; pomegranate class | CC BY 4.0 | Pin an existing export version, inspect boxes, duplicates and provenance. Page model version and dataset version counts differ. |
| Pomegranate alternative | https://universe.roboflow.com/pomegranateproject2/promogranate | 617 project images; 1,665 model-version images; pomegranate class | CC BY 4.0 | Separate original images from augmentations and forks. |

## Tamarillo evidence and access findings

The paper is **Detection of Solanum betaceum Cav fruit maturity using YOLO11-based
deep learning**:
https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1880899/full

It describes 200 original images, manual annotations, and links the GitHub and
Zenodo repositories above. The inspected GitHub commit is
`2a9ca4d97bb830b53cd240fccbe5a7fcc9e6ca3d`. Its attachment has 53,676 bytes,
SHA-256 `4638c8e1f5fe61c5811d575607bad67fd3c77c3b5a8f47c60e84db2998e1c112`.
Archive contents are three .xlsx files and one .pkl file. No tamarillo image or box
dataset has been downloaded. Zenodo returned 403 to this client; access and dataset
license remain unverified, rather than guessed from the paper's publication license.

## Supplemental synthetic source

https://universe.roboflow.com/brad-dwyer/synthetic-fruit lists 6,000 synthetic
composite images across 63 classes, including guava, tamarillo, mango and pomegranate.
It lists CC BY 4.0. It does not list jujube or lulo in its displayed class taxonomy.
The images combine fruit cutouts with backgrounds; the page reports different
model-version counts from the base project count.

This is supplemental training material only, not real-phone evaluation or a
replacement for missing natural-scene data. Inspect the cutout/background provenance
and license obligations before use. Do not add copies/forks of this same dataset
as independent sources. It has not been downloaded here.

## Sources that are not replacements for fruit detection boxes

- JujubeImageBD: https://data.mendeley.com/datasets/dydrxfpxm7 — useful classification
  originals, but the page does not establish fruit bounding-box annotations.
- eNFCG-Frutas: https://zenodo.org/records/21839052 — tamarillo classification
  coverage, but restricted files and no established detection boxes.
- Pomegranate growth-stage datasets can include buds and flowers. Do not remap
  those boxes into pomegranate fruit. A class name containing a fruit word is not
  enough to establish a valid fruit-body detector label.
- MangoFruitBD: https://data.mendeley.com/datasets/bhrz29mkmr/2 — documented YOLO
  fruit boxes, but CC BY-NC 3.0. Keep separate from the preferred CC BY collection.

## How to obtain and merge these datasets

1. On each Roboflow project, open Dataset, select an existing version, and use the
   normal download/export flow. Prefer YOLO or COCO with images AND annotations.
   SDK downloads use an authorized Roboflow API key configured locally. Do not put
   keys into this report, commit history, or screenshots. No credentials have been
   requested, discovered, or used in this search.
2. For Mendeley mango, use the public download URL stored in mango_mendeley.json.
   Verify the publisher SHA-256:
   `763f074b503846623abbf00b99ef4947aff9ef80a210b5b0e685036417f7d8ff`.
3. For tamarillo, use the linked Zenodo record through its normal public access
   flow. Confirm it contains both original images and annotations, with a dataset
   license. The GitHub attachment is insufficient.
4. Save each archive unchanged in its own raw source folder, with author, URL,
   pinned version, license, byte size and SHA-256. Add a folder README.
5. Audit image decoding, actual counts, labels, bounding boxes and transformations.
   Map fruit identities to labels.json. Preserve original condition labels separately.
6. Deduplicate exact and near copies across ALL sources before making splits.
   Roboflow project counts, version counts and augmented-image counts must not be
   added together as independent originals.
7. Include spoiled and damaged appearances in detection examples, not just ripe
   healthy orchard fruit. Inspect species consistency, especially jujube varieties.
8. Update coverage.csv and STATUS.md only after local images and boxes pass checks.
   Public listings alone do not change our verified nine-fruit coverage.

The search substantially reduces the amount of annotation we may need to do from
scratch. It does not establish 13-fruit robustness or create a trained detector.
