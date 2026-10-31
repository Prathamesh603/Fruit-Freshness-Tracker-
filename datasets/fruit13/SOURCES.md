# Where the data came from

## 1. Mendeley eight-fruit originals

- Title: Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality.
- Authors: Nusrat Sultana, Musfika Jahan, Mohammad Shorif Uddin (2022).
- Version: 1. DOI: https://doi.org/10.17632/bdd69gyhv8.1
- Publisher: https://data.mendeley.com/datasets/bdd69gyhv8/1
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Original archive reused from `../../ml/data/raw/mendeley_bdd69gyhv8/Original_Image.zip`, relative to this folder.
- Expected SHA-256: `f89d67d4c4b24810bcd8406db877db6b1ac9834cea0b0cb56d65acebe59e66ff`.
- The preparation script verifies that hash before extracting an independent copy.
- Photos are in `raw/mendeley8/images/`: 3,200 originals, 200 per source class.
- Fruits: apple, banana, grape, guava, jujube, orange, pomegranate, strawberry.
- Publisher labels: Fresh and Rotten; our normalized condition names: fresh and spoiled.
- No publisher bounding boxes. No physical-fruit or capture-session identifiers.
- The publisher's augmented archive is not used. The earlier experiment found inconsistent augmented filenames and no reliable original-to-augmentation mapping.
- Some originals were used in the completed three-fruit experiment. `baseline_overlap` marks matching hashes; these are not new external evaluation evidence.

## 2. FRUIT-16K

- Title: Spoiled and fresh fruit inspection dataset.
- Authors: Cesar Giovany Pachon Suescun, Javier Orlando Pinzon Arenas, Robinson Jimenez-Moreno (2020).
- Version: 1. DOI: https://doi.org/10.17632/6ps7gtp2wg.1
- Publisher: https://data.mendeley.com/datasets/6ps7gtp2wg/1
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Public download: https://data.mendeley.com/public-api/zip/6ps7gtp2wg/download/1
- Original downloaded archive: `raw/fruit16k/publisher.zip`.
- Local SHA-256 and byte size are recorded in `reports/downloads.json`.
- Archive contains exactly 16,000 JPEGs in 16 folders: 1,000 per folder.
- Fruits: banana, lemon, lulo, mango, orange, strawberry, tamarillo, tomato.
- `F_` means publisher fresh; `S_` means publisher spoiled. Original names are retained.
- No object bounding boxes or physical-fruit/session IDs are supplied in the archive.
- The publisher describes changing background, angle, distance, and lighting. That does not prove that all photos show independently purchased fruits.
- A sample sheet shows repeated-looking fruit and background setups. Some fresh/spoiled distinctions, especially bananas, need manual review. Green citrus is not automatically a bad label; varieties differ.

## 3. Boxed freshness repository

- Repository: https://github.com/zijianchen98/Fruit-freshness-detection-dataset
- Pinned commit: `0cdcf30cb86a6a051b666d9138b5436264972040`.
- Download: https://codeload.github.com/zijianchen98/Fruit-freshness-detection-dataset/zip/0cdcf30cb86a6a051b666d9138b5436264972040
- Archive contains 3,202 JPEG images and 3,202 Pascal VOC XML annotation files.
- Fruit types: apple, banana, orange. Source conditions: fresh, normal, rotten.
- Repository contains an Apache-2.0 LICENSE; it is preserved under the extracted folder.
- Image acquisition/ownership is not fully explained by the short README. Keep this source separate and review image provenance before including it in a redistributable training release.
- `normal` has no agreed mapping to our binary freshness labels. These images are currently detection candidates only, not additional binary classifier labels.
- Source coordinate endpoints and box quality need review before YOLO/COCO conversion.

## 4. Open Images detection subset

- Official documentation: https://storage.googleapis.com/openimages/web/download_v7.html
- We downloaded the older V5 validation bounding-box table exposed by the official storage service, not the full V7 training dataset.
- Source URLs and file hashes: `reports/downloads.json`.
- Box class names: `sources/openimages/class-descriptions-boxable.csv`.
- Selected photo attribution: `sources/openimages/selected_image_attribution.csv`.
- Individual downloaded photos: `raw/openimages/images/`.
- Selected images are required by the collector to have a source CC BY 2.0 image license. Authors and original landing URLs are retained per image. See https://creativecommons.org/licenses/by/2.0/ .
- Open Images annotations are published under CC BY 4.0; see https://storage.googleapis.com/openimages/web/index.html .
- This subset provides boxes for apple, banana, grape, lemon, mango, orange, pomegranate, strawberry, and tomato.
- Guava, jujube, lulo, and tamarillo have no specific class in this boxable label list.
- The source split is called validation. In this collection it is staged raw data with `split=unassigned`; it is not silently relabelled as an independent project test set.
- These photos have no freshness labels. Never assume they are fresh because they came from Open Images.
- Group boxes, depictions, occlusion and truncation flags remain available. Group boxes must not be treated as individual fruit counts.

## Sources investigated but not added as image training data

AgriFreshNET (https://data.mendeley.com/datasets/42m5tb7yv9/1) was investigated.
Only its public file listing was saved. Its image ZIP was not downloaded, its
reported class taxonomy was not audited, and its images are not counted here.

Fruits-360 is not part of this 13-fruit release. It has broad identity classes but
does not solve our missing binary condition labels or four missing detection classes.
Wild-FRUVEG67 is not included: dataset permissions and annotation quality need
separate review before adoption.

## Attribution when sharing

Retain this file, source licenses, author records, dataset DOIs, original class names,
and the changes made during preparation. Raw photos retain their source licenses;
they do not become original Fruit Lens photography. This project has not contacted
or obtained endorsement from any dataset author. A local checksum proves which
bytes were collected; it does not certify the correctness of the labels.
