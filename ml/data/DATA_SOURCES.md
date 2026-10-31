# Data sources and provenance

This file records the dataset used to train `mobilenetv3small-sixclass-v1`, the license position, and the integrity checks performed before splitting.

## Selected source

| Field | Value |
|---|---|
| Name | Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality |
| Version | 1 |
| DOI | `10.17632/bdd69gyhv8.1` |
| Landing page | https://data.mendeley.com/datasets/bdd69gyhv8/1 |
| Publisher | Mendeley Data |
| Contributors | Nusrat Sultana, Musfika Jahan, Mohammad Shorif Uddin |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| License text | https://creativecommons.org/licenses/by/4.0/ |
| Downloaded | 2026-10-29 |

The dataset landing page describes 3,200 original images and 12,335 augmented images across 16 classes. This project uses only six original-image classes.

### Archives

The publisher filenames contain spaces. The local copies use underscores.

| Publisher filename | Local file | Bytes | SHA-256 |
|---|---|---:|---|
| `Original Image.zip` | `Original_Image.zip` | 2,794,170,228 | `f89d67d4c4b24810bcd8406db877db6b1ac9834cea0b0cb56d65acebe59e66ff` |
| `Augmented Image.zip` | `Augmented_Image.zip` | 803,272,180 | `5c748b5904932cd8cf631a0b776253d6217d577067d9477c267014d03ac16e10` |

Local archive directory:

```text
ml/data/raw/mendeley_bdd69gyhv8/
```

No credentials or account access were used for the download. The direct file links below were read from the public Mendeley file metadata endpoint `https://data.mendeley.com/api/datasets/bdd69gyhv8/files` on 2026-10-30. Re-check the landing page if a link changes.

```powershell
New-Item -ItemType Directory -Force ml\data\raw\mendeley_bdd69gyhv8 | Out-Null

Invoke-WebRequest `
  -Uri 'https://data.mendeley.com/public-files/datasets/bdd69gyhv8/files/de93ba06-6a58-45e3-913d-837b2ae52acb/file_downloaded' `
  -OutFile 'ml\data\raw\mendeley_bdd69gyhv8\Original_Image.zip'

Invoke-WebRequest `
  -Uri 'https://data.mendeley.com/public-files/datasets/bdd69gyhv8/files/ccd1f142-03b2-473a-8c78-78920e63b8bd/file_downloaded' `
  -OutFile 'ml\data\raw\mendeley_bdd69gyhv8\Augmented_Image.zip'
```

Raw data is excluded from version control. Verify a download before using it:

```powershell
Get-FileHash -Algorithm SHA256 ml\data\raw\mendeley_bdd69gyhv8\Original_Image.zip
Get-FileHash -Algorithm SHA256 ml\data\raw\mendeley_bdd69gyhv8\Augmented_Image.zip
```

## Classes used

Only the original images were extracted. The source class names map to canonical labels as follows:

| Source directory | Canonical label | Fruit | Condition |
|---|---|---|---|
| `FreshApple` | `fresh_apple` | apple | fresh |
| `FreshBanana` | `fresh_banana` | banana | fresh |
| `FreshOrange` | `fresh_orange` | orange | fresh |
| `RottenApple` | `rotten_apple` | apple | rotten |
| `RottenBanana` | `rotten_banana` | banana | rotten |
| `RottenOrange` | `rotten_orange` | orange | rotten |

Extracted originals:

```text
ml/data/raw/mendeley_bdd69gyhv8/originals/
```

Exactly 200 original images were available for each of the six classes: 1,200 images total. The classes are balanced at 200 images each before audit exclusions.

The publisher supplies folder-level classification labels only. No bounding boxes, segmentation masks, or cropped-object annotations are provided. Images are full frames; background and crop composition vary by source photo. The publisher states that images were collected from fruit shops and real fields, but does not provide per-image camera, lighting, background, or capture-session metadata.

## Manual sample check

The sample grid at `ml/reports/sample_grid.png` was inspected. The six selected folders visually correspond to their stated fruit and condition labels, subject to the one cross-label conflict described below. The source contains no official train/validation/test split, so the split in this project is a group-aware generated split.

## Why the publisher's augmented images were not used

The augmented archive contains 12,335 images, but many filenames do not match their containing class. Examples include `Augmented Image/FreshApple/FreshOrange (1).jpg` and similar cross-fruit names. The archive does not provide a reliable mapping from an augmented file to its original source image.

Training on files that are visually close to validation or test originals would create leakage. Reconstructing the mapping would require a separate manual or perceptual-match audit. That work was outside this iteration, so the augmented archive was retained only as a downloaded source and was not used for training, validation, test, or threshold selection.

## Audit results

The audit is reproducible with `ml/scripts/audit_data.py`; machine-readable evidence is in `ml/data/manifests/data_audit.json`.

- Images decoded successfully: 1,200 of 1,200.
- Size range: width 469 to 4,160 pixels; height 446 to 4,160 pixels.
- Exact duplicate groups: 7, all containing two `fresh_orange` files and no cross-label copies.
- Perceptual grouping: 64-bit pHash, Hamming distance threshold 6, yielding 1,167 groups including the cross-label conflict.
- After excluding the conflict group: 1,166 groups are available for splitting.
- Groups with one image after exclusion: 1,138.
- Largest perceptual group after exclusion: 4 images.
- Cross-label conflicts: 1 group.
- Corrupt or unreadable images found: 0.

### Excluded conflict group

Group `g00463` contains:

```text
ml/data/raw/mendeley_bdd69gyhv8/originals/FreshOrange/FreshOrange (166).jpg
ml/data/raw/mendeley_bdd69gyhv8/originals/FreshOrange/FreshOrange (61).jpg
ml/data/raw/mendeley_bdd69gyhv8/originals/RottenOrange/RottenOrange (108).jpg
```

The same perceptually matched group is labelled both fresh and rotten. All three images were excluded because the correct condition cannot be established from the source metadata. The exclusion is recorded in `ml/data/manifests/excluded_conflict_groups.json`.

Remaining images: 1,200 - 3 = 1,197.

## Split provenance

Splits were created by `ml/scripts/prepare_splits.py` after grouping perceptual near-duplicates. The group, not the individual file, is the split unit.

| Split | Images | Per class |
|---|---:|---|
| Train | 837 | 138-140 |
| Validation | 180 | 30 each |
| Test | 180 | 30 each |
| Total | 1,197 | |

Perceptual groups: 817 train, 176 validation, 173 test.

Integrity checks passed:

- No group ID occurs in more than one split.
- No exact SHA-256 duplicate occurs in more than one split.
- Every canonical label is present in every split.
- The split manifest hash is stable.

Files:

- `ml/data/manifests/splits_v1.csv`
- `ml/data/manifests/split_summary_v1.json`

Split manifest SHA-256:

```text
47cfa8f903e04a59b9a1a059ff4d5f2267bd2010104e7f115a2a48121e8e1cd7
```

Audit SHA-256:

```text
ee4d6e50c921b388a3f23aa2a79b341af3d59fa8c4ef5f887b026038a40f4656
```

## Residual data risk

The source does not provide physical-fruit IDs, capture-session IDs, or acquisition metadata. Perceptual hashing reduces obvious train/test duplication, but visually similar photos of the same fruit taken at different moments may remain in different splits. Test metrics therefore describe this dataset's split, not performance on independent farms, kitchens, or phone cameras.

## Datasets considered but not used

### Kaggle six-class fruit dataset

| Field | Value |
|---|---|
| URL | https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification |
| Access | Kaggle credentials were not available and no authenticated download was performed |
| License | Listed as unknown; no redistribution permission established |
| Local files | None |
| Use | Not used for training, validation, test, or threshold selection |

This dataset was not a fallback for the final model because its license is unresolved, it requires account/credential access, and the selected Mendeley source has explicit CC BY 4.0 terms and sufficient class coverage.

The similarly named Zenodo record `https://zenodo.org/records/4788775` was not treated as equivalent and was not used. Its provenance, version, files, and license would need a separate audit before use.

## No phone evaluation data

No user phone photos were available. Real-device camera performance, blur, background clutter, multiple fruit, and unsupported-object behavior remain unverified.

A reserved collection area and labeling procedure are documented in `ml/data/phone_eval/README.md`. Phone data must remain separate from training and model/threshold selection, and must be audited for overlap with the training manifest before being called external evidence.

## Attribution and redistribution

CC BY 4.0 permits use and redistribution with attribution. This project:

- does not redistribute the raw dataset;
- includes six CC BY 4.0 test images under `model_exports/mobilenetv3small-sixclass-v1/parity/inputs/` solely as parity fixtures;
- supplies attribution in the bundle's `THIRD_PARTY_NOTICES.md` and model card.

No endorsement by the dataset authors or Mendeley is implied.

## Citation

Sultana, N., Jahan, M., & Uddin, M. S. (2022). *Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/bdd69gyhv8.1
