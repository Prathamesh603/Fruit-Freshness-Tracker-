# Storage cleanup — 1 November 2026

The user requested a quick cleanup before supplying new fruit detection datasets. `cleanup.json` records exact removed targets, logical byte counts and measured free space. `before.json` and `after.json` verify the same **121 preserved model/APK/Classic-snapshot records** before and after cleanup.

Completed: **5,928,980,480 bytes recovered (5.93 GB)**; drive free space increased from 6,478,196,736 to 12,407,177,216 bytes (12.41 GB). All 17 existing targets were removed successfully. All 121 preservation checks passed before and after. The exact dated cleanup script is in `tools/`; do not rerun it as general maintenance after old data has been restored.

## Removed or scheduled for removal

| Data | Why it is redundant | Restore source / instructions |
|---|---|---|
| `ml/data/raw/mendeley_bdd69gyhv8/Original_Image.zip` | The useful original photos remain in `datasets/fruit13/raw/mendeley8/images/` | https://data.mendeley.com/datasets/bdd69gyhv8/1 ; exact public file URL and SHA256 below |
| `ml/data/raw/mendeley_bdd69gyhv8/originals/` | Older six-class experiment's duplicate extracted copy; its manifest, code, results and weights remain | Copy the original matching class/photos from the retained Mendeley collection, verify against `ml/data/manifests/splits_v1.csv`; or restore ZIP then use `ml/scripts/audit_data.py` |
| `datasets/fruit13/raw/fruit16k/publisher.zip` | Its extracted 16K source photos remain | https://data.mendeley.com/public-api/zip/6ps7gtp2wg/download/1 ; download/hash record remains in `datasets/fruit13/reports/downloads.json` |
| `datasets/zenodo-fruit9-v1/downloads/AgricultureDataset.zip` | The useful nine-fruit subset, original selected labels, quarantine and audit reports remain; no need for all 30 categories on disk | https://zenodo.org/records/18904638 ; downloader and checksum record remain in that dataset folder |
| YOLOv8n v1 `data/images/{train,val}` and `data/labels/{train,val}` | Reproducible old training copies; original source data, code and split manifest remain | `python detector_training/yolov8n-fruit9-v1/prepare_data.py` (regenerates reports too; protect recorded manifests before reproducing an old run) |
| Explicit Gradle/Kotlin/CMake/build caches | Generated files, not app sources or release APKs | Rebuild using each edition's documented toolchain. Installed SDK/NDK, environments, signing/user configuration and saved APKs remain. |

The **v1 test images and labels remain**, as do comparison photos, all trained checkpoints/exports, Classic/Classic Plus/Custom apps, their verification evidence, original Bohni ZIP, current freshness photos and selected YOLO26s starting weights. Old v2/Bohni prepared copies were already cleared on 31 October; their earlier restore notes remain.

## Exact archive restore records

Mendeley originals:

```text
URL: https://data.mendeley.com/public-files/datasets/bdd69gyhv8/files/de93ba06-6a58-45e3-913d-837b2ae52acb/file_downloaded
Destination: ml/data/raw/mendeley_bdd69gyhv8/Original_Image.zip
Bytes: 2794170228
SHA256: f89d67d4c4b24810bcd8406db877db6b1ac9834cea0b0cb56d65acebe59e66ff
```

FRUIT-16K archive:

```text
URL: https://data.mendeley.com/public-api/zip/6ps7gtp2wg/download/1
Destination: datasets/fruit13/raw/fruit16k/publisher.zip
Bytes: 99928214
SHA256: 9ff1c7efeb7c130f6d476adbee5c007a12df4d69960eb2356f0b34fedb75c59c
```

Zenodo archive:

```text
URL: https://zenodo.org/api/records/18904638/files/AgricultureDataset.zip/content
Destination: datasets/zenodo-fruit9-v1/downloads/AgricultureDataset.zip
Bytes: 1005423492
SHA256: 2108857c660e62253822b3e5de28405f0afa68ae3e9230984b97fe086cc2fafc
MD5: 394a7fb681366d22bd9e1c81d97884b3
Download only: python datasets/zenodo-fruit9-v1/prepare_dataset.py --download-only
```

Historic READMEs describe the state at their creation dates. This cleanup record and each affected folder's new `STORAGE_STATE.md` take precedence for current on-disk availability. Dataset reuse needs its original license and annotation checks; restoring bytes does not make a dataset balanced or independently validated.
