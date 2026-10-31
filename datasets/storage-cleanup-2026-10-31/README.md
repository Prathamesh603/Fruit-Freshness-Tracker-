# Disk cleanup requested on 31 October 2026

The user authorized removal of unnecessary previous training data and Android/Gradle caches so that the new nine-fruit detection dataset would fit.

`cleanup.json` records each exact absolute path, its logical file sizes, whether removal succeeded, and the actual drive free space before and after. Free space increased from 2,037,760,000 to 8,107,106,304 bytes: approximately **6.07 GB recovered**. This records measured free space, not an estimate from sparse emulator images.

## Removed

- Generated Android build output and local Gradle project caches for the prototype, Custom, and Classic Plus editions.
- The project-local Gradle dependency cache. Gradle will download dependencies again at the next build.
- Downloaded Android/Gradle/NCNN installer ZIPs whose installed tools remain available.
- The v2 detector's generated `data/images` and `data/labels` copies. Its reports, split manifests, training/evaluation programs, logs, checkpoints, and model exports remain.
- The reproducible Bohni `prepared` subset. The user's original Bohni ZIP and the preparation code/reports remain.
- The unused Mendeley `Augmented_Image.zip`. This project uses originals for classifier training; the original archive and extracted freshness photos remain.

## Preserved

All saved model weights and APKs, including all previous versions; source code; training and evaluation reports; Classic's preserved baseline; signing/user configuration; Android SDK/NDK/CMake; installed Gradle; emulator user data and snapshots; freshness classifier photos; source archives needed to rebuild the useful data; the v1 detector data used for frozen comparisons; comparison photos and predicted boxes.

`preserved-artifacts.json` records SHA256 hashes for 69 candidate binary/model/APK files immediately before cleanup. Fifteen were Gradle cache `.bin` files; ten of those were intentionally removed and five were in retained caches. `preservation-verification.json` separates those cache records and confirms that all **54 actual saved model/APK files remain byte-identical**.

The caches outside this project, such as the shared `D:\.gradle` home, were not deleted. Most recoverable project cache data was in `android/.tools/gradle-home`.

## Restore generated files when needed

Run these commands from the project root only when the old data is actually needed:

```powershell
# Uses the original Bohni ZIP that remains on disk.
python datasets/bohni-v13/audit_prepare.py

# Requires the restored Bohni prepared subset and the retained v1 data.
python detector_training/yolov8n-fruit9-v2/prepare_data.py
```

Consult the existing Bohni and detector reports before overwriting generated inventories. To build an app again, use its existing build instructions; Gradle caches and app build outputs regenerate. The prototype's bootstrap utility is `android/tools/bootstrap_android.py`; installed toolchains were retained, so reinstalling all Android tools is unnecessary.

The removed unused augmentation archive can be retrieved from the Mendeley source documented in `ml/data/DATA_SOURCES.md` and `datasets/fruit13/SOURCES.md` if ever required. It is not needed for reproducing the current classifier.

The exact cleanup program is `tools/cleanup_training_space_20261031.ps1`. It checks every resolved target is inside this project before recursively deleting it. Do not rerun it as routine maintenance: it would remove regenerated copies again.
