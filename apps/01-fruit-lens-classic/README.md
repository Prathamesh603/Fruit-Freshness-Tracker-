# Fruit Lens Classic — Pretrained CPU

This folder preserves the first working Android fruit-scanning app, before integrating our own fine-tuned model. Use this edition as the first demonstration and the baseline for later comparisons.

| Item | Value |
|---|---|
| Launcher name | **Fruit Lens Classic** |
| Android application ID | `com.fruitlens.classic` |
| Version | `1.0.0-classic` / version code 1 |
| Architecture | arm64-v8a phone APK; x86_64 builds available from source |
| Minimum Android | Android 8.0 / API 26 |
| Detector | NanoDet-Plus-m, 416 × 416, existing COCO-trained weights |
| Freshness classifier | Existing Smart Fruit MobileNetV2-based model, 96 × 96, nine classes |
| Inference execution | **CPU only** — NCNN + LiteRT/XNNPACK |
| GPU / NPU / cloud | Not used for model inference |
| Supported fruit types | Apple, banana, orange |

## Folder contents

* `apk/Fruit-Lens-Classic-CPU-v1-arm64.apk` — separately named, built Classic installer.
* `apk/original/FruitLens-Original-arm64.apk` — exact, unchanged APK originally tested on the phone; application ID `com.fruitlens.app`.
* `source/android/` — copied source, original model files, labels, manifests, license texts, build wrapper, tools, and tests. Build caches and SDKs are excluded from the portable snapshot.
* `TECHNICAL_DETAILS.md` — models, preprocessing, algorithms, CPU execution, optimizations, limitations, and latency interpretation.
* `DEMO_GUIDE.md` — presentation steps and a fair comparison procedure for future apps.
* `evidence/user-phone-screenshots/` — the six screenshots supplied by the user.
* `VERSION.json`, `SNAPSHOT_MANIFEST.json`, and `SHA256SUMS.txt` — identity, provenance, and integrity records, generated after building.

## Install and demonstrate

Open the Classic arm64 APK on the Android phone and allow installation from the transfer app if Android requests it. It appears as **Fruit Lens Classic**. It can coexist with the original **Fruit Lens** app and the future **Fruit Lens Custom** app; it does not upgrade or rename an already-installed original APK.

Allow camera access or choose a photo. Place one supported fruit in good lighting for the clearest demonstration. The app also processes up to five supported fruit regions per photo. All processing runs locally after installation.

The Classic model assets and inference pipeline are preserved. The separate app ID, launcher name, version string, and in-app brand identify this edition; they do not change the neural-network predictions.

## Preservation policy

Do not install new model exports into this folder. Future training exports belong in a separate Custom edition. Verify the archived files with `python verify_snapshot.py`; a checksum mismatch means a recorded baseline file changed. The portable ZIP, when generated, excludes SDKs, caches, intermediate builds, and machine-specific `local.properties`.

This is a local demonstration/test edition. Its freshness predictions are experimental, with no independently established accuracy or calibrated probability. Reference-classifier redistribution terms remain unresolved. Keep that provenance with the app; public release requires a cleared replacement or permission.

For a source rebuild, see `source/android/README.md` and the edition-specific instructions in `TECHNICAL_DETAILS.md`. Do not use the copied generic model-installation instructions to replace Classic's frozen model.
