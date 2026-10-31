# Fruit Lens — Android

**Preserved Classic edition:** this copied source builds **Fruit Lens Classic**, ID `com.fruitlens.classic`. Its model assets are frozen. The generic model-installation commands below describe the original development workflow; do not use them to install custom weights into Classic. Follow the edition-level `../../TECHNICAL_DETAILS.md` and create a separate Custom edition for new models.

An offline Android app with CameraX preview, photo capture/image selection, native fruit detection, high-resolution fruit crops, and per-fruit visible-condition classification. Supports apples, bananas, and oranges, up to five regions per photo. No Internet permission, account, or image upload. Temporary captures are deleted after processing; selected photos are not retained.

## Models and runtime

* **Detector:** official NanoDet-Plus-m 416 checkpoint, C++/NCNN 20261020, CPU with two threads, BGR normalization and letterbox resize. Download provenance/hashes are saved beside the assets.
* **Classifier:** existing Smart Fruit MobileNetV2-based 96x96 reference, nine fruit/condition outputs, RGB `[0,1]`, nearest resize matching its notebook. LiteRT 1.4.2 CPU Interpreter establishes the initial execution baseline.
* Both engines stay initialized on one background worker, prewarm before scanning, and close after pending work finishes. Classifier tensor buffers are reused. No continuous inference backlog or OpenCV dependency.
* GPU/NPU, NCNN classifier conversion, and alternative-model speed comparisons have not been implemented or claimed faster.

The freshness reference is **experimental and not independently evaluated**. Scores are not calibrated probabilities, freshness percentages, or evidence of food safety. The app marks suggestions accordingly and abstains on fruit-type disagreement. A validated training export replaces the reference without changing the UI.

**Reference-weight redistribution terms are unresolved.** The source repository has no visible explicit license. Existing-weight builds are for local evaluation; replace the model with a cleared export or establish permission before public distribution. Detector/runtime license texts are included in assets.

## Build and install

JDK 17, Gradle 8.13, SDK 36, build-tools 36.0.0, NDK 28.2.13676358, CMake 3.22.1. Minimum Android API 26. APKs are split by architecture; Nord 2T and normal modern phones use **arm64-v8a**. x86_64 is for emulator testing.

The project-local bootstrap downloads official tools and checks their published checksums:

```powershell
python android/tools/bootstrap_android.py --native
python android/tools/prepare_detector.py
```

Install a training bundle as below. For the local-only reference, inspect and run `prepare_reference_classifier.py`; it requires `tflite==2.18.0` under `android/.tools/python`, plus NumPy/FlatBuffers. It verifies actual IO shapes and preserves an existing app model.

From `android/`, after bootstrap writes ignored `local.properties`:

```powershell
$env:GRADLE_USER_HOME = Join-Path (Get-Location) '.tools/gradle-home'
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug :app:assembleInternal
```

The wrapper pins Gradle's SHA-256. If its download is unavailable but bootstrap downloaded Gradle, substitute `.\.tools\gradle-8.13\bin\gradle.bat` for `.\gradlew.bat`.

`internal` uses R8/resource shrinking and the local development signing key for installation. It is not production signing. The ordinary `release` variant needs release signing before distribution.

Output: `app/build/outputs/apk/internal/app-arm64-v8a-internal.apk`. Debug output: `app/build/outputs/apk/debug/app-arm64-v8a-debug.apk`.

Transfer the arm64 APK to the phone and open it, allowing installation for that transfer app if Android requests it. Allow camera access or choose an image. Keep a supported fruit close to the camera in even lighting. Multiple-fruit photos are supported, but classifier quality remains experimental.

## Training-agent integration

See [the detailed prompt](../TRAINING_AGENT_PROMPT.md). The external agent owns `ml/` and `model_exports/`; app work stays here. Existing user `Training/` work is not modified.

From the project root:

```powershell
python android/tools/install_model.py 'model_exports/<candidate>'
# Preserve the installed model as a backup before replacing:
python android/tools/install_model.py 'model_exports/<candidate>' --replace
```

Rebuild/reinstall afterward. The installer checks required files, exact label order, model hash, normalization, dimensions, threshold metadata, and TFLite identifier. Runtime checks executable tensor shapes/dtypes. A header check is not graph execution or evidence of accuracy.

Contract: RGB NHWC `[1,height,width,3]`, stretch resize, declared nearest/bilinear interpolation (default bilinear), raw-channel `(pixel - mean) / std`; output `[1,labelCount]` declared logits/probabilities. FLOAT32/INT8/UINT8 IO are supported using actual tensor quantization parameters. Unvalidated thresholds never yield accepted assessments. The classifier fruit must agree with the detector fruit.

## Verification

Emulator logs/screenshots live in `verification/`. These are execution smoke tests, not an accuracy benchmark or phone-performance study.

```powershell
python -m unittest discover -s android/tools/tests -v
```

JVM checks cover normalization, quantization, stable softmax, and invalid outputs. Python checks cover bundle integrity/order, path constraints, normalization, and preserving installed work.

Project-local emulator reproduction on Windows:

```powershell
python android/tools/bootstrap_android.py --emulator
python android/tools/prepare_emulator.py
python android/tools/start_test_emulator.py
# Wait for boot, then build the test packages:
cd android
.\gradlew.bat :app:assembleDebug :app:assembleDebugAndroidTest
cd ..
python android/tools/run_smoke_tests.py
```

The isolated AVD uses ADB port 5038 and emulator port 5556. Shut down only this test emulator afterward. Tests cover a real banana fixture, empty scene, repeated inference, photo/crop/result processing, and camera capture/decoding. Fixture attribution is included in test assets, not the application APK.

Before a validated release: representative held-out freshness evaluation, fruit/lighting/occlusion edge cases, real Nord 2T camera/latency/energy checks, wider Android lifecycle/accessibility checks, and weight redistribution clearance. Sustained FPS and phone latency are unmeasured. UI processing time excludes capture and photo decoding; emulator timing is not phone performance.
