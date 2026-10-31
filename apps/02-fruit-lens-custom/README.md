# Fruit Lens Custom

Separate Android edition using our nine-fruit trained models. Classic remains unchanged.

## Use it

Install `apk/Fruit-Lens-Custom-v1-arm64.apk` on an Android8+ arm64 phone. Open **Fruit
Lens Custom**, allow the camera or choose a photo, then tap **Check this fruit**.
Use one clear fruit in even light. Scan again to check another side. No account,
Internet permission, model download or image upload is required.

Results say **Fresh**, **Not fresh**, or **Uncertain**. The classifier's original
fresh/spoiled label IDs remain unchanged internally. The UI does not call a model
score a freshness percentage. Identity disagreement or a score below0.8 gives
Uncertain. This0.8 setting is an engineering heuristic, not phone-calibrated.

Supported: apple, banana, grape, lemon, mango, orange, pomegranate, strawberry,
tomato. Other fruit types are unsupported and can still be misidentified.

## Models and speed

| Role | Model | Runtime |
|---|---|---|
| Detection | YOLOv8n nine-fruit v2,416RGB | NCNN20261020 C++ CPU,2threads, FP16stored weights, no Vulkan/FP16arithmetic |
| Classification | MobileNetV3Small nine-fruit v1,160RGB,18classes | LiteRT1.4.2 CPU Interpreter,2threads,XNNPACK, FP16weights/float32IO |

Models stay initialized on one background worker. Startup prewarms both engines;
classifier input/output buffers are reused. Detection uses the full photo and
classification uses padded fruit crops, up to5regions. The UI remains responsive
while processing. There is no continuous video inference queue.

The detector's combined dataset mAP50 is79.01%; original frozen-test mAP50 is52.36%,
below v1's58.68%. Lemon is weak. Classifier dataset accuracy is99.65% on2,004images.
These separate dataset metrics are not complete-app accuracy. Same-source photos,
unknown capture sessions and crop differences limit generalization. Android
emulator verification is recorded in `verification/`; real-phone speed and
accuracy must be measured separately.

Exact deployed hashes: `MODEL_BUNDLE.json`. Training/model details: project
`models/custom-detectors/yolov8n-fruit9-v2/` and
`models/custom-classifiers/mobilenetv3small-fruit9-v1/`.

## Build

From the project root, reuse the existing local tools:

```powershell
$env:GRADLE_USER_HOME = "$PWD/android/.tools/gradle-home"
cd apps/02-fruit-lens-custom/android
../../../android/.tools/gradle-8.13/bin/gradle.bat --offline :app:testDebugUnitTest :app:assembleInternal
```

On another machine, first follow the original bootstrap instructions in
`android/README.md`, install Git LFS and pull the weights. Set this edition's ignored
`android/local.properties` to your Android SDK. JDK17,SDK36,NDK28.2.13676358,
CMake3.22.1,Gradle8.13 are required. The native build references the shared project
NCNN installation, not a duplicate SDK. Packaging uses the local development key;
this is an installable demonstration build, not Play Store signing.

Identity: launcher **Fruit Lens Custom**, package `com.fruitlens.custom`, version
`1.0.0-custom`. Source namespace/JNI deliberately stays `com.fruitlens.app`.
Changing the application ID does not require renaming JNI entry points.

## Code map

- `android/app/src/main/java/com/fruitlens/app/MainActivity.kt`: camera/result UI.
- `ScannerViewModel.kt`: background processing, orientation, crops and lifecycle.
- `FreshnessClassifier.kt`/`ModelSpec.kt`: checked model contract, normalization,
  reused LiteRT tensors, score processing and model hash validation.
- `FruitDetector.kt`/`android/app/src/main/cpp/fruit_detector.cpp`: JNI and YOLO
  decoding, RGB letterbox114, external NMS and normalized result bounds.
- `android/app/src/main/assets/`: deployed models only; no training data.
- `verification/`: actual build/test evidence and screenshots.

Licenses and source provenance remain recorded beside the model catalog and
dataset inventories. Ultralytics software/model license applies to the detector;
do not assume that training weights yourself removes upstream terms.
