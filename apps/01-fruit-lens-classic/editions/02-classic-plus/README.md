# Fruit Lens Classic Plus

The original pretrained Classic models with the numbered, colored result interface.
Install apk/Fruit-Lens-Classic-Plus-v1-arm64.apk on Android8+ arm64 phones.
Launcher: Fruit Lens Classic Plus. Package: com.fruitlens.classic.plus.
The original Classic and Custom apps can stay installed.

## What changed

Every displayed fruit has a photo-local ID starting at #1. The ID appears on its
bounding box and matching detail card. Cards show Detection score and Prediction
score; uncertain cards also show the classifier model pick. Green-to-red boxes
follow relative Fresh vs Not fresh scores; gray means low score or model identity
disagreement. The separate Unripe class is amber. These colors/scores do not measure
freshness percentage, shelf life or food safety. IDs restart on each photo and
are not video tracking IDs.

## What models run

The original Classic NanoDet-Plus-m416 NCNN detector and SmartFruit MobileNetV2
reference classifier are copied byte-for-byte. Detection supports apple, banana
and orange only. The classifier has nine outputs: each fruit times fresh, rotten
or unripe. Source labels and model manifests are unchanged. Visible UI maps rotten
to Not fresh and retains Unripe as its own condition.

Detector: original BGR means/std,416letterbox, score0.4/NMS0.5/max5regions.
Classifier:96RGB,pixel/255,nearest resize. Both engines use CPU2threads with no
GPU/NPU delegates. Models are initialized once and prewarmed on a background
worker. Colors and score formatting add no model inference.

Compared with the original Classic UI, Plus adds the Custom-style score0.8 display
abstention and identity disagreement handling. thresholdValidated remains false;
these are visual suggestions. Do not carry over the old phone latency or dataset
accuracy claims to this build without matched testing.

## Preservation and verification

Original source, weights, archived APKs, README files and71recorded baseline files
remain unchanged in the Classic folder. The new edition lives entirely under
editions/02-classic-plus/. MODEL_BUNDLE.json records exact deployed hashes.
VERSION.json and verification/ record actual build, model, camera, ID/score UI,
signature/alignment and coexistence checks. The old classifier's unresolved
redistribution terms remain recorded in its copied metadata; local demo only.

## Build and code

Use the existing project SDK/NCNN tools. From the project root:

```powershell
$env:GRADLE_USER_HOME = "$PWD/android/.tools/gradle-home"
android/.tools/gradle-8.13/bin/gradle.bat -p apps/01-fruit-lens-classic/editions/02-classic-plus/android --offline :app:testDebugUnitTest :app:assembleInternal
```

JDK17,SDK36,NDK28.2.13676358,CMake3.22.1,Gradle8.13; see the original Android
bootstrap instructions for another computer. Set android/local.properties to your
SDK path. Model artifacts require Git LFS on a fresh clone. Build uses local
development signing, not Play Store signing. Namespace/JNI remains com.fruitlens.app.

MainActivity renders boxes, IDs, cards and camera; ScannerViewModel manages one
worker and crops; FruitDetector and C++ use the original NanoDet decoder;
FreshnessClassifier preserves original tensors and computes display-only scores;
FreshnessDisplay centralizes condition names, color rules and score formatting.
Training code and trained Custom models are not used or modified here.
