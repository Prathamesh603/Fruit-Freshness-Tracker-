# Verification record — 30 October 2026

The app builds successfully for arm64-v8a and x86_64. The optimized phone installer is `../dist/FruitLens-Android-arm64.apk` (16.95 MiB), signed with the development key. Artifact metadata and hash are in `summary.json`.

Passed: six Python bundle checks, five JVM numerical checks, and four Android 11 x86_64 emulator integration tests. The Android tests execute the real detector and freshness model, reject an empty scene, repeat inference, decode a real CameraX capture, and verify a photo/crop/result flow with a rendered Compose result. `instrumentation.txt` and `inference-logcat.txt` contain the final run. The banana fixture is illustrative, not a representative accuracy evaluation.

The optimized APK itself was installed and exercised through startup, model warmup, camera binding/capture, native inference and visible result rendering. APK signature and 16-KiB ZIP/native ELF alignment checks passed. The APK requests camera permission and no Internet permission.

`result-preview.png` is the rendered banana result from the test. `optimized-camera-result.png` records the optimized build declining to classify an unsupported emulator camera scene. The separate manual photo-picker selection experiment was inconclusive: its chooser displayed the fixture, but injected selection did not complete; it is not counted as a passing check.

Unmeasured: actual Nord 2T/other-phone speed, energy, sustained scanning, representative freshness accuracy and calibrated confidence. Reference freshness weight redistribution rights remain unresolved. This is a local-test app with experimental freshness suggestions, not a validated public release.
