# Kotlin application code

MainActivity owns camera/result rendering and simple Fresh/Not fresh/Uncertain
wording. ScannerViewModel initializes both engines on one worker, decodes and
orients a chosen/captured image, detects regions, crops with6%padding, classifies,
and updates the screen. It deletes temporary captures and closes engines after
queued work. FruitDetector bridges native code; FreshnessClassifier loads a
hash-checked TFLite bundle and reuses buffers; ModelSpec enforces dimensions,
RGB normalization and exact labels. ScoreMath handles numerical conversions.

Low score or detector/classifier identity disagreement produces Uncertain.
thresholdValidated remains false: displayed estimates have not been calibrated
for real phone cameras. Application package is Custom; source namespace stays
com.fruitlens.app so JNI continues to resolve.
