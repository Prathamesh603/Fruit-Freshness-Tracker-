# Custom app updates

## 1.1.0-custom, version code 2

Install `apk/Fruit-Lens-Custom-v1.1-arm64.apk` to update Custom. The original v1 APK
and its Git tag remain available. Classic is unchanged.

Each displayed fruit region receives a photo-local ID, starting at #1. The same
ID appears on its bounding box and on its detail card. IDs are assigned after
tiny, unclassified regions have been filtered, so there are no gaps. They remain
fixed while viewing that result and restart on the next photo. These are not
persistent video-tracking IDs or reliable counts of individual fruit in bunches.

Boxes and card ID badges share one color rule: red through amber to green using
the classifier's relative Fresh / (Fresh + Not fresh) score for the predicted
fruit. Low-confidence, missing/invalid scores or detector/classifier identity
disagreement use neutral gray and Uncertain. Text labels remain visible so the
interface does not rely on red/green perception alone. The color is a model-score
visualization, not a measured percentage of remaining freshness or shelf life.

Model weights, input preprocessing, CPU backends and inference thresholds are
unchanged. Regression checks cover score normalization, color endpoints,
uncertain cases, actual two-fruit detection, matching overlay/card IDs, all 18
classifier outputs, camera capture, and the optimized internal build.
