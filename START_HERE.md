# Fruit Lens — start here

## Current work

**Next model: YOLO26s, nine-fruit detector v3. Training has not started.**

1. Add each new dataset link to [datasets/fruit9-v3/links.csv](datasets/fruit9-v3/links.csv).
2. Save new source downloads under `datasets/fruit9-v3/incoming/`; audit before merging.
3. Prepare the reviewed dataset under `datasets/fruit9-v3/prepared/`.
4. Keep all new detector code, configuration, logs, results and checkpoints in [detector_training/yolo26s-fruit9-v3/](detector_training/yolo26s-fruit9-v3/WORKFLOW.md).
5. Compare the new model with the preserved apps before deploying it.

## Find existing work

| What you need | Open this |
|---|---|
| Classic app, pretrained models and original APKs | [apps/01-fruit-lens-classic/README.md](apps/01-fruit-lens-classic/README.md) |
| Classic Plus, IDs/colors/scores with Classic models | [Classic Plus](apps/01-fruit-lens-classic/editions/02-classic-plus/README.md) |
| Current Custom app and APK | [apps/02-fruit-lens-custom/README.md](apps/02-fruit-lens-custom/README.md) |
| All preserved model versions and hashes | [models/README.md](models/README.md) |
| First nine-fruit detector | [YOLOv8n v1](detector_training/yolov8n-fruit9-v1/README.md) |
| Current nine-fruit detector, code/losses/results | [YOLOv8n v2](detector_training/yolov8n-fruit9-v2/MODEL_CARD.md) |
| Current nine-fruit freshness classifier | [MobileNetV3-Small](classifier_training/mobilenetv3small-fruit9-v1/MODEL_CARD.md) |
| Earlier three-fruit/six-class classifier | [ml/README.md](ml/README.md) |
| Real comparison photos and predicted boxes | [Comparison report](detector_training/web-image-comparison-v1/report.html) |
| New dataset links, downloads and audits | [datasets/fruit9-v3/README.md](datasets/fruit9-v3/README.md) |
| Old data restore instructions and cleanup record | [Latest cleanup](datasets/storage-cleanup-2026-11-01/README.md) |
| Complete historical project guide | [PROJECT_GUIDE.md](PROJECT_GUIDE.md) |

## Folder rules

`apps/` contains runnable app editions. `models/` is the stable model catalog. `detector_training/` and `classifier_training/` contain versioned experiments. `datasets/` contains source data, links and audits. `tools/` contains shared helpers.

Old code paths remain in place because recorded manifests and scripts refer to them. Existing READMEs, logs, results and model files are preserved. A new experiment gets a new folder, never a replacement checkpoint in an old folder. `android/` is the original prototype; `ml/`, `model_exports/`, and `Training/` are legacy/user work and are not the new detector workspace.
