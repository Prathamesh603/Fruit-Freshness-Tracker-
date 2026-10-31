# Fruit freshness detection

Android-first offline fruit scanner.

The existing pretrained-model version is preserved separately as [Fruit Lens Classic](apps/01-fruit-lens-classic/README.md). Its source, original model weights, APKs, execution details, and demonstration notes stay together. Future custom-model work uses a different folder, name, and application ID; see [app editions](apps/README.md).

* [Android app and build instructions](android/README.md)
* [Detailed external training-agent prompt](TRAINING_AGENT_PROMPT.md)
* [Expanded 13-fruit dataset collection and folder guide](datasets/fruit13/README.md)
* [Actual dataset counts, review findings, and missing annotations](datasets/fruit13/reports/STATUS.md)
* [YOLOv8n nine-fruit detector experiment](detector_training/yolov8n-fruit9-v1/README.md)
* App sources: `android/`.
* External agent outputs: `ml/` and `model_exports/`.

The first app uses NanoDet-Plus-m for localization and an existing MobileNetV2-based freshness classifier for local evaluation. Apples, bananas, and oranges are supported. Reference freshness weights are experimental; a validated and appropriately licensed fine-tuned export replaces them through the bundle installer.
