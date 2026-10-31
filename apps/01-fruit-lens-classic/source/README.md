# Fruit freshness detection

Android-first offline fruit scanner.

* [Android app and build instructions](android/README.md)
* [Detailed external training-agent prompt](TRAINING_AGENT_PROMPT.md)
* App sources: `android/`.
* External agent outputs: `ml/` and `model_exports/`.

The first app uses NanoDet-Plus-m for localization and an existing MobileNetV2-based freshness classifier for local evaluation. Apples, bananas, and oranges are supported. Reference freshness weights are experimental; a validated and appropriately licensed fine-tuned export replaces them through the bundle installer.
