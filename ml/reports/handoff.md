# Handoff

## Status

The ML pipeline is complete and the final bundle is assembled at:

```text
model_exports/mobilenetv3small-sixclass-v1
```

No Android source, app asset, or Gradle file was modified by the ML workflow.

## Bundle contract

| Field | Value |
|---|---|
| Model ID | `mobilenetv3small-sixclass-v1` |
| TFLite file | `freshness.tflite` |
| SHA-256 | `dd519afd78d27d731fc4cef0e9a02b96086e42122a51a768874b8b6fdfc9052e` |
| Size | 3,763,884 bytes |
| Input | float32 `[1, 224, 224, 3]`, RGB, NHWC |
| Output | float32 `[1, 6]` softmax probabilities |
| Resize | stretch to 224 x 224 |
| Normalization | `(pixel - 127.5) / 127.5` per channel |
| Threshold | `0.5833432078361511` |
| Threshold validated | `true` |
| Labels | `fresh_apple`, `fresh_banana`, `fresh_orange`, `rotten_apple`, `rotten_banana`, `rotten_orange` |

The label order is part of the contract. Do not reorder `labels.json` independently of the model.

## Verification performed

- `android/tools/install_model.py` `validate_bundle` accepted the bundle read-only.
- The validator confirmed six labels, the TFLite hash, the model ID, `minScore`, and `thresholdValidated: true`.
- The bundle contains all required files plus parity fixtures, notices, and checksums.
- Six bundled parity fixtures pass with the bundled TFLite interpreter and score tolerance 0.0001.
- Keras-to-TFLite test parity: 180 images, top-1 agreement 100%, maximum absolute probability difference `7.45e-06`.
- `SHA256SUMS.txt` covers the bundle contents except itself.

The install step was intentionally not run. To install, the app owner should use:

```powershell
python android\tools\install_model.py model_exports\mobilenetv3small-sixclass-v1 --replace
```

That command modifies `android/app/src/main/assets`, so it belongs to the Android integration step, not the ML step.

## Evaluation summary

Actual TFLite artifact, untouched 180-image test split:

- Accuracy 97.78%, macro F1 97.78%.
- Condition accuracy 97.78%.
- 1 false-fresh, 3 false-rotten.
- Coverage at threshold 99.44%; accepted accuracy 98.32%.
- All four errors were same-fruit wrong-condition errors.
- The only false-fresh test error scored 0.5154 and was abstained.

The validation split was used to choose the checkpoint and threshold. The test split was not used for model selection.

## App integration requirements

- Preserve RGB channel order; the model was not trained on BGR.
- Use stretch resize to 224 x 224. Do not substitute a center crop unless its effect is tested.
- Normalize with 127.5 mean and 127.5 standard deviation after converting to float32.
- Add the batch dimension and keep NHWC layout.
- Treat class scores as softmax probabilities.
- The app's existing acceptance rule is correct: accept only if `thresholdValidated` is true and the maximum score is at least `minScore`.
- Display an abstention/uncertain result below threshold. Do not silently pick the top class.
- Do not describe an accepted prediction as a food-safety guarantee. The model only recognizes visible dataset conditions.

## Known limitations

- The test set is controlled, small, and not captured with a phone. Generalization to real kitchens, markets, lighting, blur, occlusion, and multiple fruit is unmeasured.
- The validation split saturated, so checkpoint and threshold selection had limited resolution.
- The dataset has no capture-session IDs; perceptual grouping reduces but does not eliminate leakage risk.
- The model has no unknown/OOD class. A high softmax score does not prove the image contains one supported fruit.
- Android latency and camera preprocessing were not measured.
- The threshold was selected on 180 validation images; confidence intervals are wide.

## Related records

- `model_exports/mobilenetv3small-sixclass-v1/MODEL_CARD.md`
- `model_exports/mobilenetv3small-sixclass-v1/evaluation.json`
- `model_exports/mobilenetv3small-sixclass-v1/THIRD_PARTY_NOTICES.md`
- `ml/reports/comparison.md`
- `ml/reports/baseline.md`
- `ml/reports/environment.md`
- `ml/reports/license_audit.md`
- `ml/data/DATA_SOURCES.md`
- `ml/reports/eval_mv3s_finetune_224_v1_test.json`
- `ml/reports/parity_mv3s_finetune_224_v1_test.json`
