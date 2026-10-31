# Model comparison and selection

## Candidates

| Candidate | Validation | Test | Size | License | Decision |
|---|---|---|---|---|---|
| Existing nine-class ripeness model | Not evaluated on this split | 58.33% accuracy, 62.87% macro F1, 29 unripe outputs | 2.67 MB | No repository LICENSE; training data license unknown | Rejected |
| MobileNetV3Small frozen head | 100% macro F1, validation loss 0.08084, best epoch 8 | Not run; lost the pre-registered validation tie-break | 4.37 MB Keras checkpoint | Apache-2.0 backbone | Not selected |
| MobileNetV3Small fine-tuned last 40 layers | 100% macro F1, validation loss 0.02551, best epoch 1 | 97.78% accuracy, 97.78% macro F1, 1 false-fresh, 3 false-rotten | 3.76 MB TFLite | Apache-2.0 backbone | Selected |

The external candidate is not a controlled architecture comparison: it was trained on a different dataset and predicts a different nine-class taxonomy. The two MobileNetV3Small runs use the same split, preprocessing, seed, and augmentation.

## Selection method

The selection rule was written to `ml/reports/model_selection.json` before test evaluation:

1. Maximise validation macro F1.
2. Break ties with lower validation loss.
3. If still tied, prefer the frozen backbone because it has fewer trainable parameters.

Both internal candidates saturated the 180-image validation split at 100% macro F1. The fine-tuned checkpoint had the lower validation loss and was selected. The test split was not used to choose the architecture, checkpoint, or threshold.

This is a weak model-selection signal. A saturated validation set cannot distinguish a model that is robustly correct from one that has merely fitted the controlled dataset. Test evaluation is therefore reported as evidence, not as independent proof of field performance.

## Selected model test results

Actual TFLite artifact on 180 test images:

| Metric | Value |
|---|---:|
| Accuracy | 97.78% |
| Macro F1 | 97.78% |
| Condition accuracy | 97.78% |
| Log loss | 0.03916 |
| Expected calibration error, 10 bins | 0.02979 |
| False-fresh errors | 1 |
| False-rotten errors | 3 |
| Threshold | 0.583343 |
| Coverage at threshold | 99.44% (179/180) |
| Accepted accuracy | 98.32% (176/179) |
| Accepted macro F1 | 98.31% |

All four errors were same-fruit wrong-condition errors. No test image was assigned to the wrong fruit. The single false-fresh error had score 0.5154 and was abstained by the selected threshold. The remaining three accepted errors were fresh-to-rotten or rotten-to-fresh mistakes within the same fruit.

Approximate uncertainty, using 95% Wilson intervals:

- Accuracy: 94.4% to 99.1%.
- False-fresh rate among 90 fresh images: 0.2% to 6.0%.
- False-rotten rate among 90 rotten images: 1.1% to 9.3%.

These intervals describe sampling uncertainty on this controlled test set, not farm or consumer-camera performance.

## Existing-model comparison

The existing candidate was tested under the same manifest:

| Metric | Existing candidate | Selected model |
|---|---:|---:|
| Accuracy (unripe = error) | 58.33% | 97.78% |
| Macro F1 | 62.87% | 97.78% |
| Valid six-class coverage | 83.89% | 100% |
| Unripe outputs | 29 | Not applicable |
| Condition accuracy | 66.67% | 97.78% |
| False-fresh | 11 | 1 |
| False-rotten | 20 | 3 |

The existing candidate was rejected for accuracy and false-fresh behaviour, for producing a class the app cannot display, and for unresolved code, weight, and training-data licensing. Details: `ml/reports/baseline.md`.

## Deployment comparison

- The selected TFLite file is float32, 3,763,884 bytes, input `[1,224,224,3]`, output `[1,6]`.
- It uses only built-in TensorFlow Lite operators and passed Android `validate_bundle` checks.
- Keras-to-TFLite top-1 agreement is 100% on the test split, with maximum absolute probability difference `7.45e-06`.
- Desktop CPU, two LiteRT threads: inference p50 1.287 ms, p95 2.824 ms; desktop preprocess plus inference p50 36.590 ms, p95 108.031 ms.
- Android latency and on-device camera performance were not measured. No emulator or phone benchmark exists.

## Decision

The fine-tuned MobileNetV3Small is the only candidate that:

- satisfies the six-class Android contract;
- has a documented open-source backbone licence;
- has a reproducible, group-aware split and a released TFLite artifact;
- passed class-name, threshold-validation, label-count, byte-size, tensor-shape, dtype, and parity checks;
- has substantially fewer false-fresh errors than the examined existing model.

Its main unresolved risk is generalisation. The test set is controlled, small, and not from phone cameras. The threshold abstains on one known false-fresh error, but it is not an out-of-distribution detector. The app must not describe an accepted prediction as a food-safety determination.

Reports:

- `ml/reports/eval_mv3s_finetune_224_v1_test.json`
- `ml/reports/predictions_mv3s_finetune_224_v1_test.csv`
- `ml/reports/parity_mv3s_finetune_224_v1_test.json`
- `ml/reports/benchmark_mv3s_finetune_224_v1.json`
- `model_exports/mobilenetv3small-sixclass-v1/MODEL_CARD.md`
