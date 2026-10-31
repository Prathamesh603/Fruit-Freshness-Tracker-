# Baseline: examined existing model

Before training, the repository's candidate model was evaluated against the project's untouched test split.

## Candidate identity

| Field | Value |
|---|---|
| Repository | https://github.com/JosephMoustaid/smart-fruit-models-notebooks.git |
| Commit | `200d8dffb48679567777a6ab5adc1d87eeea1c88` |
| Local clone | `ml/reference/candidate_repo` |
| Artifact | `ml/reference/candidate_repo/ripeness_model.tflite` |
| Artifact SHA-256 | `e09307ca0aacdfc83666ca8ff25c02659d6323a1372409b390561619778b10a4` |
| Size | 2,673,592 bytes |
| Input | float32 `[1, 96, 96, 3]`, RGB, values scaled by `1/255` |
| Output | float32 `[1, 9]` |
| Architecture | MobileNetV2 backbone, ImageNet weights, frozen base, global average pooling, dense 128 ReLU, dropout 0.2, 9-class softmax |
| Labels | fresh apple, fresh banana, fresh orange, rotten apple, rotten banana, rotten orange, unripe apple, unripe banana, unripe orange |
| Repository license | No LICENSE file; no license declared |
| Training data | External “Fruit Ripeness: Unripe, Ripe, and Rotten” dataset referenced by notebook; not included and not licensed in the repository |

The reference notebook builds the model with a `96 x 96` input, rescales pixels to `[0, 1]`, trains with augmented `ImageDataGenerator` input, and uses 12 epochs. The clone does not contain the training data, so the artifact cannot be retrained from the repository alone.

## Evaluation protocol

The candidate was run on the same 180-image test manifest used for the project model: 30 images per canonical class. Its nine outputs were mapped as follows:

- `freshapples` -> `fresh_apple`
- `freshbanana` -> `fresh_banana`
- `freshoranges` -> `fresh_orange`
- `rottenapples` -> `rotten_apple`
- `rottenbanana` -> `rotten_banana`
- `rottenoranges` -> `rotten_orange`
- any `unripe *` -> not a valid six-class output

Unripe outputs were counted as errors in the primary accuracy. They were also counted separately so the six-class-only behaviour is visible.

Report: `ml/reports/eval_candidate_9class_test.json`
Predictions: `ml/reports/predictions_candidate_9class_test.csv`

## Results

| Metric | Value |
|---|---:|
| Images | 180 |
| Six-class mapped accuracy (unripe = error) | 58.33% |
| Six-class macro F1 | 62.87% |
| Valid six-class coverage | 83.89% |
| Unripe outputs | 29 |
| Accuracy on valid six-class outputs | 69.54% |
| Fruit-only accuracy | 83.33% |
| Condition accuracy over all images | 66.67% |
| False-fresh errors | 11 |
| False-rotten errors | 20 |

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `fresh_apple` | 0.71 | 0.50 | 0.59 | 30 |
| `fresh_banana` | 1.00 | 0.50 | 0.67 | 30 |
| `fresh_orange` | 0.58 | 0.60 | 0.59 | 30 |
| `rotten_apple` | 0.48 | 0.37 | 0.42 | 30 |
| `rotten_banana` | 0.82 | 0.90 | 0.86 | 30 |
| `rotten_orange` | 0.68 | 0.63 | 0.66 | 30 |

Confusion matrix rows are the six actual classes; columns are the six canonical predictions followed by `unripe`:

```json
[[15, 0, 8, 4, 0, 0, 3],
 [ 0,15, 0, 2, 6, 0, 7],
 [ 0, 0,18, 2, 0, 6, 4],
 [ 4, 0, 1,11, 0, 2,12],
 [ 0, 0, 0, 0,27, 1, 2],
 [ 2, 0, 4, 4, 0,19, 1]]
```

The candidate is trained on a different taxonomy and a different dataset. This is a baseline comparison, not a controlled architecture experiment.

## Decision

The candidate was rejected for this app:

1. It emits a ninth-class family (`unripe`) that the six-class Android contract and UI do not support.
2. Its accuracy and macro F1 are substantially below the selected model on the same test images.
3. It produces many more false-fresh errors: 11 versus 1 for the selected model.
4. The repository has no LICENSE file and the training-data license is not established, so the artifact cannot be redistributed in the app bundle.
5. Its training data is not included, so the reported artifact cannot be independently reproduced from the clone.

The artifact remains only under `ml/reference/` for comparison and is not part of the final bundle.
