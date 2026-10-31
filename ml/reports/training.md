# Training report

## Data

- Manifest: `ml/data/manifests/splits_v1.csv`
- SHA-256: `47cfa8f903e04a59b9a1a059ff4d5f2267bd2010104e7f115a2a48121e8e1cd7`
- Train: 837 images
- Validation: 180 images, 30 per class
- Test: 180 images, 30 per class
- Split unit: 64-bit pHash group, not individual file
- Augmented source archive: not used

## Architecture

- MobileNetV3Small, ImageNet initialization from Keras Applications.
- Input 224 x 224 RGB.
- Global average pooling, dropout 0.2, six-class softmax.
- 942,582 parameters.
- Training resize 256, then random crop from 197-256 pixels to 224.
- Augmentation: horizontal flip, brightness delta 12/255, contrast 0.85-1.15, saturation 0.85-1.15.
- No test-time augmentation.
- Loss: sparse categorical cross-entropy.
- Optimizer: Adam with gradient-norm clipping 1.0.
- Scheduler: none; fixed learning rate per stage.
- Class weights or resampling: none; class counts are approximately balanced.
- Seed: 42; deterministic op setting enabled.

## Toolchain smoke check

Before training, a minimal untrained MobileNetV3Small six-output TFLite file was exported and invoked on a real image with the desktop LiteRT runtime. It returned the expected six probabilities and a uniform untrained result.

```text
ml/reference/smoke_untrained.tflite
```

SHA-256:

```text
02864cc01162dec7f41feb62e1cf239d4b2c8147d391bb11141595cb9df0d8e0
```

This file proves the export/runtime path only. It is untrained, is not a candidate, and is not included in any bundle.

## Stage 1: frozen backbone

Config: `ml/configs/mv3s_frozen_224_v1.json`

| Setting | Value |
|---|---:|
| Backbone trainable | No |
| Epochs requested | 14 |
| Early-stopping patience | 4 |
| Batch size | 24 |
| Learning rate | 0.001 |
| Best epoch | 8 |
| Epochs completed | 12 |
| Best validation macro F1 | 1.0 |
| Best validation loss | 0.08083995431661606 |
| Final validation loss | 0.052258193492889404 |
| Duration | 130.89 seconds |

The best epoch is the checkpoint with the highest validation macro F1, not the final epoch. Once validation macro F1 reached 1.0, later epochs could not improve it; the checkpoint was saved at the first maximum.

Checkpoint:

```text
ml/checkpoints/mv3s_frozen_224_v1/best.keras
```

SHA-256:

```text
a3bd563a97ed2dcffb4e084c1ddf9ff425eddd4ac003643c519213b3b144454c
```

## Stage 2: fine-tune last 40 layers

Config: `ml/configs/mv3s_finetune_224_v1.json`

| Setting | Value |
|---|---:|
| Initialization | Stage 1 best checkpoint |
| Backbone trainable | Last 40 layers |
| Batch normalization | Frozen |
| Epochs requested | 8 |
| Early-stopping patience | 3 |
| Batch size | 24 |
| Learning rate | 0.0001 |
| Best epoch | 1 |
| Epochs completed | 4 |
| Best validation macro F1 | 1.0 |
| Best validation loss | 0.02550596185028553 |
| Final validation loss | 0.006495602894574404 |
| Duration | 44.67 seconds |

Validation macro F1 first reached 1.0 at epoch 1 and never exceeded it. Model checkpointing therefore retained the epoch-1 weights. The training loss and later validation loss continued to fall, but the validation macro F1 did not improve beyond the saturated maximum.

Checkpoint:

```text
ml/checkpoints/mv3s_finetune_224_v1/best.keras
```

SHA-256:

```text
d7b86370692bceb18fbf4c829314a058b331241c368817a94f29414c26639b67
```

## Validation history

Stage 1:

| Epoch | Val accuracy | Val macro F1 | Val loss |
|---:|---:|---:|---:|
| 1 | 0.7556 | 0.7487 | 0.78566 |
| 2 | 0.9389 | 0.9395 | 0.39568 |
| 3 | 0.9444 | 0.9458 | 0.26397 |
| 4 | 0.9778 | 0.9778 | 0.18446 |
| 5 | 0.9889 | 0.9889 | 0.14115 |
| 6 | 0.9944 | 0.9944 | 0.11427 |
| 7 | 0.9944 | 0.9944 | 0.09557 |
| 8 | 1.0000 | 1.0000 | 0.08084 |
| 9 | 1.0000 | 1.0000 | 0.07224 |
| 10 | 1.0000 | 1.0000 | 0.06299 |
| 11 | 1.0000 | 1.0000 | 0.05804 |
| 12 | 1.0000 | 1.0000 | 0.05226 |

Stage 2:

| Epoch | Val accuracy | Val macro F1 | Val loss |
|---:|---:|---:|---:|
| 1 | 1.0000 | 1.0000 | 0.02551 |
| 2 | 0.9944 | 0.9944 | 0.01656 |
| 3 | 1.0000 | 1.0000 | 0.01478 |
| 4 | 1.0000 | 1.0000 | 0.00650 |

The validation set saturated, so these numbers do not establish that the model is robust outside the source distribution.

## Selection

The pre-registered selection policy is in `ml/reports/model_selection.json`:

1. Maximize validation macro F1.
2. Break ties with lower validation loss.
3. If still tied, prefer the frozen backbone.

The fine-tuned run won the validation-loss tie-break. Test metrics were generated only after this decision. No test image was used to choose the architecture, checkpoint, augmentation, or threshold.

## Export

The selected Keras checkpoint was converted with `ml/scripts/export_tflite.py` to:

```text
ml/checkpoints/mv3s_finetune_224_v1/export/freshness.tflite
```

SHA-256:

```text
dd519afd78d27d731fc4cef0e9a02b96086e42122a51a768874b8b6fdfc9052e
```

The TFLite model is 3,763,884 bytes with float32 `[1,224,224,3]` input and float32 `[1,6]` output.

## Reproducibility caveats

- ImageNet weight download requires network access on a fresh machine.
- CPU oneDNN may produce small numerical differences across CPU or library versions.
- Rerunning the same run ID overwrites its checkpoint directory.
- Training was fast enough that resume support was not needed; no interrupted run was resumed.
- The source split should not be regenerated after seeing test results.
