# MobileNetV3Small six-class v1

## What and why

Fresh/rotten apple,banana,orange;6classes.1,197images:837train/180val/180test. Mendeley8-fruit source subset.12head+4fine epochs, TensorFlow CPU. Test176/180(97.78%), not phone validated.

## Architecture and size

942,582 parameters. Optimized graph inspection is recorded separately.

| File | Bytes | SHA256 |
|---|---:|---|
| weights/freshness.tflite | 3,763,884 | `dd519afd78d27d731fc4cef0e9a02b96086e42122a51a768874b8b6fdfc9052e` |

## Input, outputs and execution

224RGB, bilinear resize, (pixel-127.5)/127.5. Float32TFLite; six labels in source export metadata.

## Training, evaluation and code

Start at `ml/README.md` from the project root. That experiment owns exact configurations, split manifests, logs, evaluation and source attribution. Layers/operator summaries are measurements of the deployed graph where the original architecture is unavailable, not invented original layer counts.

## Version policy

Keep weights immutable. A changed training checkpoint needs a new version folder. Android Classic and Custom have separate package IDs and model assets. Test scores describe the stated evaluation split; they do not guarantee camera accuracy or food safety. Full training data remains local and is not included in Git.
