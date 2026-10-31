# MobileNetV3Small nine-fruit v1

## What and why

Fresh/spoiled for9fruits,18joint classes.13,354candidate images:9,338train/2,012val/2,004test. Mendeley8-fruit originals + FRUIT-16K; no freshness invented from Bohni. Frozen ImageNet features train head(up to30epochs), then12fine epochs; best fine11. TensorFlow2.21/Keras3.15 CPU. Joint identity+condition test accuracy99.65%(1,997/2,004), macro F1.99283. Same-source grouped split; adjacent captures/source backgrounds may inflate scores. No independent camera accuracy.

## Architecture and size

949,506 parameters; last40backbone layers selected for fine-tuning with BatchNorm frozen. Full Keras layer summary and configuration saved beside weights.

| File | Bytes | SHA256 |
|---|---:|---|
| weights/freshness.keras | 9,964,863 | `c03f4ac9d79a4bb9ae2d653665caf20476aa52b0f2d56cded4df9a31afab1a62` |
| weights/freshness_float16.tflite | 1,965,992 | `34190f8beee78b9be9b605fdd12428aba074eda36df38a5f9d5ba63a961267b7` |
| weights/freshness_float32.tflite | 3,795,240 | `fcd1ea470eb0f7eed1c7b1aeb6b1718fcd9f3e1843ec1a3f7cf51d741eb94a81` |

## Input, outputs and execution

160RGB, bilinear stretch, (pixel-127.5)/127.5. Output18softmax probabilities; order fresh/spoiled pairs for apple,banana,grape,lemon,mango,orange,pomegranate,strawberry,tomato. Float16weights with float32IO1.97MB; float32export3.80MB. App uses Fresh/Not fresh/Uncertain while source label IDs are preserved.

## Training, evaluation and code

Start at `classifier_training/mobilenetv3small-fruit9-v1/MODEL_CARD.md` from the project root. That experiment owns exact configurations, split manifests, logs, evaluation and source attribution. Layers/operator summaries are measurements of the deployed graph where the original architecture is unavailable, not invented original layer counts.

## Version policy

Keep weights immutable. A changed training checkpoint needs a new version folder. Android Classic and Custom have separate package IDs and model assets. Test scores describe the stated evaluation split; they do not guarantee camera accuracy or food safety. Full training data remains local and is not included in Git.
