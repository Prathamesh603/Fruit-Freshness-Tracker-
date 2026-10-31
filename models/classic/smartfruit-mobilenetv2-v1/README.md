# SmartFruit MobileNetV2 based

## What and why

Existing JosephMoustaid/smart-fruit-models-notebooks model. Nine classes: apple/banana/orange times fresh/rotten/unripe. Local training not performed. Original training dataset/split and independently measured accuracy unavailable. Redistribution terms unresolved; retained for local baseline comparison.

## Architecture and size

Original trainable parameter/layer count cannot be recovered reliably from the optimized TFLite graph alone. Graph weight and operator counts are separately measured in inspection.json.

| File | Bytes | SHA256 |
|---|---:|---|
| weights/freshness.tflite | 2,673,592 | `e09307ca0aacdfc83666ca8ff25c02659d6323a1372409b390561619778b10a4` |

## Input, outputs and execution

96RGB, nearest resize, pixel/255. LiteRT CPU2threads. Nine probability outputs; no unsupported-fruit rejection.

## Training, evaluation and code

Start at `apps/01-fruit-lens-classic/TECHNICAL_DETAILS.md` from the project root. That experiment owns exact configurations, split manifests, logs, evaluation and source attribution. Layers/operator summaries are measurements of the deployed graph where the original architecture is unavailable, not invented original layer counts.

## Version policy

Keep weights immutable. A changed training checkpoint needs a new version folder. Android Classic and Custom have separate package IDs and model assets. Test scores describe the stated evaluation split; they do not guarantee camera accuracy or food safety. Full training data remains local and is not included in Git.
