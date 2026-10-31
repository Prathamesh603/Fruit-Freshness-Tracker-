# YOLOv8n nine-fruit v1

## What and why

Nine fruits.3,221prepared images.60epochs, best49, GTX1650 CUDA. GitHub fruit freshness boxes + Open Images. Test mAP50:58.68%; mAP50-95:47.72%. Mango0% with tiny support. Not camera validated.

## Architecture and size

3,012,603 unfused parameters. Detailed architecture saved beside weights.

| File | Bytes | SHA256 |
|---|---:|---|
| weights/fruit-detector-yolov8n-9class-v1.pt | 6,215,139 | `813fd499ece31bffd60d6f81cf75709ec43cd3a4d5b9a698227c7403b10dd190` |

## Input, outputs and execution

416RGB/255, letterbox114. Class order apple,banana,grape,lemon,mango,orange,pomegranate,strawberry,tomato. See training model.json for raw tensor contract.

## Training, evaluation and code

Start at `detector_training/yolov8n-fruit9-v1/MODEL_CARD.md` from the project root. That experiment owns exact configurations, split manifests, logs, evaluation and source attribution. Layers/operator summaries are measurements of the deployed graph where the original architecture is unavailable, not invented original layer counts.

## Version policy

Keep weights immutable. A changed training checkpoint needs a new version folder. Android Classic and Custom have separate package IDs and model assets. Test scores describe the stated evaluation split; they do not guarantee camera accuracy or food safety. Full training data remains local and is not included in Git.
