# NanoDet-Plus-m416

## What and why

COCO80 categories; app uses banana/apple/orange. Official pretrained export; no local training or matched accuracy benchmark.

## Architecture and size

Original architecture parameter/layer count is not independently established for this exact asset. See graph/operator inspection for executable structure.

| File | Bytes | SHA256 |
|---|---:|---|
| weights/nanodet.bin | 2,397,184 | `f7555ca6cb9650ee41e2f0a53fdf46b6e187a2a09792208d8c3eec033b6ac615` |
| weights/nanodet.param | 23,123 | `d79e18ecd8595081bb29fdf0c790a47fcea96fbf58598350569a92e71e915b72` |

## Input, outputs and execution

Input416 BGR, mean103.53/116.28/123.675 and std57.375/57.12/58.395. CPU NCNN2threads; no Vulkan. Output3598 rows of80scores+32distance logits. See Classic TECHNICAL_DETAILS.md.

## Training, evaluation and code

Start at `apps/01-fruit-lens-classic/TECHNICAL_DETAILS.md` from the project root. That experiment owns exact configurations, split manifests, logs, evaluation and source attribution. Layers/operator summaries are measurements of the deployed graph where the original architecture is unavailable, not invented original layer counts.

## Version policy

Keep weights immutable. A changed training checkpoint needs a new version folder. Android Classic and Custom have separate package IDs and model assets. Test scores describe the stated evaluation split; they do not guarantee camera accuracy or food safety. Full training data remains local and is not included in Git.
