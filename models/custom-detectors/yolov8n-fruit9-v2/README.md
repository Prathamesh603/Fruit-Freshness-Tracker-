# YOLOv8n nine-fruit v2

## What and why

Nine fruits.4,936images:3,761train/591val/584test. Bohni v13 eight-fruit subset plus frozen v1 data.43epochs, best31, GTX1650 CUDA. Combined test mAP50:79.01%; mAP50-95:62.04%. Original frozen test mAP50:52.36%, below v1(58.68%). Lemon remains weak; v2 is not a universal improvement.

## Architecture and size

3,012,603 unfused parameters; fused checkpoint graph3,007,403. Detailed module/layer tables saved beside weights.

| File | Bytes | SHA256 |
|---|---:|---|
| weights/detector.pt | 6,218,275 | `440c4ef36dc825878d1bd84a58eb91885726b9714774ddcc42081b11341d9419` |
| weights/detector.torchscript | 12,384,862 | `f91fce5e7f991afc813c7feb3d4cea5b76f988aace576333bb909061db03eaf4` |
| weights/ncnn/detector.bin | 6,068,488 | `f359a44c5916ed7f37c2392aa19bd8153000c183391bdb7a3b038479c315d70e` |
| weights/ncnn/detector.param | 16,547 | `6fea23c3582bd94bbf0ef316b93ce973d5f7f1b57dcd3bd19eab9e6505330e7a` |

## Input, outputs and execution

416RGB/255, letterbox114. NCNN input in0; output out0 is13x3549:xywh then9scores. FP16stored weights, CPU2threads, no Vulkan, FP16arithmetic disabled. App NMS0.5/score0.4/max5regions are engineering settings, not calibrated accuracy thresholds.

## Training, evaluation and code

Start at `detector_training/yolov8n-fruit9-v2/MODEL_CARD.md` from the project root. That experiment owns exact configurations, split manifests, logs, evaluation and source attribution. Layers/operator summaries are measurements of the deployed graph where the original architecture is unavailable, not invented original layer counts.

## Version policy

Keep weights immutable. A changed training checkpoint needs a new version folder. Android Classic and Custom have separate package IDs and model assets. Test scores describe the stated evaluation split; they do not guarantee camera accuracy or food safety. Full training data remains local and is not included in Git.
