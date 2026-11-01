# Selected next detector: YOLO26s

Selected on 1 November 2026 for the next nine-fruit detection experiment. **Training has not started.** This folder currently contains official COCO-pretrained starting weights, not a finished nine-fruit detector or a new mobile app.

## Why this model

The small-scale YOLO26 detector is a reasonable next experiment between our lightweight YOLOv8n and substantially heavier medium/large models. The official documentation reports 9.5 million parameters for the fused deployment architecture, while the full training checkpoint retains extra training components. Its official COCO benchmark is evidence for selecting an experiment, not proof of better performance on our fruit photos or phones.

Official documentation: https://docs.ultralytics.com/models/yolo26/

## Download and identity

- Source: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
- Local file: `pretrained/yolo26s.pt`
- Actual checkpoint size: **20,422,725 bytes**, approximately **20.42 MB** in decimal units.
- SHA256: `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`.
- Publisher release SHA256 and file size both verified.
- `pretrained/provenance.json` records the release asset ID, source URL, hash and exact verification status.

The checkpoint has not yet been executed in our environment or exported to Android. The final NCNN/other mobile export size may differ from the training checkpoint size. No phone latency, accuracy improvement or CPU/GPU speedup is claimed.

## Intended fruit task

Target names: apple, banana, grape, lemon, mango, orange, pomegranate, strawberry, tomato. These are the intended fine-tuning targets, not the label set of the downloaded COCO checkpoint. It must be fine-tuned before it becomes our nine-fruit detector.

Fruit identity and bounding-box location are the task. The existing MobileNetV3-Small freshness classifier remains separate and unchanged.

## What must happen before training

1. Obtain the user's newly found fruit-specific image/box datasets and audit their actual original counts, labels, geometry and diversity.
2. Build a documented nine-class dataset, addressing class imbalance and apple/tomato/pomegranate confusion. Do not train blindly on the imbalanced Zenodo subset; use it as supplementary data only.
3. Protect an independent phone evaluation set and group source/augmentation families before splitting. Check exact and perceptual overlap against previous holdouts.
4. Use an isolated, compatible modern Ultralytics environment. Existing detector experiments use Ultralytics 8.3.40 and must not be upgraded in place just to load this checkpoint. Verify CUDA, a small training batch, output labels and export parity before a full run.
5. Record the actual training configuration, GPU memory use, selected checkpoint, per-fruit metrics and phone latency here. Choose input size/batch from measured accuracy and GTX 1650 4 GB memory, not the checkpoint MB alone.
6. Validate the mobile output contract deliberately. Our current app parser is hard-coded for YOLOv8n's nine-class 416-pixel output; YOLO26 outputs are not a drop-in replacement. The optional NMS-free head also requires deliberate export/parser verification.

Existing Classic, Classic Plus, Custom, all previous models, reports and APKs are preserved. No app points to this checkpoint.
