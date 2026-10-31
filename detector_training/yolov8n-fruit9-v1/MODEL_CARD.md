# YOLOv8n nine-fruit detector: first experiment

## What this model does

Detects apple, banana, grape, lemon, mango, orange, pomegranate, strawberry and
tomato in an image. It outputs fruit identities and bounding boxes. It does NOT
predict freshness, internal quality, remaining shelf life, or food safety.

This is a completed experimental run, not a validated nine-fruit Android release.
Sparse categories and same-source testing limit what the reported scores establish.

## Training

- Architecture: YOLOv8n, initialized with official COCO-pretrained weights.
- Fruit classes: 9; parameters: 3,012,603.
- Exact class order: `apple, banana, grape, lemon, mango, orange, pomegranate, strawberry, tomato`.
- Training device: NVIDIA GeForce GTX 1650; CUDA build 12.1.
- PyTorch: 2.5.1+cu121; Ultralytics: 8.3.40.
- Completed epochs: 60; maximum requested: 60; early-stopping patience: 12.
- Input size: 416 pixels; batch size: 8; optimizer: AdamW; initial LR: 0.001.
- Seed: 42; deterministic setting enabled; AMP disabled; disk/RAM image caching disabled.
- Only best/last checkpoints retained. The framework's validation fitness selected best.pt.
- Training invocation duration: 41.60 minutes.
- Full settings: reports/training_config.json; epoch history: runs/train/results.csv.

## Data preparation

The collected pool contained 3,615 images. After source filtering,
3,221 remain. Exclusions: {'source_image_has_target_group_or_depiction_box': 176, 'exact_pixels_with_inconsistent_annotations': 218}.

Final split: 2,255 train,
483 validation, 483 test.
There is no exact SHA256 or defined perceptual-group overlap between these splits.
Groups use 64-bit pHash, Hamming radius six, with eight rotation/reflection variants
and transitive union. Physical-fruit/session IDs are unavailable; these checks
cannot prove complete physical-fruit separation.

Split manifest SHA256: `b3c6d6fb576c13ca89fba8e5832f2ba4c3eec3b521924c82dacfba32d37ca889`.
The split was fixed before model training and test prediction. The held-out test
does not select epochs or deployment thresholds.

Prepared images are aspect-preserving RGB JPEG copies with the long side at most
640 pixels. Ultralytics letterboxes them to 416 for this experiment. The source
condition labels fresh/normal/rotten are collapsed to fruit identity only.
Source photos and existing classifier/app files are unchanged.

## Held-out test results

- Test images: 483; labelled fruit boxes: 1187.
- mAP50: 58.68%.
- mAP50-95: 47.72%.
- Framework mean precision: 65.81%.
- Framework mean recall: 58.62%.

| Fruit | Test images | Test boxes | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| apple | 219 | 281 | 98.6% | 97.2% | 98.9% | 97.9% |
| banana | 267 | 483 | 98.7% | 99.8% | 99.5% | 90.2% |
| grape | 2 | 10 | 41.8% | 30.0% | 19.0% | 8.6% |
| lemon | 8 | 29 | 70.6% | 24.9% | 51.7% | 40.3% |
| mango | 1 | 1 | 100.0% | 0.0% | 0.0% | 0.0% |
| orange | 240 | 338 | 98.5% | 99.4% | 99.5% | 98.4% |
| pomegranate | 1 | 1 | 28.9% | 100.0% | 99.5% | 49.7% |
| strawberry | 6 | 31 | 31.2% | 27.8% | 31.5% | 20.9% |
| tomato | 7 | 13 | 24.0% | 48.5% | 28.6% | 23.5% |

AP uses confidence floor 0.001, class-aware NMS IoU 0.7, and max_det 300.
Precision/recall use the framework's maximum-F1 operating point; they are not
performance at a separately validated app confidence threshold. The qualitative
sample sheet uses 0.25 solely as an illustrative prediction setting.

For mango, the framework reports precision 1.0 despite recall and AP being zero;
this is an empty-prediction convention, not successful mango detection. Do not
interpret that precision number alone as quality.

The sample sheet was inspected after evaluation. It shows additional wrong-class
boxes on grapes/lemons, mango pieces predicted as citrus, and extra boxes on
pomegranate seeds and prepared-food scenes. Pomegranate's high AP50 comes from
one test image/one labelled box and does not establish reliability. This model
needs more reviewed, diverse data before a nine-fruit deployment claim.

Confusion matrices and PR curves are in runs/test/. Read reports/test_evaluation.json
and reports/per_class_test.csv for the machine-readable results. The sample sheet
is reports/test_prediction_samples.jpg; it is not a complete failure analysis.

## Critical limitations

- Mango has one training photo, one validation photo and one test photo after
  conservative source filtering. Pomegranate also has only one test photo.
  A high or low AP for these classes does not establish generalization.
- Class/source imbalance is severe. Rare-class AP must be reported separately.
- Open Images includes cut fruit and prepared-food scenes. Detector outputs from
  such photos must not automatically be treated as valid freshness assessments.
- No independent phone-photo dataset, unsupported-fruit test, field study, battery
  benchmark, or Android latency measurement has been completed.
- The source annotation convention and image provenance have limitations listed
  in README.md and ../../datasets/fruit13/SOURCES.md.
- Broader or more independent evaluation, and more mango/pomegranate data, are
  required before product-readiness claims.

## Artifacts and export

| File | Bytes | SHA256 |
|---|---:|---|
| fruit-detector-yolov8n-9class-v1.pt | 6,215,139 | `813fd499ece31bffd60d6f81cf75709ec43cd3a4d5b9a698227c7403b10dd190` |
| fruit-detector-yolov8n-9class-v1.torchscript | 12,379,870 | `a14d08fe9a25cd126f3ff75f2af3a8a7b8a2616727e838267db55a9fe091ef2d` |

artifacts/model.json specifies preprocessing, class IDs, and raw output layout.
TorchScript parity passed on 12 held-out sample images; maximum
absolute raw-output difference: 0.00344849.

Desktop CPU raw-forward median: 26.97 ms;
p95: 39.90 ms, with four CPU threads. This excludes
decode, preprocessing, NMS, classifier work, and rendering. It is NOT Android or
tap-to-result latency. CUDA validation timings are recorded separately in evaluation JSON.

## License and later expansion

Retain Ultralytics' upstream software/checkpoint license terms and all dataset
attribution. The source GitHub image provenance remains unresolved, so no public
redistribution clearance is claimed. This run performs no public upload.

For thirteen fruits, add reviewed guava/jujube/lulo/tamarillo boxes and retain data
for all original nine classes. Train a separate thirteen-class head/checkpoint;
do not overwrite this baseline or assume its class IDs equal the thirteen-class
taxonomy. Existing Fruit Lens Classic and the MobileNetV3 freshness model remain separate.
