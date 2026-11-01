# Alternative pretrained models — checked 1 November 2026

The user accepts approximately 20–50 MB models and wants broader fruit detection and freshness coverage. This review inspected primary model cards, actual configuration/label files, release metadata and existing project benchmarks. No new model weights were downloaded, run, trained, exported, or installed in an app. Sizes below are decimal MB of published checkpoint files, not Android RAM or final APK/export sizes.

## Recommendation

**First new detector to benchmark: YOLOE-26s-seg**, with a fixed list of our fruit names. Its official checkpoint is 31,072,171 bytes (31.07 MB). YOLOE-11s-seg is a second candidate at 27,803,986 bytes (27.80 MB). These are open-vocabulary models: a name can be supplied without training a new class head, but that does NOT establish reliable recognition of every requested fruit. Fruit-specific accuracy, distance performance, false positives and Android latency remain unmeasured.

Set the fruit prompts on the development computer, then bake those classes into an exported model. The official docs describe static ONNX/LiteRT and other exports. Text setup requires a separate text encoder (YOLOE-26 docs describe approximately 254 MB), so the original Python setup/download footprint is larger than the 31 MB detector. The text encoder need not be a runtime input to the static export. Exported classes are fixed; changing them requires a new export.

This would require a separately verified integration: it is not a drop-in replacement for our current hard-coded nine-class NCNN output parser. No NCNN compatibility or Android speed claim is established by this review.

Sources:
- https://docs.ultralytics.com/models/yoloe/
- https://github.com/ultralytics/assets/releases/tag/v8.4.0
- https://github.com/THU-MIG/yoloe
- Saved actual release asset sizes/digests: `yoloe-assets.json`.

## Existing fixed-class fruit detectors

| Model | Actual published checkpoint | Scope | Assessment |
|---|---:|---|---|
| Piyu12 fruit-veg YOLO11m | 40,565,868 bytes / 40.57 MB | 35 produce categories; eight of our nine targets, no lemon | Already benchmarked locally; not an established upgrade |
| Aniket2003333333 fruit-veg YOLO11l | 51,245,849 bytes / 51.25 MB | Same listed 35 categories, including eight current targets, no lemon | Secondary evaluation candidate; slightly above 50 MB, no matched local evaluation |

The 35 categories include vegetables and Green_Orange/Orange labels, not 35 distinct fruit species. Fruit labels include apple, banana, cherry, coconut, grape, kiwi, mango, melon, orange, peach, pear, pineapple, pomegranate, strawberry, tomato, and watermelon. Published model-card metrics are not comparable with our test results without matched evaluation.

The YOLO11m reference was previously checked at 416 and 640 pixels. On the old frozen eight-shared-class pool, its mAP50 at 640 was 48.44%, compared with 59.55% for our v1 detector on the same eight-class scope. This was not an independent phone benchmark; source overlap was unknown, and several fruit classes had very few test examples. Therefore, this does not prove universal inferiority, but we should not recommend it as an upgrade based on its model-card 92.5% mAP50 alone.

Sources:
- https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector
- https://huggingface.co/Aniket2003333333/fruit-veg-yolo11l-detector
- Project evidence: `detector_training/reference-yolo11m-bohni/NATIVE_SIZE_640.md`.

## Freshness classifiers: actual coverage matters

### Aryaman9999/freshness2 — ResNet-50

Actual `config.json` has 24 joint labels: Fresh/Rotten for **12 produce types**: apple, banana, cucumber, grape, guava, jujube, okra, orange, pomegranate, potato, strawberry, tomato. This includes nine fruit types when tomato is included, but a different nine from our model: **mango and lemon are absent**, while guava and jujube are added.

The published FP32 safetensors file is **94,483,312 bytes / 94.48 MB**. FP16 parameter storage would be roughly 47 MB, which is an estimate, NOT a verified exported artifact. Conversion, operator support, prediction parity, peak memory and phone latency must be measured. The model card gives no validation metrics. Source dataset: `jojogo9/freshness_of_fruits_and_veges_256`.

This is an available trained baseline to investigate for guava/jujube coverage, not a proven accuracy upgrade or an existing ready-to-install 47 MB mobile model.

Source: https://huggingface.co/Aryaman9999/freshness2

### Aryaman9999/Freshness-Fruit_Vegies — ViT

Same actual 24-label produce coverage as the ResNet model, but **343,291,648 bytes / 343.29 MB**. This is not a 20–50 MB option. Do not confuse 24 labels with 24 fruit species.

Source: https://huggingface.co/Aryaman9999/Freshness-Fruit_Vegies

### Piyu12 / Aniket MobileNetV2 freshness models

Both model cards describe fresh/stale binary classification, trained on apple, banana, bitter gourd, capsicum, orange and tomato. That is six produce types, not freshness support for all 35 detector categories. **Mango, grape, lemon, pomegranate and strawberry from our current nine are not listed.**

Actual state dictionaries are **9,158,475 bytes / 9.16 MB**. Their state-dictionary SHA256 metadata is identical (`10ddb94422bb09dec7eef3559508b8c3bcb3db45441cf72b16988b73cbb78d84`); they should not be counted as two independent trained baselines. This identity is based on published LFS metadata, not a fresh binary download. TorchScript files differ in hash, which can reflect serialization/export differences.

Sources:
- https://huggingface.co/Piyu12/fruit-veg-freshness-mobilenetv2
- https://huggingface.co/Aniket2003333333/fruit-veg-freshness-mobilenetv2

### nathansekar/food-freshness-detector — ResNet18

Actual weights are **46,914,522 bytes / 46.91 MB**. Outputs are Fresh, Slightly Spoiled, Rotten. However, its model card explicitly states **synthetic training images** and supplies no verified fruit-by-fruit real-camera coverage. The word detector in the repository name does not make it a bounding-box detector; it is an image classifier. Not our preferred real-fruit replacement.

Source: https://huggingface.co/nathansekar/food-freshness-detector

## Broader research with incomplete deployment evidence

**Multi-task NoisyViT / Freshness44** studies 22 produce types and 44 combined fresh/rotten labels. The paper links the Freshness44 dataset, but this review did not establish a downloadable trained checkpoint or a verified 20–50 MB export. Treat it as research/a data expansion direction, not a ready app model. It classifies images/crops rather than providing the full multi-fruit detection pipeline.

Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12526848/

**codernayeem/fruit-classification-dataset** describes 13 fruits with Good/Medium/Bad grades and fruit-specific MobileNetV3-Large/EfficientNet-B0/ResNet-50 models. The README lists weights under `Result/` and links an external Google Drive folder. The complete checked GitHub tree (`truncated: false`, revision in `thirteen-fruit-tree.json`) contains **zero model weight files**. The Drive weight availability was not verified. Moreover, this is a collection of per-fruit classifiers, not automatically a single 20–50 MB model covering all 13. Its 13 fruits differ substantially from our nine, excluding apple, orange and pomegranate.

Source: https://github.com/codernayeem/fruit-classification-dataset

Other candidates were screened: large ViT freshness models, tiny-data EfficientNet experiments, a binary MobileNetV3 TFLite export using the Project-AgML corpus, and repositories without clear labels/training data. Downloadable weights or a high reported score alone do not establish better fruit coverage or independent-camera robustness. Model-index and metadata snapshots are retained alongside this report.

## Practical next experiment

1. Benchmark official YOLOE-26s and YOLOE-11s on the same fixed fruit/confusing-object photos as our existing detectors. Include distance changes and the apple/tomato/pomegranate failures.
2. Keep our current nine-fruit freshness classifier as a baseline; additional detected fruits must show freshness unavailable until a suitable classifier is validated.
3. If guava/jujube expansion is a priority, investigate ResNet-50 `freshness2` and a verified FP16 export as an external comparison, preserving unsupported mango/lemon routing.
4. For substantially more freshness categories, use verified multi-fruit labeled data to train a common classifier rather than assume a generic binary classifier supports every detected fruit.

No model in this review is claimed best on our phones. The strongest new *candidate to test* is YOLOE-26s; a verified broad, real-camera freshness replacement within 20–50 MB has not been established.
