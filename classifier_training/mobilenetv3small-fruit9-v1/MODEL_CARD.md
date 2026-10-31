# MobileNetV3Small nine-fruit visible-condition classifier

ImageNet initialized MobileNetV3Small,160x160RGB,18 joint fruit-condition classes. Training: native Windows TensorFlow CPU. GPU was not used. Frozen features train a weighted head, then the last40 backbone layers are fine-tuned with batch normalization frozen. Best candidate selected by validation loss only. Exact configuration: reports/config.json.

Data: 13354 candidate images from Mendeley Sultana8-fruit originals and FRUIT-16K, covering nine fruit identities. Publisher condition folders supply labels. Bohni does not supply freshness labels and is not used. Sources, attribution, licenses and original hashes: datasets/fruit13/SOURCES.md and source metadata. Split counts: {'test': 2004, 'train': 9338, 'val': 2012}. pHash groups and exact hashes do not cross splits. Physical fruit/session independence is unknown; adjacent captures and source backgrounds can inflate performance.

Joint18 test accuracy: 99.65%. Macro F1: 0.9928. Fruit identity accuracy: 99.85%. Test images: 2004.

| Fruit | Test images | Joint18 accuracy | Condition accuracy given correct fruit | Spoiled recall given correct fruit |
|---|---:|---:|---:|---:|
| apple | 59 | 100.00% | 100.00% | 100.00% |
| banana | 358 | 100.00% | 100.00% | 100.00% |
| grape | 59 | 98.31% | 98.31% | 96.67% |
| lemon | 231 | 100.00% | 100.00% | 100.00% |
| mango | 284 | 99.65% | 100.00% | 100.00% |
| orange | 357 | 99.44% | 99.72% | 100.00% |
| pomegranate | 59 | 96.61% | 98.31% | 96.55% |
| strawberry | 336 | 100.00% | 100.00% | 100.00% |
| tomato | 261 | 99.62% | 99.62% | 100.00% |

Known-fruit columns select only the two outputs for the actual fruit; they do not include detector mistakes. This model has no unknown/reject class and does not estimate shelf life, internal quality, or food safety. No real-phone or independent field validation has occurred. Fresh means source-labeled visually fresh, not a safety determination. Allow uncertain output and fruit-identity mismatch rejection in a future app; thresholds require phone validation.

Deployment artifacts: float32 and float16 TFLite, with exact labels/preprocessing/hashes in artifacts/model.json. Logs: logs/train.log; epoch losses: logs/head_epochs.csv and logs/fine_epochs.csv. Checkpoints: checkpoints/best.keras and last.keras. Exports verified against Keras on 90 held-out images. Float16 compresses weights, does not guarantee faster device inference. Existing detector, Classic and six-class freshness model preserved.

Training used 9,338 images; validation 2,012; test 2,004. Best fine-tuning validation epoch:11. Training/preparation before evaluation took405.6seconds. Derived training caches were removed after completion, recovering0.98GiB. Full Classic snapshot verification passed71 files.
