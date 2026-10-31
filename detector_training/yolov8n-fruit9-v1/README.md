# YOLOv8n nine-fruit detector, experiment 1

This experiment fine-tunes official pretrained YOLOv8n weights to detect apple,
banana, grape, lemon, mango, orange, pomegranate, strawberry and tomato. It predicts
fruit identity and bounding boxes. It does not assess freshness.

## Folder guide

- `runtime_environment.json`: points to the isolated CUDA environment. It was moved
  to a task-specific temporary directory on C: to preserve D: space. Use its
  `Scripts/python.exe` with `-m pip`; relocated executable wrappers are not used.
- `pretrained/`: the official starting checkpoint and its URL/hash record.
- `data/images/`: size-bounded JPEG copies for this experiment, separated by split.
- `data/labels/`: matching YOLO text boxes. Source photos and XML files stay unchanged.
- `reports/`: fixed split manifest, exclusions, environment, training history, evaluation and limitations.
- `runs/train/`: training curves, best checkpoint and last checkpoint.
- `artifacts/`: final reviewed checkpoint copy and deployment metadata, when complete.
- `runtime_config/`: local library settings; experiment-specific, not global settings.
- `prepare_data.py`: audits/remaps boxes, groups related images and writes fixed splits.
- `train_detector.py`: trains on CUDA, using validation to select the best checkpoint.
- `evaluate_detector.py`: evaluates the selected checkpoint on the held-out test split.
- `complete_after_training.py`: waits for this run to finish, then runs evaluation,
  export parity, model-card generation and final integrity checks.
- `finalize_report.py` and `verify_experiment.py`: generate the final model card
  from measured results and verify data/artifact hashes and protected model files.

## Data and review policy

Sources are the boxed GitHub fruit dataset and the bounded Open Images subset
documented in `../../datasets/fruit13/SOURCES.md`. Fresh, normal and rotten source
box labels are mapped to fruit identity only. The word normal is not mapped to fresh.

Entire Open Images photos containing a target group-of or depiction box are
excluded. Identical images with inconsistent box annotations are also excluded.
Exact copies are grouped, and 64-bit perceptual hashes within Hamming distance six
are joined transitively, including eight rotation/reflection variants. Groups are
kept in one split. A 70/15/15 split is selected
with seed 42 using only label/group counts, before training or test predictions.
Every class is present in all three splits. Source session/physical-fruit IDs are
unavailable, so perceptual grouping cannot prove complete physical-fruit separation.

Source photos are converted to RGB JPEGs with the long side at most 640 pixels;
aspect ratio is preserved and normalized box coordinates remain unchanged.
Original image dimensions and EXIF orientation are checked before conversion.
VOC coordinates are treated as supplied continuous endpoints, without guessing a
universal one-pixel correction. Geometry checks and sample overlays are recorded.

The Open Images subset includes whole fruit, cut fruit, garnishes and prepared-food
scenes. These source categories are kept as fruit-identity detection examples;
they are not automatically suitable freshness-classifier crops. A sample review
is not exhaustive manual verification of every annotation.

## Important limits

This is a first experimental detector. Mango has only three retained photos: one
in each split. Pomegranate has only nine retained photos. Their test AP has very
little statistical support. Aggregate scores must not hide sparse classes.
The test set comes from the same public sources, not independently collected phones.
The GitHub boxed repository has an Apache-2.0 license, but underlying image
provenance remains unresolved; see the source register before redistribution.

Ultralytics' upstream license terms apply to its software/checkpoints; retain its
license and attribution. This run does not publish data, weights, or an APK online.
No claim of food safety, freshness accuracy, Android latency or all-fruit support
can be inferred from detector metrics.

## Training and evaluation

The intended run is 60 maximum epochs, early-stopping patience 12, input size 416,
batch size 8, AdamW, seed 42 and CUDA on the GTX 1650. Mixed precision and image
caching are disabled. Only best/last checkpoints are retained to limit disk use.
`reports/training_config.json` and `reports/environment.json` record actual settings.

The best validation checkpoint is evaluated once on the frozen test split. Test
predictions do not choose the epoch, threshold, image size, or model architecture.
Report per-class support, precision, recall, AP50 and AP50-95, not just a single
accuracy percentage. Desktop timing is not phone latency.

Use the Python executable in runtime_environment.json. The command order is
prepare_data.py, train_detector.py, evaluate_detector.py, finalize_report.py,
verify_experiment.py. Do not rerun preparation after training starts. The completion
helper replaces the last three manual commands when launched alongside training.
Actual package versions are recorded in reports/requirements-lock.txt. A temporary
training environment may later be removed by the OS; saved models and reports stay
in this experiment folder on D:.

## Later expansion to thirteen fruits

Keep this nine-class checkpoint as a baseline. Add reviewed guava, jujube, lulo and
tamarillo boxes, then fine-tune a thirteen-class model using all previous classes
as well. Training only on four new classes risks forgetting the original nine.
The class order in this experiment differs from the planned thirteen-class list:
always read the model's names/metadata, never assume class IDs match across models.
