# Fruit Freshness ML Pipeline

This directory contains the reproducible training, evaluation, and Android export pipeline for `mobilenetv3small-sixclass-v1`.

The model classifies one visible apple, banana, or orange as fresh or visibly spoiled. It is not a food-safety system and has no unknown/out-of-distribution class.

## Final artifacts

| Path | Purpose |
|---|---|
| `model_exports/mobilenetv3small-sixclass-v1/` | Final Android candidate bundle |
| `ml/checkpoints/mv3s_finetune_224_v1/best.keras` | Selected Keras checkpoint |
| `ml/checkpoints/mv3s_finetune_224_v1/export/freshness.tflite` | Staging TFLite artifact |
| `ml/data/manifests/splits_v1.csv` | Immutable grouped split manifest |
| `ml/reports/` | Evaluation, parity, benchmark, comparison, and handoff records |

Bundle SHA-256:

```text
dd519afd78d27d731fc4cef0e9a02b96086e42122a51a768874b8b6fdfc9052e
```

## Results at a glance

- Test accuracy: 97.78% (176/180).
- Test macro F1: 97.78%.
- False-fresh: 1; false-rotten: 3.
- All test errors were same-fruit wrong-condition errors.
- Threshold 0.583343; test coverage 99.44%, accepted accuracy 98.32%.
- Validation was 100% macro F1 for both trained candidates, so model choice used the lower validation loss as a pre-registered tie-break.
- The TFLite model is 3,763,884 bytes, float32 `[1,224,224,3]` to `[1,6]`.
- Keras/TFLite top-1 agreement: 100% on 180 test images.

See `model_exports/mobilenetv3small-sixclass-v1/MODEL_CARD.md` for limitations.

## Layout

```text
ml/
  configs/                Training configurations
  data/
    DATA_SOURCES.md       Dataset provenance, licence, audit, split evidence
    manifests/            Audit and split records
    raw/                  Downloaded archives and extracted originals (not tracked)
  reference/              Examined external candidate (not redistributable)
  reports/                Metrics, parity, benchmark, licence, handoff
  scripts/                Pipeline entry points
  tests/                  Reproducibility and bundle-contract tests
  checkpoints/            Training outputs (not tracked)
  requirements.txt        Direct pinned dependencies
  requirements-lock.txt   Full pip freeze used for the run
  .venv/                  Local Python 3.11 environment
```

## Environment

Recorded host and package details: `ml/reports/environment.md`.

Create the environment from the repository root:

```powershell
py -3.11 -m venv ml\.venv
ml\.venv\Scripts\python.exe -m pip install --upgrade pip
ml\.venv\Scripts\python.exe -m pip install -r ml\requirements-lock.txt
```

Verify CPU-only TensorFlow and the package versions:

```powershell
ml\.venv\Scripts\python.exe -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
```

Expected device list:

```text
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
```

Native Windows TensorFlow does not use the installed GTX 1650. The pipeline was trained on CPU.

## 1. Acquire the data

Download both archives from:

https://data.mendeley.com/datasets/bdd69gyhv8/1

No credentials are required. These direct links were read from the public Mendeley file API on 2026-10-30:

```powershell
New-Item -ItemType Directory -Force ml\data\raw\mendeley_bdd69gyhv8 | Out-Null

Invoke-WebRequest `
  -Uri 'https://data.mendeley.com/public-files/datasets/bdd69gyhv8/files/de93ba06-6a58-45e3-913d-837b2ae52acb/file_downloaded' `
  -OutFile 'ml\data\raw\mendeley_bdd69gyhv8\Original_Image.zip'

Invoke-WebRequest `
  -Uri 'https://data.mendeley.com/public-files/datasets/bdd69gyhv8/files/ccd1f142-03b2-473a-8c78-78920e63b8bd/file_downloaded' `
  -OutFile 'ml\data\raw\mendeley_bdd69gyhv8\Augmented_Image.zip'
```

Expected local layout:

```text
ml/data/raw/mendeley_bdd69gyhv8/Original_Image.zip
ml/data/raw/mendeley_bdd69gyhv8/Augmented_Image.zip
```

Verify the hashes before doing anything else:

```powershell
Get-FileHash -Algorithm SHA256 ml\data\raw\mendeley_bdd69gyhv8\Original_Image.zip
Get-FileHash -Algorithm SHA256 ml\data\raw\mendeley_bdd69gyhv8\Augmented_Image.zip
```

Expected:

```text
Original_Image.zip   f89d67d4c4b24810bcd8406db877db6b1ac9834cea0b0cb56d65acebe59e66ff
Augmented_Image.zip  5c748b5904932cd8cf631a0b776253d6217d577067d9477c267014d03ac16e10
```

Only the original archive is extracted and used. The augmented archive is retained for provenance but excluded because its filenames are corrupted and cannot be reliably mapped back to source images. See `ml/data/DATA_SOURCES.md`.

The dataset is CC BY 4.0. Attribution is required. The raw data is not redistributed by this project.

## 2. Audit and split

From the repository root:

```powershell
ml\.venv\Scripts\python.exe ml\scripts\audit_data.py
ml\.venv\Scripts\python.exe ml\scripts\prepare_splits.py --version v1 --seed 42
```

`audit_data.py` extracts six original classes, validates every image, computes SHA-256 and pHash groups, records the cross-label conflict, and writes:

- `ml/data/manifests/data_audit.json`
- `ml/data/manifests/excluded_conflict_groups.json`
- `ml/reports/sample_grid.png`

`prepare_splits.py` writes the group-aware train/validation/test manifest:

- `ml/data/manifests/splits_v1.csv`
- `ml/data/manifests/split_summary_v1.json`

Expected split counts: 837 train, 180 validation, 180 test. Validation and test have 30 images per class.

Do not create a replacement split after looking at test results. If a new split is genuinely required, use a new version and document the reason.

## 3. Train

Stage 1, frozen backbone:

```powershell
ml\.venv\Scripts\python.exe ml\scripts\train.py --config ml\configs\mv3s_frozen_224_v1.json
```

Stage 2, fine-tune the last 40 layers:

```powershell
ml\.venv\Scripts\python.exe ml\scripts\train.py --config ml\configs\mv3s_finetune_224_v1.json
```

The fine-tune config reads the frozen checkpoint through `initialize_from`. Both runs write to `ml/checkpoints/<run_id>/`:

- `best.keras`
- `last.keras`
- `history.csv`
- `training_summary.json`
- `config.json`

CPU training took 130.89 seconds for the frozen run and 44.67 seconds for fine-tuning. Resume support was not required. Rerunning a run ID overwrites its outputs; preserve the original reports before an intentional rerun.

Training uses seed 42 and deterministic op settings. CPU oneDNN may still produce small numerical differences across hardware or library versions.

## 4. Select the checkpoint

The selection decision is recorded in `ml/reports/model_selection.json`:

1. maximize validation macro F1;
2. break ties with lower validation loss;
3. if still tied, prefer the frozen run.

The fine-tuned `best.keras` was selected. Do not choose a model or threshold using the test split.

## 5. Export TFLite

```powershell
New-Item -ItemType Directory -Force ml\checkpoints\mv3s_finetune_224_v1\export | Out-Null

ml\.venv\Scripts\python.exe ml\scripts\export_tflite.py `
  --checkpoint ml\checkpoints\mv3s_finetune_224_v1\best.keras `
  --output ml\checkpoints\mv3s_finetune_224_v1\export\freshness.tflite `
  --metadata ml\checkpoints\mv3s_finetune_224_v1\export\export_metadata.json
```

Expected tensor contract:

```text
input:  [1, 224, 224, 3] float32
output: [1, 6] float32
```

Expected SHA-256:

```text
dd519afd78d27d731fc4cef0e9a02b96086e42122a51a768874b8b6fdfc9052e
```

The model uses built-in TFLite operators only. Do not add Flex ops for the first Android bundle.

## 6. Evaluate validation and select the threshold

```powershell
ml\.venv\Scripts\python.exe ml\scripts\evaluate.py `
  --model ml\checkpoints\mv3s_finetune_224_v1\export\freshness.tflite `
  --backend tflite `
  --split val `
  --run-id mv3s_finetune_224_v1 `
  --report ml\reports\eval_mv3s_finetune_224_v1_val.json `
  --predictions ml\reports\predictions_mv3s_finetune_224_v1_val.csv `
  --select-threshold `
  --target-coverage 0.95 `
  --threshold-output ml\reports\threshold_mv3s_finetune_224_v1_val.json
```

Threshold policy: among thresholds with coverage at least 95%, maximize accepted accuracy, then coverage, then threshold. Validation selected `0.5833432078361511`. Validation was saturated at 100%, so this threshold has limited resolution.

## 7. Evaluate test

```powershell
ml\.venv\Scripts\python.exe ml\scripts\evaluate.py `
  --model ml\checkpoints\mv3s_finetune_224_v1\export\freshness.tflite `
  --backend tflite `
  --split test `
  --run-id mv3s_finetune_224_v1 `
  --report ml\reports\eval_mv3s_finetune_224_v1_test.json `
  --predictions ml\reports\predictions_mv3s_finetune_224_v1_test.csv `
  --threshold-json ml\reports\threshold_mv3s_finetune_224_v1_val.json
```

This is an evaluation-only step. Do not change the model or threshold in response to test metrics without declaring that the test set was used for tuning.

## 8. Verify Keras/TFLite parity

```powershell
ml\.venv\Scripts\python.exe ml\scripts\verify_export.py `
  --keras ml\checkpoints\mv3s_finetune_224_v1\best.keras `
  --tflite ml\checkpoints\mv3s_finetune_224_v1\export\freshness.tflite `
  --split test `
  --report ml\reports\parity_mv3s_finetune_224_v1_test.json
```

Expected: top-1 agreement 100%, maximum absolute probability difference below `1e-4`.

## 9. Benchmark

```powershell
ml\.venv\Scripts\python.exe ml\scripts\benchmark.py `
  --model ml\checkpoints\mv3s_finetune_224_v1\export\freshness.tflite `
  --split test `
  --images 60 `
  --warmup 20 `
  --runs 300 `
  --threads 2 `
  --report ml\reports\benchmark_mv3s_finetune_224_v1.json
```

These are desktop CPU numbers. They are not Android latency measurements.

## 10. Evaluate the existing candidate

```powershell
ml\.venv\Scripts\python.exe ml\scripts\evaluate_existing.py `
  --model ml\reference\candidate_repo\ripeness_model.tflite `
  --labels ml\reference\candidate_repo\ripeness_labels.txt `
  --split test `
  --report ml\reports\eval_candidate_9class_test.json `
  --predictions ml\reports\predictions_candidate_9class_test.csv
```

The candidate is nine-class, licence-unresolved, and not part of the bundle. Details: `ml/reports/baseline.md`.

## 11. Build the Android bundle

```powershell
ml\.venv\Scripts\python.exe ml\scripts\finalize_bundle.py `
  --tflite ml\checkpoints\mv3s_finetune_224_v1\export\freshness.tflite
```

The script refuses to overwrite an existing bundle. Remove the old bundle only when intentionally regenerating it.

It writes:

```text
model_exports/mobilenetv3small-sixclass-v1/
  freshness.tflite
  model.json
  labels.json
  MODEL_CARD.md
  evaluation.json
  THIRD_PARTY_NOTICES.md
  SHA256SUMS.txt
  parity/
    README.md
    expected.json
    verify_parity.py
    inputs/*.png
```

The manifest sets:

- `modelId: mobilenetv3small-sixclass-v1`
- `minScore: 0.5833432078361511`
- `thresholdValidated: true`
- labels in canonical order
- RGB, stretch resize, mean/std 127.5

## 12. Verify the final bundle

Run the bundled parity verifier:

```powershell
ml\.venv\Scripts\python.exe model_exports\mobilenetv3small-sixclass-v1\parity\verify_parity.py
```

Run the ML tests:

```powershell
ml\.venv\Scripts\python.exe -m pytest ml\tests -q
```

The Android bundle validator can be invoked read-only through `android/tools/install_model.py`. Installing the model writes into `android/app/src/main/assets`; only the Android integration owner should run the install step.

## Android manifest semantics

The app must apply the manifest exactly:

- RGB input, NHWC.
- Stretch to 224 x 224.
- Normalize each channel as `(pixel - 127.5) / 127.5`.
- Output is six softmax probabilities in label order.
- Accept only when `thresholdValidated` is true and the maximum score is at least `minScore`.
- Show an uncertain/abstained state below threshold.

Do not modify `android/` from this ML workflow.

## Licence summary

- Dataset: CC BY 4.0; attribution to Sultana, Jahan, and Uddin (2022), DOI `10.17632/bdd69gyhv8.1`.
- Backbone: Keras Applications MobileNetV3Small ImageNet weights; Keras/TensorFlow Apache-2.0.
- Existing candidate: no LICENSE file; not redistributed.
- Project: no root LICENSE file; confirm release terms before external redistribution.

See `ml/data/DATA_SOURCES.md`, `ml/reports/license_audit.md`, and the bundle notices.

## Known limitations

- Test images are controlled dataset photos, not phone-camera images.
- No OOD/unknown class; the model always emits six scores.
- No Android latency or field validation.
- Dataset lacks capture-session IDs; pHash grouping reduces but does not eliminate leakage risk.
- Validation saturation limits the strength of model and threshold selection.
- The threshold does not guarantee food safety. An accepted result is only a prediction of visible condition under dataset-like conditions.
