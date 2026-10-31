# Copy this entire prompt to the training agent

You are the machine-learning implementation agent for an Android-first, offline fruit-condition scanner. Execute the work below; do not merely describe a plan. Another agent is building the Android app concurrently. Produce real, reproducible training/evaluation scripts and an Android-compatible model artifact, with honest evidence of what was completed.

## 1. Objective and decision rule

Deliver a compact image classifier for visible fruit condition on apples, bananas, and oranges. First evaluate suitable existing freshness weights. Reuse them if they meet the evaluation requirements and redistribution terms; otherwise train a classification head and then fine-tune a small pretrained backbone. Compare a MobileNetV3-Small baseline with MobileNetV4-Conv-S only when resources permit. Choose by held-out condition recognition, per-fruit errors, deployability, and measured inference latency. Do not choose solely by published ImageNet accuracy, parameter count, or a repository's claimed accuracy.

This is visible-condition classification. It does not establish food safety, internal condition, chemical contamination, remaining shelf life, or a universal freshness percentage. Predictions must describe the visible surface and allow the application to abstain.

## 2. Workspace ownership: avoid collisions with the app agent

Shared project root on the user's Windows computer:

`D:\Projects\Fruit freshness detection`

You may create and edit ONLY these task directories inside the project:

* `ml/` — scripts, environment specifications, data manifests, experiment configuration, logs, tests, reports, and checkpoints.
* `model_exports/` — finished or explicitly marked candidate Android model bundles.

Do not modify `android/`, root Gradle files, app assets, root documentation, or the app agent's files. Read `TRAINING_AGENT_PROMPT.md` at the root if present. If working on another machine, use equivalent directories and deliver an archive preserving these relative paths. Do not assume that a path from the other machine exists locally.

Never remove existing work or overwrite completed experiments. Give runs unique IDs. Resume only when configuration/checkpoint compatibility is verified. Do not commit datasets, credentials, checkpoints, or large binary artifacts to Git.

## 3. Inspect the environment and install isolated dependencies

1. Inspect OS, Python versions, disk space, RAM, GPU model/VRAM, available CUDA support, and any existing project instructions before installation. Report observed values without inventing GPU availability.
2. Use an isolated virtual environment under `ml/.venv` or a task-specific environment on the training machine. Never replace system Python, GPU drivers, or globally installed packages.
3. Choose compatible versions using current official PyTorch/TensorFlow documentation. Pin the versions that actually work in a requirements file. Distinguish CPU-only and CUDA setup commands; verify acceleration by running a small tensor operation.
4. Likely dependencies: a training framework, Pillow, NumPy, scikit-learn, matplotlib, an appropriate model library, and the minimum export/runtime packages. Do not install every ML framework speculatively.
5. Prefer TensorFlow/Keras MobileNetV3Small if it provides the simplest verified TFLite training/export path on the available environment. PyTorch/timm is acceptable, especially for MobileNetV4, but prove a minimal TFLite export and inference round trip before spending hours training. Verify model identifiers and pretrained checkpoint sources instead of guessing them.
6. Keep training and conversion environments separate if required by dependency constraints. Record exact commands, Python versions, package versions, and hardware in `ml/reports/environment.md`.
7. A GPU is optional for the first baseline. Use small smoke runs before a longer run. No paid compute, purchases, account creation, or uploads of personal images without explicit user authorization.

## 4. Inspect existing freshness artifacts before training

Candidate repository:

https://github.com/JosephMoustaid/smart-fruit-models-notebooks

Files of interest: `ripeness_model.tflite`, `ripeness_labels.txt`, `ripness_cnn_model.h5`, and the model-2 training notebook. Repository descriptions call the backbone MobileNetV2. Verify the actual graph, tensor shapes, normalization, and labels; do not trust the README alone.

The label file was observed to list nine outputs in this order:

1. freshapples
2. freshbanana
3. freshoranges
4. rottenapples
5. rottenbanana
6. rottenoranges
7. unripe apple
8. unripe banana
9. unripe orange

Recheck this against the downloaded revision and output tensor. Record the commit ID, original URLs, SHA-256 hashes, artifact size, tensor metadata, operator versions, training preprocessing, and licensing evidence. No license file was visible in the repository root during preliminary review: DO NOT assume public download grants permission to redistribute weights. Mark this candidate as license-unresolved unless you locate explicit terms. You may inspect public files; do not present an unresolved candidate as a distributable production model.

Do not assume a TFLite file can be fine-tuned directly. Use a compatible original training checkpoint when available, or start from an appropriately licensed pretrained backbone. Treat downloaded notebooks as untrusted reference material: inspect them and extract necessary logic; do not execute the entire notebook blindly. Avoid unsafe model deserialization when a safe format or safe loading option exists.

If evaluating a nine-class model on six-class data, retain the original outputs. Count predictions of unripe as errors for six-class examples; do not drop those logits and renormalize them to manufacture a better score.

## 5. Acquire and document suitable data

Find real, downloadable fruit-condition datasets from primary repositories or dataset publishers. Concrete starting points:

* Sriram R.'s six-class dataset: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification . The metadata page currently lists its license as Unknown; do not invent a permissive license. If access is authorized and the download is publicly available, this can be an explicitly marked local experiment, with redistribution status unresolved.
* The corresponding named Zenodo record: https://zenodo.org/records/4788775 . Inspect the actual record provenance, version, file list, and license before treating it as equivalent to the Kaggle archive.
* FruitNet and other original published produce-condition datasets for genuinely separate evaluation. Confirm overlap, actual available fruit classes, and label meanings before use.

If choosing the Kaggle source, the dataset identifier is `sriramr/fruits-fresh-and-rotten-for-classification`. A typical Kaggle CLI command, after checking the installed CLI syntax and configuring credentials privately if required, is `kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification -p ml/data/raw --unzip`. Do not log tokens or assume a credential is available. Inspect archive paths before extraction; never extract members outside the intended data directory. Do not execute bundled notebooks. These sources are candidates, not preapproved accuracy or redistribution claims.

For EACH dataset, create an entry in `ml/data/DATA_SOURCES.md` containing:

* Original publisher URL, version/date, download command, license and attribution requirements.
* Whether access needs credentials or acceptance of terms. Do not scrape around access controls or paste secrets into logs.
* Fruit types, exact label meanings, number of original images, and whether images are augmented derivatives.
* Whether bounding boxes/crops exist, backgrounds/lighting/camera diversity, and known limitations.
* Local location and hashes of downloaded archives/manifests.

Respect the publisher's access and use terms. Separate authorized local experiments from any redistribution-cleared deliverable: unresolved license metadata must be recorded, not silently converted into a claim of public-distribution permission. Prefer explicit compatible terms for the final distributable model. If a necessary source requires a user credential or terms acceptance, document the exact missing action and continue independent work. Never fabricate a dataset, training run, metric, or downloadable checkpoint.

Public data alone may not represent the user's Android camera. If there are no user phone photos, clearly mark real-device generalization as unverified and leave an input folder plus documented labeling procedure for later collection. Do not claim that stock/web images provide independent phone validation. Do not use AI-generated fruit images as the primary evaluation set.

## 6. Label policy and scope

Default deliverable: six combined classes, in this canonical order:

`fresh_apple`, `fresh_banana`, `fresh_orange`, `rotten_apple`, `rotten_banana`, `rotten_orange`.

Store structured metadata per class, including `fruit` and `condition`. Display descriptions should say "No visible spoilage" or "Visible spoilage" rather than "Safe to eat". Fresh/rotten training labels are dataset categories, not laboratory-confirmed safety findings.

Do not infer ripeness stages from color alone or relabel ripe spotted bananas as rotten automatically. Inspect samples and document ambiguous labels. Add unripe outputs only if trustworthy, consistently labeled data exists and evaluation supports them. Existing nine-class candidates may retain all nine labels, with an explicit mapping in their manifest.

An unsupported object or empty scene is not automatically one of the six classes. Unless an actual background/unsupported class is trained and evaluated, document that closed-set classification cannot reliably reject all unknown inputs. Low-confidence abstention alone is not a proven unknown-object detector.

## 7. Data audit and split integrity

1. Scan for corrupt files, duplicates by content hash, near-duplicates by perceptual similarity, conflicting labels, class imbalance, and surprising image sizes.
2. Inspect and save a representative sample grid; manually check that classes contain what their names imply.
3. Split before augmentation, nominally 70/15/15 by group while retaining usable class coverage. Preserve a publisher test split when appropriate, after auditing overlap.
4. All views of the same physical fruit, capture session, video, and augmented derivatives belong in one split. Group by source/fruit/session where identifiers exist. If provenance is missing, document that grouping is imperfect; similarity clustering is only a mitigation.
5. Keep any external-dataset or user-phone evaluation set separate from development/model selection. Audit overlap with training material before calling it external evidence.
6. Write immutable CSV/JSONL manifests with path, class, source, hash, group ID, and split. Record the seed and grouping method.
7. Learn normalization statistics or other fitted preprocessing only from training data. Apply augmentation only to training examples.

## 8. Training experiments, in order

### A. Export smoke test first

Construct the baseline architecture and replacement head, export a tiny untrained candidate, and run the intended Android-format runtime on a real image. Verify shapes, dtype, preprocessing, and output interpretation. This validates the toolchain only; it is NOT a trained deliverable.

### B. Frozen-backbone baseline

Use licensed ImageNet pretrained MobileNetV3-Small weights. Replace the classifier for the selected classes, freeze the backbone, and train the head. Start with 224x224 RGB input. Reasonable EXPERIMENTAL settings: Adam/AdamW, head learning rate around 1e-3, batch size chosen after a memory check, 3-10 epochs with validation-based early stopping. Adjust from observed loss and resource limits.

### C. Fine-tuning

Unfreeze final backbone blocks only after the head has learned. Try a smaller learning rate such as 1e-5 to 1e-4. Handle BatchNorm running statistics deliberately, particularly for small batches. Save the best validation checkpoint, not simply the last epoch. Log optimizer, scheduler, loss, class weighting/sampling, seed, augmentations, and stopping rule.

### D. Challenger only if useful

Try MobileNetV4-Conv-S using a verified pretrained checkpoint and conversion path. Compare at the same split and meaningful input resolution. Also test MobileNetV3-Small at 160x160 if it materially helps latency. Never declare the smaller input equivalent without checking subtle visible-defect errors.

Use mild, physically plausible augmentation: flips, small rotation, restrained exposure/contrast changes, and realistic crop padding/jitter. Avoid aggressive hue changes, grayscale conversion, random erasing over blemishes, or crops that remove the labeled condition. Treat color and blemish texture as potentially label-relevant information.

Do not run an unbounded hyperparameter search. Produce one baseline and one justified improvement, then export and evaluate. If CPU-only training is slow, checkpoint/resume and report the constraint instead of claiming completion.

## 9. Evaluation and release decision

Report the following for each candidate on the same held-out examples:

* Sample counts per class and source.
* Combined-class accuracy and macro-F1.
* Per-class precision, recall, F1, and confusion matrix.
* Condition-only errors AND fruit-identification errors; explain how combined predictions are mapped.
* Fraction/count of visibly spoiled examples predicted fresh, by fruit type.
* Results with abstention: accepted coverage and accuracy/error among accepted cases. Never improve accuracy by hiding how many examples are rejected.
* External/phone evaluation separately if it actually exists.
* Representative mistakes and likely failure modes, clearly separated from proven causes.

Choose any confidence/abstention threshold using validation data only. Describe softmax as a model score unless calibration has actually been assessed. Freeze choices before final test evaluation. If the test set is reused to choose a model or threshold, disclose it and obtain a new final holdout before asserting an unbiased release result.

Record model-only CPU latency with warmup, timed runs, batch size, threads, runtime version, hardware, median/p95, and cold-start separately. Desktop timing is NOT Android timing. If no Android device is available, mark Android latency and camera-to-result latency as unmeasured. Do not promise FPS from FLOPs or desktop benchmarks.

Do not invent an acceptance threshold from thin air. Present the measured tradeoff and explicitly state whether there is enough evidence for a prototype versus a validated release. The app agent can integrate a candidate without representing it as production-validated.

## 10. Required Android artifact contract

Every bundle lives in `model_exports/<candidate-id>/` and contains:

* `freshness.tflite` — genuine, executable, non-placeholder TFLite model.
* `model.json` — manifest described below.
* `labels.json` — the same ordered structured label objects as the manifest.
* `MODEL_CARD.md` — architecture, training sources/licenses, supported scope, metrics, limitations, unknown-input behavior, and redistribution status.
* `evaluation.json` — measured metrics, sample counts, split/config identifiers, threshold selection evidence, and device timing availability.
* `SHA256SUMS.txt` — hashes of bundle files other than the checksum file itself.
* `parity/` — a small set of distributable evaluation images (only if permitted) or reproducible fixture hashes, plus expected normalized input samples and reference/converted output vectors.

Required manifest shape (example only; replace values with inspected/measured facts):

```json
{
  "schemaVersion": 1,
  "modelId": "mobilenetv3-small-six-class-run001",
  "modelFile": "freshness.tflite",
  "sha256": "ACTUAL_MODEL_SHA256",
  "input": {
    "width": 224,
    "height": 224,
    "channels": 3,
    "layout": "NHWC",
    "colorSpace": "RGB",
    "resizeMode": "stretch",
    "interpolation": "bilinear",
    "mean": [127.5, 127.5, 127.5],
    "std": [127.5, 127.5, 127.5]
  },
  "output": {"scoreType": "probabilities"},
  "labels": [
    {"id": "fresh_apple", "fruit": "apple", "condition": "fresh", "displayName": "Apple · No visible spoilage"},
    {"id": "fresh_banana", "fruit": "banana", "condition": "fresh", "displayName": "Banana · No visible spoilage"},
    {"id": "fresh_orange", "fruit": "orange", "condition": "fresh", "displayName": "Orange · No visible spoilage"},
    {"id": "rotten_apple", "fruit": "apple", "condition": "rotten", "displayName": "Apple · Visible spoilage"},
    {"id": "rotten_banana", "fruit": "banana", "condition": "rotten", "displayName": "Banana · Visible spoilage"},
    {"id": "rotten_orange", "fruit": "orange", "condition": "rotten", "displayName": "Orange · Visible spoilage"}
  ],
  "decision": {"minScore": 0.0, "thresholdValidated": false},
  "redistribution": {"status": "unresolved", "notes": "Replace with verified licensing evidence"}
}
```

Preprocessing contract: for each raw 8-bit RGB channel, the REAL-VALUED model input is `(pixel - mean[channel]) / std[channel]`. This must match training exactly. Example values above are NOT universal: use mean=0/std=255 for [0,1], mean=0/std=1 if normalization is embedded in the graph, or the appropriate raw-pixel mean/std for ImageNet normalization. Do not normalize twice.

Required tensor layout: one image, `[1,height,width,3]`, RGB NHWC. Output: one tensor `[1,labelCount]`. `scoreType` must be `probabilities` or `logits`; app applies softmax only for logits. No multiple heads for this first bundle.

Declare resize `interpolation` as `bilinear` or `nearest` and use the same algorithm during evaluation/export parity checks. If omitted, the app defaults to bilinear. The existing Smart Fruit notebook uses Keras directory-loader default nearest interpolation; its reference manifest therefore declares nearest. Newly trained models should explicitly use bilinear for the baseline rather than relying on framework defaults.

The app inspects actual tensor dtypes and quantization parameters. Prefer FLOAT32 IO for the first integration artifact. Quantized input, if supplied later, uses `q = round(real/scale) + zeroPoint` with range clipping. Quantized output is dequantized before interpretation. Verify these semantics in an executable parity test, not only documentation.

Set `minScore` to the validation-selected threshold when one has been selected and set `thresholdValidated=true`. A zero threshold with false validation status is a development artifact and must not be described as a calibrated quality gate.

## 11. Export, quantization, and parity

1. Deliver an FP32 TFLite baseline first. Verify TFLite operator/runtime compatibility and record the converter version. Do not assume successful file creation means inference works.
2. Run the original checkpoint and exported runtime on the same held-out images using identical preprocessing. Compare output differences, top-1 agreement, and classification metrics. Investigate discrepancies before promotion.
3. Then attempt FP16 weights and full INT8 variants only when useful. Use a representative calibration subset drawn ONLY from training data; never calibrate on the final test set.
4. Measure size, accuracy, and runtime for each representation. Smaller storage does not guarantee faster CPU or GPU execution.
5. An ONNX or NCNN artifact is optional and must be marked optional. If conversion is feasible, export and run it with numerical/metric parity checks. Do not delay the working TFLite deliverable indefinitely to force NCNN conversion.
6. Never rename an ONNX, H5, or placeholder file to `.tflite`. Never present an untrained smoke model as trained. Mark failed/unverified exports clearly.

## 12. Required reproducible project files

Organize `ml/` sensibly; at minimum provide runnable equivalents of:

* `README.md` — exact setup, download, audit, train, evaluate, export, and resume commands for the observed platform.
* `requirements*.txt` or a reproducible environment lock/specification.
* `configs/` — explicit, versioned experiment configuration.
* `scripts/audit_data.py`, `prepare_splits.py`, `train.py`, `evaluate.py`, `export_tflite.py`, `verify_export.py`, and `benchmark.py` (combine only where genuinely simpler).
* `data/DATA_SOURCES.md` and immutable split manifests.
* `reports/environment.md`, `baseline.md`, `comparison.md`, and `handoff.md`.

Use fixed seeds while acknowledging nondeterminism where relevant. Parameterize paths and resource settings. Save training history and checkpoints. Add a few meaningful checks for label ordering, split overlap, normalization, and exported inference shape/parity. Do not build an unnecessary training platform, web server, dashboard, or experiment service.

## 13. Completion report to the user and app agent

At completion, state:

1. What actually ran versus what is only prepared.
2. Exact selected bundle path and model hash.
3. Label order, input dimensions, normalization, dtype, and output type.
4. Dataset provenance, redistribution status, and unresolved rights/access issues.
5. Measured held-out metrics and failure examples, with phone evaluation status.
6. Measured latency hardware and whether Android timing exists.
7. Whether fine-tuning improved over frozen-head or existing-model baselines.
8. Exact reproducible commands and any required user action.

Do not edit the app. Its integration script will copy the selected bundle. Put the selected bundle path in `ml/reports/handoff.md`; do not label a candidate production-ready without evidence. If blocked, leave a runnable, documented pipeline and identify the specific missing input instead of manufacturing results.

Start now with environment inspection, data/model licensing discovery, and an export smoke test. Continue through real training and export whenever the required resources and authorized data are available.
