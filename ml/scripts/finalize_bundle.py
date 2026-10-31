from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image, ImageOps

from common import (
    CHECKPOINT_ROOT,
    EXPORT_ROOT,
    LABELS,
    LABEL_META,
    ML_ROOT,
    PROJECT_ROOT,
    labels_payload,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)

PARITY_VERIFY_SCRIPT = '''from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
LABELS = json.loads((ROOT / "labels.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((ROOT / "parity" / "expected.json").read_text(encoding="utf-8"))
MODEL = ROOT / SPEC["modelFile"]
INTERPRETER = Interpreter(model_path=str(MODEL))
INTERPRETER.allocate_tensors()
INPUT = INTERPRETER.get_input_details()[0]
OUTPUT = INTERPRETER.get_output_details()[0]
MEAN = np.asarray(SPEC["input"]["mean"], dtype=np.float32)
STD = np.asarray(SPEC["input"]["std"], dtype=np.float32)
TOLERANCE = float(EXPECTED["absolute_tolerance"])

for fixture in EXPECTED["fixtures"]:
    path = ROOT / "parity" / "inputs" / fixture["file"]
    with Image.open(path) as image:
        image = image.convert("RGB")
        pixels = np.asarray(image, dtype=np.float32)
    normalized = ((pixels - MEAN) / STD)[None, ...]
    INTERPRETER.set_tensor(INPUT["index"], normalized)
    INTERPRETER.invoke()
    scores = INTERPRETER.get_tensor(OUTPUT["index"])[0]
    expected = np.asarray(fixture["probabilities"], dtype=np.float32)
    difference = float(np.max(np.abs(scores - expected)))
    if difference > TOLERANCE:
        raise SystemExit(f"FAIL {fixture['file']}: max difference {difference:.8f}")
    top1 = LABELS[int(scores.argmax())]["id"]
    if top1 != fixture["top1"]:
        raise SystemExit(f"FAIL {fixture['file']}: top1 {top1} != {fixture['top1']}")
    print(f"PASS {fixture['file']}: top1={top1}, max_diff={difference:.8f}")

print(f"All {len(EXPECTED['fixtures'])} parity fixtures passed.")
'''


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def load_rows(split: str, manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def load_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_parity(stage: Path, manifest_path: Path, spec: dict, threshold: float) -> dict:
    rows = load_rows("test", manifest_path)
    selected = {}
    for row in rows:
        selected.setdefault(row["label"], row)
    missing = set(LABELS) - set(selected)
    if missing:
        raise RuntimeError(f"parity split missing labels: {sorted(missing)}")

    fixtures = []
    input_dir = stage / "parity" / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    interpreter = Interpreter(model_path=str(stage / "freshness.tflite"))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    mean = np.asarray(spec["input"]["mean"], dtype=np.float32)
    std = np.asarray(spec["input"]["std"], dtype=np.float32)
    for label in LABELS:
        row = selected[label]
        with Image.open(PROJECT_ROOT / row["relative_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((spec["input"]["width"], spec["input"]["height"]), Image.Resampling.BILINEAR)
            output = input_dir / f"{label}.png"
            image.save(output)
            pixels = np.asarray(image, dtype=np.float32)
        normalized = ((pixels - mean) / std)[None, ...]
        interpreter.set_tensor(input_info["index"], normalized)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_info["index"])[0].astype(np.float32)
        top_index = int(scores.argmax())
        fixtures.append(
            {
                "file": output.name,
                "source_relative_path": row["relative_path"],
                "label": label,
                "sha256": sha256_file(output),
                "probabilities": [float(value) for value in scores],
                "top1": LABELS[top_index],
                "max_score": float(scores[top_index]),
                "accepted_at_bundle_threshold": bool(scores[top_index] >= threshold),
            }
        )
    payload = {
        "description": "Parity fixtures derived from valid test images. Inputs are RGB PNGs resized with stretch to the manifest dimensions.",
        "preprocessing": "Normalize each channel as (pixel - mean) / std, add batch dimension, float32 NHWC.",
        "model_file": "freshness.tflite",
        "model_sha256": sha256_file(stage / "freshness.tflite"),
        "labels": LABELS,
        "threshold": threshold,
        "absolute_tolerance": 0.0001,
        "fixtures": fixtures,
    }
    write_json(stage / "parity" / "expected.json", payload)
    (stage / "parity" / "verify_parity.py").write_text(PARITY_VERIFY_SCRIPT, encoding="utf-8")
    (stage / "parity" / "README.md").write_text(
        "\n".join(
            [
                "# Parity fixtures",
                "",
                "These six images are test-split images from the Mendeley dataset (CC BY 4.0, DOI 10.17632/bdd69gyhv8.1), resized to 224x224.",
                "They are included only to verify that `freshness.tflite`, the manifest preprocessing, and the declared label order produce the expected scores.",
                "",
                "Run from this directory:",
                "",
                "```text",
                "python parity/verify_parity.py",
                "```",
                "",
                "Requires Python 3.11, Pillow, NumPy, and ai-edge-litert. Expected scores were generated by the bundled TFLite file.",
                "Tolerance is 1e-4 absolute probability difference.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "fixture_count": len(fixtures),
        "absolute_tolerance": 0.0001,
        "expected_file": "parity/expected.json",
        "verify_script": "parity/verify_parity.py",
        "input_files": [fixture["file"] for fixture in fixtures],
    }


def build_notices(dataset: dict) -> str:
    return f"""# Third-party notices

## Fruit images

Dataset: “Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality”  
Contributors: Nusrat Sultana, Musfika Jahan, Mohammad Shorif Uddin  
Publisher: Mendeley Data, version 1, published 8 April 2022  
DOI: 10.17632/bdd69gyhv8.1  
License: Creative Commons Attribution 4.0 International (CC BY 4.0)  
URL: https://data.mendeley.com/datasets/bdd69gyhv8/1

The training data used here are the publisher's original images from six classes. The publisher's augmented archive was not used. Six resized test images are redistributed under CC BY 4.0 in `parity/inputs/` for tensor-parity verification. Attribution is preserved in this notice and in `MODEL_CARD.md`.

## Pretrained backbone

Architecture: MobileNetV3Small  
Implementation and ImageNet weights: Keras Applications / TensorFlow Keras  
Keras license: Apache License 2.0, https://github.com/keras-team/keras/blob/master/LICENSE  
MobileNetV3 reference: Andrew Howard et al., “Searching for MobileNetV3,” ICCV 2019, https://arxiv.org/abs/1905.02244  
Keras Applications weights documentation states that pretrained application weights are released under the Apache License: https://keras.io/api/applications/

The backbone weights are used as an initialization and are included only as transformed model parameters inside `freshness.tflite`. The ImageNet dataset itself is not redistributed.

## Runtime

TensorFlow, Keras, LiteRT, NumPy, Pillow, and scikit-learn are used for training, conversion, evaluation, or verification. Their respective licenses apply to those tools; no runtime binaries are included in this bundle.

## Project license

The repository does not contain a project-level LICENSE file. This bundle is a candidate artifact, not a legal opinion. Before external redistribution, preserve the CC BY 4.0 attribution above and confirm the applicable project and model-release terms.
"""


def build_model_card(
    model_id: str,
    created_at: str,
    artifact_sha: str,
    test: dict,
    val: dict,
    threshold_payload: dict,
    parity: dict,
    benchmark: dict,
    candidate: dict,
    audit: dict,
    split_summary: dict,
    predictions: list[dict],
    config: dict,
) -> str:
    threshold = float(threshold_payload["chosen"]["threshold"])
    error_rows = [row for row in predictions if row["label"] != row["predicted"]]
    false_fresh_rows = [
        row for row in predictions
        if row["label"].startswith("rotten_") and row["predicted"].startswith("fresh_")
    ]
    accepted_false_fresh = [row for row in false_fresh_rows if row["accepted"] == "True"]
    error_lines = [
        f"- `{row['label']}` → `{row['predicted']}`, score `{float(row['max_score']):.4f}`, "
        f"{'accepted' if row['accepted'] == 'True' else 'abstained'}: `{row['relative_path']}`"
        for row in error_rows
    ]
    test_per_class_lines = [
        f"| `{label}` | {entry['precision']:.3f} | {entry['recall']:.3f} | {entry['f1']:.3f} | {entry['support']} |"
        for label, entry in test["per_class"].items()
    ]
    train_count = sum(split_summary["counts"]["train"].values())
    val_count = sum(split_summary["counts"]["val"].values())
    test_count = sum(split_summary["counts"]["test"].values())
    correct_count = int(round(test["accuracy"] * test["count"]))
    accuracy_low, accuracy_high = wilson_interval(correct_count, test["count"])
    fresh_count = sum(test["label_counts"][label] for label in LABELS if label.startswith("fresh_"))
    false_fresh_low, false_fresh_high = wilson_interval(test["false_fresh_count"], fresh_count)
    return f"""# Model card: {model_id}

## Summary

This is a six-class image classifier for visible condition of apples, bananas, and oranges. It classifies a supplied fruit image as fresh or visibly spoiled for each fruit. It does not detect the fruit or estimate a safe-to-eat date.

- Model ID: `{model_id}`
- Bundle created: {created_at}
- Artifact: `freshness.tflite`, SHA-256 `{artifact_sha}`
- Architecture: MobileNetV3Small with a six-class softmax head
- Parameters: 942,582 total; 3,763,884-byte float32 TFLite file
- Input: RGB, NHWC `[1, 224, 224, 3]`, stretch resize, `(pixel - 127.5) / 127.5`
- Output: `[1, 6]` float32 softmax probabilities
- Labels: `{', '.join(LABELS)}`
- Decision threshold: `{threshold:.6f}` (`thresholdValidated: true`)

## Intended use

Intended for an offline Android demonstration that checks one visible apple, banana, or orange at a time. It is suitable for comparing ML pipeline quality and for educational use.

Not suitable for food-safety decisions, medical decisions, shelf-life prediction, pesticide or contamination detection, unseen fruit types, or images where the fruit is heavily occluded, cut, cooked, or poorly lit. “Rotten” in this project means a visible-spoilage category in the source dataset, not a laboratory determination that a fruit is unsafe.

## Training data

- Source: Mendeley Data, DOI `10.17632/bdd69gyhv8.1`, CC BY 4.0.
- Contributors: Nusrat Sultana, Musfika Jahan, Mohammad Shorif Uddin.
- The source contains 3,200 original and 12,335 publisher-augmented images across 16 classes.
- This model used only the six relevant original classes: FreshApple, FreshBanana, FreshOrange, RottenApple, RottenBanana, RottenOrange.
- 1,200 originals were inspected. One 3-image cross-label near-duplicate group was excluded. Final dataset: {split_summary['total_images']} images.
- Split: {train_count} train, {val_count} validation, {test_count} test. Counts by class are recorded in `evaluation.json`.
- The publisher's augmented archive was not used because its filenames are corrupted and individual augmented files cannot be mapped back to source capture groups. Using them could leak validation or test content into training.

Splits were made after grouping 64-bit perceptual-hash near-duplicates. The manifest is `ml/data/manifests/splits_v1.csv` with SHA-256 `{split_summary['split_manifest_sha256']}`. The source dataset does not provide physical-fruit or capture-session identifiers, so residual leakage between visually similar but perceptually distinct images cannot be excluded.

## Training procedure

- Backbone initialized with Keras Applications MobileNetV3Small ImageNet weights.
- Stage 1: frozen backbone, global average pooling, dropout 0.2, six-class softmax head; Adam, learning rate 0.001.
- Stage 2: last 40 backbone layers unfrozen with batch-normalization layers frozen; Adam, learning rate 0.0001.
- Selected run: `{config['run_id']}`. The selected checkpoint was chosen on the validation split by macro F1 and then validation loss; the test split was not used for selection.
- Training augmentation: random crop/zoom, horizontal flip, brightness, contrast, and saturation. No test-time augmentation.
- CPU-only TensorFlow {val['model_details'].get('tensorflow', '2.21.0') if 'model_details' in val else '2.21.0'} on Windows. Deterministic op setting and seed {config.get('seed', 42)} were used.

## Evaluation

Evaluation used the actual TFLite artifact on {test['count']} untouched test images. The test split contains 30 images per class.

| Metric | Value |
|---|---:|
| Six-class accuracy | {percent(test['accuracy'])} |
| Six-class macro F1 | {percent(test['macro_f1'])} |
| Condition accuracy | {percent(test['condition_accuracy'])} |
| False-fresh errors (all) | {test['false_fresh_count']} |
| False-fresh errors (accepted) | {len(accepted_false_fresh)} |
| False-rotten errors (all) | {test['false_rotten_count']} |
| Coverage at threshold | {percent(test['coverage'])} |
| Accepted accuracy at threshold | {percent(test['accepted_accuracy'])} |
| Expected calibration error, 10 bins | {test['expected_calibration_error_10bin']:.4f} |

Per-class test metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{chr(10).join(test_per_class_lines)}

Confusion matrix rows are actual classes and columns are predicted classes in the label order above:

```json
{json.dumps(test['confusion_matrix'])}
```

There were {len(error_rows)} test errors. All were same-fruit condition confusions; none were wrong-fruit predictions. The threshold abstained on the one false-fresh test error, but this small test set does not establish an operational safety guarantee.

### Threshold selection

The validation split was used before test evaluation. The policy was: among thresholds with coverage at least 95%, maximize accepted accuracy, then coverage, then threshold. Validation had {val['count']} images and the selected threshold `{threshold:.6f}` achieved `{threshold_payload['chosen']['coverage'] * 100:.2f}%` coverage with `{percent(threshold_payload['chosen']['accepted_accuracy'])}` accepted accuracy. Validation macro F1 was {percent(val['macro_f1'])}. The threshold is a closed-set confidence cutoff, not an out-of-distribution detector.

### Error examples

{chr(10).join(error_lines) if error_lines else '- None.'}

## Robustness and limitations

- Test images come from controlled dataset conditions and are not phone photos. No phone, field, lighting, blur, occlusion, background, or multiple-fruit evaluation was performed.
- No Android latency measurement was made. Desktop-only benchmark: median TFLite invoke {benchmark['inference_only']['p50_ms']:.3f} ms, p95 {benchmark['inference_only']['p95_ms']:.3f} ms with two threads. Desktop preprocessing plus inference was much slower and is not an Android estimate.
- The test split is small ({test['count']} images). The approximate 95% Wilson interval for accuracy is {percent(accuracy_low)} to {percent(accuracy_high)}; for the false-fresh rate it is {percent(false_fresh_low)} to {percent(false_fresh_high)}. These intervals describe this controlled test set and are not deployment guarantees.
- The dataset does not provide capture-session IDs. Perceptual hashing reduces obvious leakage but does not prove physical-fruit separation.
- The validation set saturated at 100% macro F1, so model selection and threshold selection had limited resolution.
- The app must supply a correctly oriented, reasonably centered single-fruit image. The model has no “unknown” class and will force one of the six labels unless the app applies the confidence threshold.
- Softmax scores are model scores. The observed ECE is {test['expected_calibration_error_10bin']:.4f}, but calibration was not separately tuned or validated on external data.

## Comparison to the examined existing model

The examined repository model is a nine-class fresh/rotten/unripe TFLite classifier with unresolved licensing. Mapped onto this six-class test set, it produced {percent(candidate['mapped_label_accuracy'])} accuracy, {percent(candidate['mapped_macro_f1'])} macro F1, and {candidate['unripe_predictions']} unripe outputs across {candidate['count']} images. It also produced {candidate['false_fresh_count']} false-fresh and {candidate['false_rotten_count']} false-rotten condition errors. It was not used in this bundle.

## Parity and bundle integrity

- Keras-to-TFLite parity on {parity['count']} test images: top-1 agreement {percent(parity['top1_agreement'])}, maximum absolute probability difference {parity['max_absolute_probability_difference']:.8e}.
- `parity/` contains six valid test images and expected TFLite probabilities. Run `python parity/verify_parity.py` with the test dependencies.
- `SHA256SUMS.txt` covers the bundle files.

## Licensing and attribution

- Fruit images: CC BY 4.0; attribution to Sultana, Jahan, and Uddin (2022), DOI `10.17632/bdd69gyhv8.1`. See `THIRD_PARTY_NOTICES.md`.
- Backbone: Keras Applications MobileNetV3Small ImageNet weights; Keras is Apache-2.0 and its applications documentation states the pretrained weights are Apache-licensed. The ImageNet dataset is not redistributed.
- The repository has no project-level LICENSE file. Confirm project terms before external redistribution.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="mobilenetv3small-sixclass-v1")
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--checkpoint-summary", default="ml/checkpoints/mv3s_finetune_224_v1/training_summary.json")
    parser.add_argument("--test-report", default="ml/reports/eval_mv3s_finetune_224_v1_test.json")
    parser.add_argument("--val-report", default="ml/reports/eval_mv3s_finetune_224_v1_val.json")
    parser.add_argument("--threshold", default="ml/reports/threshold_mv3s_finetune_224_v1_val.json")
    parser.add_argument("--parity-report", default="ml/reports/parity_mv3s_finetune_224_v1_test.json")
    parser.add_argument("--predictions", default="ml/reports/predictions_mv3s_finetune_224_v1_test.csv")
    parser.add_argument("--benchmark", default="ml/reports/benchmark_mv3s_finetune_224_v1.json")
    parser.add_argument("--candidate-report", default="ml/reports/eval_candidate_9class_test.json")
    parser.add_argument("--audit", default="ml/data/manifests/data_audit.json")
    parser.add_argument("--split-summary", default="ml/data/manifests/split_summary_v1.json")
    parser.add_argument("--manifest", default="ml/data/manifests/splits_v1.csv")
    args = parser.parse_args()

    tflite_path = resolve(args.tflite)
    manifest_path = resolve(args.manifest)
    test = read_json(resolve(args.test_report))
    val = read_json(resolve(args.val_report))
    threshold_payload = read_json(resolve(args.threshold))
    parity_report = read_json(resolve(args.parity_report))
    benchmark = read_json(resolve(args.benchmark))
    candidate = read_json(resolve(args.candidate_report))
    audit = read_json(resolve(args.audit))
    split_summary = read_json(resolve(args.split_summary))
    config = read_json(resolve(args.checkpoint_summary))["config"]
    predictions = load_predictions(resolve(args.predictions))
    threshold = float(threshold_payload["chosen"]["threshold"])

    final_dir = EXPORT_ROOT / args.model_id
    if final_dir.exists():
        raise FileExistsError(f"{final_dir} already exists; refusing to overwrite a candidate bundle")
    stage = Path(tempfile.mkdtemp(prefix="bundle-stage-", dir=CHECKPOINT_ROOT))
    try:
        shutil.copyfile(tflite_path, stage / "freshness.tflite")
        artifact_sha = sha256_file(stage / "freshness.tflite")
        labels = labels_payload()
        (stage / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
        spec = {
            "schemaVersion": 1,
            "modelId": args.model_id,
            "modelVersion": "1.0.0",
            "modelFile": "freshness.tflite",
            "sha256": artifact_sha,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "trainingRun": config["run_id"],
            "input": {
                "width": 224,
                "height": 224,
                "layout": "NHWC",
                "channels": 3,
                "colorSpace": "RGB",
                "resizeMode": "stretch",
                "interpolation": "bilinear",
                "mean": [127.5, 127.5, 127.5],
                "std": [127.5, 127.5, 127.5],
            },
            "output": {
                "scoreType": "probabilities",
                "tensorShape": [1, 6],
            },
            "labels": labels,
            "decision": {
                "minScore": threshold,
                "thresholdValidated": True,
                "validationSplit": "val",
                "validationImages": val["count"],
                "validationCoverage": threshold_payload["chosen"]["coverage"],
                "validationAcceptedAccuracy": threshold_payload["chosen"]["accepted_accuracy"],
                "note": "Closed-set confidence cutoff selected on the validation split; it is not an OOD guarantee.",
            },
            "redistribution": {
                "status": "unresolved",
                "notes": "Dataset is CC BY 4.0 and the backbone is Apache-2.0, but the repository has no project-level LICENSE. Confirm project release terms before external redistribution.",
            },
        }
        write_json(stage / "model.json", spec)

        created_at = spec["createdAt"]
        parity = build_parity(stage, manifest_path, spec, threshold)
        parity["count"] = parity_report["count"]
        parity["top1_agreement"] = parity_report["top1_agreement"]
        parity["max_absolute_probability_difference"] = parity_report["max_absolute_probability_difference"]

        evaluation = {
            "schemaVersion": 1,
            "modelId": args.model_id,
            "artifact": {
                "file": "freshness.tflite",
                "bytes": (stage / "freshness.tflite").stat().st_size,
                "sha256": artifact_sha,
                "format": "LiteRT / TFLite float32, TFLITE_BUILTINS only",
                "input": spec["input"],
                "output": spec["output"],
            },
            "dataset": {
                "source": "Mendeley Data",
                "doi": "10.17632/bdd69gyhv8.1",
                "license": "CC BY 4.0",
                "contributors": ["Nusrat Sultana", "Musfika Jahan", "Mohammad Shorif Uddin"],
                "selected_source_classes": audit["selected_source_classes"],
                "original_images_audited": audit["image_count"],
                "excluded_images": split_summary["excluded_images"],
                "excluded_reason": "One near-duplicate group contained both fresh_orange and rotten_orange labels.",
                "augmented_archive_used": False,
                "split_manifest_sha256": split_summary["split_manifest_sha256"],
                "split_counts": split_summary["counts"],
            },
            "training": {
                "run_id": config["run_id"],
                "architecture": "MobileNetV3Small",
                "pretrained_weights": "Keras Applications ImageNet",
                "input_size": 224,
                "classes": LABELS,
                "selection_rule": "Validation macro F1, then validation loss; test was not used for selection.",
                "test_used_for_selection": False,
                "estimated_parameters": 942582,
            },
            "validation": {
                "count": val["count"],
                "accuracy": val["accuracy"],
                "macro_f1": val["macro_f1"],
                "condition_accuracy": val["condition_accuracy"],
                "expected_calibration_error_10bin": val["expected_calibration_error_10bin"],
                "threshold_selection": {
                    "target_coverage": threshold_payload["target_coverage"],
                    "rule": threshold_payload["selection_rule"],
                    "chosen": threshold_payload["chosen"],
                },
            },
            "test": test,
            "parity": parity,
            "desktop_benchmark": benchmark,
            "existing_candidate_comparison": candidate,
            "redistribution": spec["redistribution"],
            "not_evaluated": [
                "Android device latency and memory",
                "Phone-camera or field images",
                "Out-of-distribution and unknown-class rejection",
                "Food safety or edibility",
            ],
        }
        write_json(stage / "evaluation.json", evaluation)
        (stage / "MODEL_CARD.md").write_text(
            build_model_card(
                args.model_id,
                created_at,
                artifact_sha,
                test,
                val,
                threshold_payload,
                parity,
                benchmark,
                candidate,
                {"image_count": audit["image_count"]},
                split_summary,
                predictions,
                config,
            ),
            encoding="utf-8",
        )
        (stage / "THIRD_PARTY_NOTICES.md").write_text(build_notices(evaluation["dataset"]), encoding="utf-8")

        files = sorted(path for path in stage.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
        lines = [f"{sha256_file(path)}  {path.relative_to(stage).as_posix()}" for path in files]
        (stage / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stage), str(final_dir))
        print(json.dumps(
            {
                "bundle": final_dir.relative_to(PROJECT_ROOT).as_posix(),
                "model_sha256": artifact_sha,
                "model_bytes": (final_dir / "freshness.tflite").stat().st_size,
                "threshold": threshold,
                "files": len(files) + 1,
            },
            indent=2,
        ))
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
