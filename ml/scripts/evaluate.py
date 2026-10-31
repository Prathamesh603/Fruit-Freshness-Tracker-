from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image, ImageOps
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from tensorflow import keras

from common import LABELS, MANIFEST_ROOT, PROJECT_ROOT, read_json, sha256_file, write_json

CONDITIONS = ["fresh", "rotten"]
CONDITION_OF_LABEL = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
FRUIT_OF_LABEL = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int32)
FRUIT_NAMES = ("apple", "banana", "orange")
FRUIT_LABEL_INDICES = {"apple": (0, 3), "banana": (1, 4), "orange": (2, 5)}


def load_rows(split: str, manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def load_images(rows: list[dict], size: int) -> np.ndarray:
    images = []
    for row in rows:
        path = PROJECT_ROOT / row["relative_path"]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((size, size), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images)


def predict_keras(model_path: Path, images: np.ndarray) -> np.ndarray:
    model = keras.models.load_model(model_path)
    inputs = (images.astype(np.float32) - 127.5) / 127.5
    return model.predict(inputs, batch_size=32, verbose=0)


def predict_tflite(model_path: Path, images: np.ndarray) -> tuple[np.ndarray, dict]:
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    height, width = int(input_info["shape"][1]), int(input_info["shape"][2])
    if input_info["dtype"] != np.float32:
        raise ValueError(f"Expected float32 input, got {input_info['dtype']}")
    resized = np.stack(
        [
            np.asarray(
                Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
            for image in images
        ]
    )
    probabilities = []
    for image in resized:
        batch = ((image.astype(np.float32) - 127.5) / 127.5)[None, ...]
        interpreter.set_tensor(input_info["index"], batch)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_info["index"])[0].astype(np.float32)
        if np.any(~np.isfinite(scores)):
            raise RuntimeError("TFLite model produced non-finite scores")
        if scores.min() < -0.01 or scores.max() > 1.01 or abs(scores.sum() - 1.0) > 0.1:
            raise RuntimeError("TFLite output does not match declared probability format")
        probabilities.append(scores)
    details = {
        "input_shape": input_info["shape"].tolist(),
        "input_dtype": np.dtype(input_info["dtype"]).name,
        "output_shape": output_info["shape"].tolist(),
        "output_dtype": np.dtype(output_info["dtype"]).name,
    }
    return np.stack(probabilities), details


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> tuple[float, list]:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows = []
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index < bins - 1:
            mask = (confidence >= lower) & (confidence < upper)
        else:
            mask = (confidence >= lower) & (confidence <= upper)
        count = int(mask.sum())
        if count == 0:
            rows.append({"lower": float(lower), "upper": float(upper), "count": 0, "accuracy": None, "mean_confidence": None})
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        ece += (count / len(labels)) * abs(accuracy - mean_confidence)
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    return float(ece), rows


def apply_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return probabilities.max(axis=1) >= threshold


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float | None = None) -> dict:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    accepted = np.ones(len(labels), dtype=bool) if threshold is None else apply_threshold(probabilities, threshold)
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(LABELS))), zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(LABELS))))
    actual_condition = CONDITION_OF_LABEL[labels]
    predicted_condition = CONDITION_OF_LABEL[predictions]
    condition_matrix = confusion_matrix(actual_condition, predicted_condition, labels=[0, 1])
    error_decomposition = Counter()
    for truth, prediction in zip(labels, predictions):
        if truth == prediction:
            error_decomposition["correct"] += 1
        elif CONDITION_OF_LABEL[truth] == CONDITION_OF_LABEL[prediction]:
            error_decomposition["same_condition_wrong_fruit"] += 1
        elif FRUIT_OF_LABEL[truth] == FRUIT_OF_LABEL[prediction]:
            error_decomposition["same_fruit_wrong_condition"] += 1
        else:
            error_decomposition["wrong_fruit_and_condition"] += 1
    accepted_count = int(accepted.sum())
    accepted_labels = labels[accepted]
    accepted_predictions = predictions[accepted]
    accepted_correct = accepted_predictions == accepted_labels
    accepted_accuracy = float(accepted_correct.mean()) if accepted_count else None
    accepted_macro_f1 = (
        float(f1_score(accepted_labels, accepted_predictions, labels=list(range(len(LABELS))), average="macro", zero_division=0))
        if accepted_count
        else None
    )
    confidence_bins = []
    for lower, upper in ((0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.000001)):
        mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        confidence_bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": float((predictions[mask] == labels[mask]).mean()) if count else None,
            }
        )
    ece, calibration_rows = expected_calibration_error(labels, probabilities)
    per_fruit = {}
    for fruit in FRUIT_NAMES:
        fresh_index, rotten_index = FRUIT_LABEL_INDICES[fruit]
        fruit_mask = (labels == fresh_index) | (labels == rotten_index)
        truth = np.where(labels[fruit_mask] == rotten_index, 1, 0)
        predicted = np.where(
            predictions[fruit_mask] == rotten_index,
            1,
            np.where(predictions[fruit_mask] == fresh_index, 0, 2),
        )
        per_fruit[fruit] = {
            "count": int(fruit_mask.sum()),
            "condition_correct": int((truth == predicted).sum()),
            "condition_errors": int((truth != predicted).sum()),
            "false_fresh": int(((truth == 1) & (predicted == 0)).sum()),
            "false_rotten": int(((truth == 0) & (predicted == 1)).sum()),
            "other_fruit_predictions": int((predicted == 2).sum()),
        }
    result = {
        "count": len(labels),
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=list(range(len(LABELS))))),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "min_confidence": float(confidence.min()),
        "max_confidence": float(confidence.max()),
        "expected_calibration_error_10bin": ece,
        "calibration_bins": calibration_rows,
        "confidence_accuracy_bins": confidence_bins,
        "per_class": {
            label: {
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(per_class_support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": matrix.tolist(),
        "condition_accuracy": float((actual_condition == predicted_condition).mean()),
        "condition_macro_f1": float(f1_score(actual_condition, predicted_condition, labels=[0, 1], average="macro", zero_division=0)),
        "condition_confusion_matrix": condition_matrix.tolist(),
        "false_fresh_count": int(((actual_condition == 1) & (predicted_condition == 0)).sum()),
        "false_rotten_count": int(((actual_condition == 0) & (predicted_condition == 1)).sum()),
        "error_decomposition": dict(error_decomposition),
        "per_fruit": per_fruit,
        "threshold": threshold,
        "accepted_count": accepted_count,
        "coverage": accepted_count / len(labels),
        "accepted_accuracy": accepted_accuracy,
        "accepted_macro_f1": accepted_macro_f1,
        "abstained_count": len(labels) - accepted_count,
    }
    return result, predictions, confidence, accepted


def threshold_sweep(labels: np.ndarray, probabilities: np.ndarray, target_coverage: float) -> dict:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    candidate_thresholds = np.unique(confidence)
    candidates = []
    for threshold in candidate_thresholds:
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        if coverage < target_coverage:
            continue
        accepted_accuracy = float((predictions[accepted] == labels[accepted]).mean())
        candidates.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "accepted_count": int(accepted.sum()),
                "accepted_accuracy": accepted_accuracy,
            }
        )
    if not candidates:
        chosen = {"threshold": 0.0, "coverage": 1.0, "accepted_count": len(labels), "accepted_accuracy": float((predictions == labels).mean())}
    else:
        candidates.sort(key=lambda item: (item["accepted_accuracy"], item["coverage"], item["threshold"]), reverse=True)
        chosen = candidates[0]
    sweep = []
    for threshold in np.linspace(0.0, 0.99, 100):
        accepted = confidence >= threshold
        sweep.append(
            {
                "threshold": float(threshold),
                "coverage": float(accepted.mean()),
                "accepted_count": int(accepted.sum()),
                "accepted_accuracy": float((predictions[accepted] == labels[accepted]).mean()) if accepted.any() else None,
            }
        )
    return {
        "target_coverage": target_coverage,
        "selection_rule": "Among thresholds with coverage >= target, maximize accepted accuracy, then coverage, then threshold.",
        "chosen": chosen,
        "sweep": sweep,
    }


def save_predictions(
    path: Path,
    rows: list[dict],
    labels: np.ndarray,
    probabilities: np.ndarray,
    accepted: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["relative_path", "label", "predicted", "max_score", "accepted"] + [f"prob_{label}" for label in LABELS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, truth, scores, is_accepted in zip(rows, labels, probabilities, accepted):
            record = {
                "relative_path": row["relative_path"],
                "label": LABELS[int(truth)],
                "predicted": LABELS[int(scores.argmax())],
                "max_score": float(scores.max()),
                "accepted": bool(is_accepted),
            }
            for index, label in enumerate(LABELS):
                record[f"prob_{label}"] = float(scores[index])
            writer.writerow(record)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["tflite", "keras"], default="tflite")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--manifest", default="ml/data/manifests/splits_v1.csv")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--target-coverage", type=float, default=0.95)
    parser.add_argument("--threshold-output", default=None)
    parser.add_argument("--threshold-json", default=None)
    args = parser.parse_args()

    model_path = resolve(args.model)
    manifest_path = resolve(args.manifest)
    rows = load_rows(args.split, manifest_path)
    labels = np.asarray([LABELS.index(row["label"]) for row in rows], dtype=np.int32)
    images = load_images(rows, 224)
    if args.backend == "tflite":
        probabilities, model_details = predict_tflite(model_path, images)
    else:
        probabilities = predict_keras(model_path, images)
        model_details = {"backend": "keras"}

    if args.threshold_json:
        existing_payload = read_json(resolve(args.threshold_json))
        threshold = float(existing_payload["chosen"]["threshold"])
    else:
        threshold = 0.0

    threshold_payload = None
    if args.select_threshold:
        threshold_payload = threshold_sweep(labels, probabilities, args.target_coverage)
        threshold = float(threshold_payload["chosen"]["threshold"])

    result, predictions, confidence, accepted = metrics(labels, probabilities, threshold)
    result["model"] = model_path.relative_to(PROJECT_ROOT).as_posix()
    result["model_sha256"] = sha256_file(model_path)
    result["model_details"] = model_details
    result["split"] = args.split
    result["split_manifest"] = manifest_path.relative_to(PROJECT_ROOT).as_posix()
    result["split_manifest_sha256"] = sha256_file(manifest_path)
    result["run_id"] = args.run_id
    result["label_counts"] = dict(Counter(row["label"] for row in rows))

    if args.select_threshold and threshold_payload is not None:
        threshold_payload["split"] = args.split
        threshold_payload["model"] = result["model"]
        threshold_payload["model_sha256"] = result["model_sha256"]
        threshold_payload["count"] = len(labels)
        threshold_payload["chosen_metrics"] = {
            key: result[key]
            for key in ("coverage", "accepted_count", "accepted_accuracy", "accepted_macro_f1", "abstained_count")
        }
        if args.threshold_output:
            write_json(resolve(args.threshold_output), threshold_payload)
        result["threshold_selection"] = threshold_payload

    report_path = resolve(args.report)
    predictions_path = resolve(args.predictions)
    write_json(report_path, result)
    save_predictions(predictions_path, rows, labels, probabilities, accepted)
    print(json.dumps(
        {
            "run_id": args.run_id,
            "split": args.split,
            "count": result["count"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "condition_accuracy": result["condition_accuracy"],
            "false_fresh_count": result["false_fresh_count"],
            "threshold": result["threshold"],
            "coverage": result["coverage"],
            "accepted_accuracy": result["accepted_accuracy"],
            "ece_10bin": result["expected_calibration_error_10bin"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
