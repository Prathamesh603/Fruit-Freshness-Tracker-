from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image, ImageOps

from common import LABELS, PROJECT_ROOT, sha256_file, write_json

CANDIDATE_TO_CANONICAL = {
    0: "fresh_apple",
    1: "fresh_banana",
    2: "fresh_orange",
    3: "rotten_apple",
    4: "rotten_banana",
    5: "rotten_orange",
}
UNRIPE = {6: "unripe_apple", 7: "unripe_banana", 8: "unripe_orange"}
CONDITION_OF_LABEL = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)


def load_rows(split: str, manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def load_images(rows: list[dict], width: int, height: int) -> np.ndarray:
    images = []
    for row in rows:
        with Image.open(PROJECT_ROOT / row["relative_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--manifest", default="ml/data/manifests/splits_v1.csv")
    parser.add_argument("--report", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    model_path = resolve(args.model)
    labels_path = resolve(args.labels)
    manifest_path = resolve(args.manifest)
    rows = load_rows(args.split, manifest_path)
    truth = np.asarray([LABELS.index(row["label"]) for row in rows], dtype=np.int32)
    candidate_labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(candidate_labels) != 9:
        raise ValueError(f"Expected 9 candidate labels, got {len(candidate_labels)}")

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    height, width = int(input_info["shape"][1]), int(input_info["shape"][2])
    images = load_images(rows, width, height)
    probabilities = []
    for image in images:
        input_tensor = (image.astype(np.float32) / 255.0)[None, ...]
        interpreter.set_tensor(input_info["index"], input_tensor)
        interpreter.invoke()
        probabilities.append(interpreter.get_tensor(output_info["index"])[0].astype(np.float32))
    probabilities = np.stack(probabilities)
    candidate_predictions = probabilities.argmax(axis=1)
    canonical_predictions = np.full(len(rows), -1, dtype=np.int32)
    for candidate_index, canonical_label in CANDIDATE_TO_CANONICAL.items():
        canonical_predictions[candidate_predictions == candidate_index] = LABELS.index(canonical_label)

    valid = canonical_predictions >= 0
    correct = canonical_predictions == truth
    confusion = np.zeros((6, 7), dtype=np.int64)
    for actual, predicted in zip(truth, canonical_predictions):
        column = int(predicted) if predicted >= 0 else 6
        confusion[actual, column] += 1

    per_class = {}
    for index, label in enumerate(LABELS):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    macro_f1 = float(np.mean([entry["f1"] for entry in per_class.values()]))
    valid_mask = valid
    actual_condition = CONDITION_OF_LABEL[truth]
    predicted_condition = np.full(len(rows), -1, dtype=np.int32)
    predicted_condition[valid_mask] = CONDITION_OF_LABEL[canonical_predictions[valid_mask]]
    condition_correct = predicted_condition == actual_condition
    false_fresh = int(((actual_condition == 1) & (predicted_condition == 0)).sum())
    false_rotten = int(((actual_condition == 0) & (predicted_condition == 1)).sum())
    fruit_of_actual = truth % 3
    fruit_of_candidate = np.asarray(
        [candidate_index % 3 if candidate_index < 6 else candidate_index - 6 for candidate_index in candidate_predictions]
    )
    result = {
        "candidate": {
            "model": model_path.relative_to(PROJECT_ROOT).as_posix(),
            "model_sha256": sha256_file(model_path),
            "labels": candidate_labels,
            "model_bytes": model_path.stat().st_size,
            "license_status": "unresolved",
            "preprocessing_assumption": "RGB, stretch resize, float32 [0,1] from the published notebook",
            "input_shape": input_info["shape"].tolist(),
            "output_shape": output_info["shape"].tolist(),
        },
        "split": args.split,
        "count": len(rows),
        "mapped_label_accuracy": float(correct.mean()),
        "mapped_macro_f1": macro_f1,
        "valid_six_class_coverage": float(valid.mean()),
        "unripe_predictions": int((~valid).sum()),
        "accuracy_on_valid_predictions": float(correct[valid].mean()) if valid.any() else None,
        "predicted_label_distribution": dict(Counter(
            CANDIDATE_TO_CANONICAL.get(int(index), UNRIPE.get(int(index), f"index_{index}"))
            for index in candidate_predictions
        )),
        "condition_accuracy_over_all": float(condition_correct.mean()),
        "false_fresh_count": false_fresh,
        "false_rotten_count": false_rotten,
        "fruit_only_accuracy": float((fruit_of_candidate == fruit_of_actual).mean()),
        "per_class": per_class,
        "confusion_matrix_actual_by_predicted": confusion.tolist(),
        "confusion_columns": LABELS + ["unripe"],
        "notes": [
            "Predictions outside the six canonical classes are counted as errors and shown as 'unripe'.",
            "This candidate was trained on a different dataset and is not licensed for redistribution.",
        ],
    }
    report_path = resolve(args.report)
    write_json(report_path, result)

    predictions_path = resolve(args.predictions)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "label", "candidate_label", "mapped_prediction", "max_score"])
        for row, actual, candidate_index, scores in zip(rows, truth, candidate_predictions, probabilities):
            writer.writerow(
                [
                    row["relative_path"],
                    LABELS[int(actual)],
                    candidate_labels[int(candidate_index)],
                    CANDIDATE_TO_CANONICAL.get(int(candidate_index), UNRIPE.get(int(candidate_index), "unknown")),
                    float(scores.max()),
                ]
            )
    print(json.dumps(
        {
            key: result[key]
            for key in (
                "count",
                "mapped_label_accuracy",
                "mapped_macro_f1",
                "valid_six_class_coverage",
                "unripe_predictions",
                "condition_accuracy_over_all",
                "false_fresh_count",
                "false_rotten_count",
            )
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
