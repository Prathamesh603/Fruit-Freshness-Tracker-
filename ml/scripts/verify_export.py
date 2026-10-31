from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

from common import LABELS, PROJECT_ROOT, sha256_file, write_json
from evaluate import predict_tflite


def load_rows(split: str, manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def load_images(rows: list[dict], size: int) -> np.ndarray:
    images = []
    for row in rows:
        with Image.open(PROJECT_ROOT / row["relative_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((size, size), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras", required=True)
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--manifest", default="ml/data/manifests/splits_v1.csv")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    keras_path = resolve(args.keras)
    tflite_path = resolve(args.tflite)
    manifest_path = resolve(args.manifest)
    rows = load_rows(args.split, manifest_path)
    images = load_images(rows, 224)
    truth = np.asarray([LABELS.index(row["label"]) for row in rows], dtype=np.int32)

    keras_model = keras.models.load_model(keras_path)
    keras_probabilities = keras_model.predict(
        (images.astype(np.float32) - 127.5) / 127.5,
        batch_size=32,
        verbose=0,
    )
    tflite_probabilities, model_details = predict_tflite(tflite_path, images)
    absolute_difference = np.abs(keras_probabilities - tflite_probabilities)
    keras_predictions = keras_probabilities.argmax(axis=1)
    tflite_predictions = tflite_probabilities.argmax(axis=1)
    result = {
        "split": args.split,
        "count": len(rows),
        "keras": keras_path.relative_to(PROJECT_ROOT).as_posix(),
        "keras_sha256": sha256_file(keras_path),
        "tflite": tflite_path.relative_to(PROJECT_ROOT).as_posix(),
        "tflite_sha256": sha256_file(tflite_path),
        "tflite_model_details": model_details,
        "max_absolute_probability_difference": float(absolute_difference.max()),
        "mean_absolute_probability_difference": float(absolute_difference.mean()),
        "top1_agreement": float((keras_predictions == tflite_predictions).mean()),
        "top1_disagreements": int((keras_predictions != tflite_predictions).sum()),
        "keras_accuracy": float((keras_predictions == truth).mean()),
        "tflite_accuracy": float((tflite_predictions == truth).mean()),
        "per_example_max_difference": [
            {
                "relative_path": row["relative_path"],
                "label": LABELS[int(label)],
                "keras_prediction": LABELS[int(keras_prediction)],
                "tflite_prediction": LABELS[int(tflite_prediction)],
                "max_probability_difference": float(difference),
            }
            for row, label, keras_prediction, tflite_prediction, difference in zip(
                rows,
                truth,
                keras_predictions,
                tflite_predictions,
                absolute_difference.max(axis=1),
            )
        ],
    }
    write_json(resolve(args.report), result)
    print(json.dumps(
        {
            key: result[key]
            for key in (
                "count",
                "max_absolute_probability_difference",
                "mean_absolute_probability_difference",
                "top1_agreement",
                "top1_disagreements",
                "keras_accuracy",
                "tflite_accuracy",
            )
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
