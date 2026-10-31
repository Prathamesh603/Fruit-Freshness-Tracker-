from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image, ImageOps

from common import PROJECT_ROOT, sha256_file, write_json


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_rows(split: str, manifest_path: Path, limit: int) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    return rows[:limit]


def load_images(rows: list[dict], width: int, height: int) -> np.ndarray:
    images = []
    for row in rows:
        with Image.open(PROJECT_ROOT / row["relative_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images)


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean_ms": float(array.mean() * 1000),
        "min_ms": float(array.min() * 1000),
        "p50_ms": float(np.percentile(array, 50) * 1000),
        "p90_ms": float(np.percentile(array, 90) * 1000),
        "p95_ms": float(np.percentile(array, 95) * 1000),
        "max_ms": float(array.max() * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="ml/data/manifests/splits_v1.csv")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--images", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    model_path = resolve(args.model)
    manifest_path = resolve(args.manifest)
    rows = load_rows(args.split, manifest_path, args.images)
    load_started = time.perf_counter()
    interpreter = Interpreter(model_path=str(model_path), num_threads=args.threads)
    interpreter.allocate_tensors()
    load_seconds = time.perf_counter() - load_started
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    height, width = int(input_info["shape"][1]), int(input_info["shape"][2])
    images = load_images(rows, width, height)
    normalized = ((images.astype(np.float32) - 127.5) / 127.5)[:, None, ...]

    for index in range(args.warmup):
        interpreter.set_tensor(input_info["index"], normalized[index % len(normalized)])
        interpreter.invoke()

    inference_times = []
    end_to_end_times = []
    for index in range(args.runs):
        sample = normalized[index % len(normalized)]
        interpreter.set_tensor(input_info["index"], sample)
        start = time.perf_counter()
        interpreter.invoke()
        inference_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        with Image.open(PROJECT_ROOT / rows[index % len(rows)]["relative_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32)
        batch = ((array - 127.5) / 127.5)[None, ...]
        interpreter.set_tensor(input_info["index"], batch)
        interpreter.invoke()
        end_to_end_times.append(time.perf_counter() - start)

    result = {
        "model": model_path.relative_to(PROJECT_ROOT).as_posix(),
        "model_sha256": sha256_file(model_path),
        "model_bytes": model_path.stat().st_size,
        "runtime": {
            "backend": "ai-edge-litert",
            "threads": args.threads,
            "input_shape": input_info["shape"].tolist(),
            "input_dtype": np.dtype(input_info["dtype"]).name,
            "output_shape": output_info["shape"].tolist(),
        },
        "load_seconds": load_seconds,
        "inference_only": summarize(inference_times),
        "desktop_preprocess_plus_inference": summarize(end_to_end_times),
        "notes": [
            "Measured on the development Windows CPU with two LiteRT threads; this is not an Android device benchmark.",
            "desktop_preprocess_plus_inference includes PIL decode, EXIF transpose, RGB conversion, resize, float normalization, tensor copy, and invoke.",
            "Do not substitute these timings for on-device measurements.",
        ],
    }
    write_json(resolve(args.report), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
