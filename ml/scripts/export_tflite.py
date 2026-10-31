from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
from tensorflow import keras

from common import PROJECT_ROOT, sha256_bytes, sha256_file, write_json

TFLITE_OPS = [tf.lite.OpsSet.TFLITE_BUILTINS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    model = keras.models.load_model(checkpoint)
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    converter.target_spec.supported_ops = TFLITE_OPS
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    started = time.time()
    tflite_model = converter.convert()
    conversion_seconds = time.time() - started

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(tflite_model)

    interpreter = Interpreter(model_path=str(output))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    rng = np.random.default_rng(42)
    sample = rng.uniform(-1.0, 1.0, size=input_info["shape"]).astype(input_info["dtype"])
    interpreter.set_tensor(input_info["index"], sample)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_info["index"])
    ops = interpreter._get_ops_details()
    metadata = {
        "source_checkpoint": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "tflite_path": output.relative_to(PROJECT_ROOT).as_posix(),
        "tflite_bytes": len(tflite_model),
        "tflite_sha256": sha256_bytes(tflite_model),
        "conversion_seconds": round(conversion_seconds, 3),
        "converter": {
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "optimizations": [],
            "supported_ops": ["TFLITE_BUILTINS"],
            "input_type": "float32",
            "output_type": "float32",
        },
        "input": {
            "name": input_info["name"],
            "shape": input_info["shape"].tolist(),
            "dtype": np.dtype(input_info["dtype"]).name,
        },
        "output": {
            "name": output_info["name"],
            "shape": output_info["shape"].tolist(),
            "dtype": np.dtype(output_info["dtype"]).name,
        },
        "sample_output_sum": float(scores.sum()),
        "op_count": len(ops),
        "op_names": sorted({op["op_name"] for op in ops}),
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
