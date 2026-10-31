from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import f1_score
from tensorflow import keras

from common import (
    CHECKPOINT_ROOT,
    CONFIG_ROOT,
    LABELS,
    MANIFEST_ROOT,
    PROJECT_ROOT,
    read_json,
    sha256_file,
    write_json,
)

AUTOTUNE = tf.data.AUTOTUNE


def load_rows(split: str, manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    return rows


def read_one(row: dict, size: int) -> np.ndarray:
    path = PROJECT_ROOT / row["relative_path"]
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def load_images(rows: list[dict], size: int, cache_key: str) -> np.ndarray:
    cache_dir = CHECKPOINT_ROOT / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}_{size}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
        images = list(executor.map(lambda row: read_one(row, size), rows))
    array = np.stack(images).astype(np.uint8)
    np.save(cache_path, array)
    return array


def build_model(config: dict) -> keras.Model:
    backbone = keras.applications.MobileNetV3Small(
        include_top=False,
        weights=config["weights"],
        input_shape=(config["input_size"], config["input_size"], 3),
        include_preprocessing=False,
        name="backbone",
    )
    inputs = keras.Input(shape=(config["input_size"], config["input_size"], 3), name="image")
    features = backbone(inputs, training=False)
    pooled = keras.layers.GlobalAveragePooling2D(name="global_pool")(features)
    dropped = keras.layers.Dropout(config["dropout"], name="dropout")(pooled)
    outputs = keras.layers.Dense(len(LABELS), activation="softmax", name="condition")(dropped)
    return keras.Model(inputs=inputs, outputs=outputs, name=config["run_id"])


def augment(image: np.ndarray, label: int, size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32)
    source_size = tf.shape(image)[0]
    crop_size = tf.random.uniform([], minval=int(size * 0.88), maxval=source_size + 1, dtype=tf.int32)
    image = tf.image.random_crop(image, [crop_size, crop_size, 3])
    image = tf.image.resize(image, (size, size), method="bilinear")
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=12.0)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = (image - 127.5) / 127.5
    return image, label


def normalize(image: np.ndarray, label: int) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5
    return image, label


def make_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    training: bool,
    size: int,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(images), seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.map(lambda image, label: augment(image, label, size), num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(normalize, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


class MacroF1Callback(keras.callbacks.Callback):
    def __init__(self, val_images: np.ndarray, val_labels: np.ndarray, batch_size: int) -> None:
        super().__init__()
        self.val_images = val_images
        self.val_labels = val_labels
        self.batch_size = batch_size

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs if logs is not None else {}
        images = (self.val_images.astype(np.float32) - 127.5) / 127.5
        probabilities = self.model.predict(images, batch_size=self.batch_size, verbose=0)
        predictions = probabilities.argmax(axis=1)
        logs["val_macro_f1"] = float(f1_score(self.val_labels, predictions, average="macro"))


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def configure_fine_tune(model: keras.Model, config: dict) -> None:
    backbone = model.get_layer("backbone")
    backbone.trainable = True
    for layer in backbone.layers[:-config["fine_tune_layers"]]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--initialize-from", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = read_json(config_path)
    run_dir = CHECKPOINT_ROOT / config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)

    keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()

    manifest_path = PROJECT_ROOT / config["split_manifest"]
    split_hash = sha256_file(manifest_path)[:12]
    train_rows = load_rows("train", manifest_path)
    val_rows = load_rows("val", manifest_path)
    train_images = load_images(train_rows, config["train_size"], f"train_{split_hash}")
    val_images = load_images(val_rows, config["input_size"], f"val_{split_hash}")
    train_labels = np.asarray([LABELS.index(row["label"]) for row in train_rows], dtype=np.int32)
    val_labels = np.asarray([LABELS.index(row["label"]) for row in val_rows], dtype=np.int32)

    initialize_from = args.initialize_from or config.get("initialize_from")
    if initialize_from:
        initialize_path = Path(initialize_from)
        if not initialize_path.is_absolute():
            initialize_path = PROJECT_ROOT / initialize_path
        model = keras.models.load_model(initialize_path)
        configure_fine_tune(model, config)
    else:
        model = build_model(config)
        model.get_layer("backbone").trainable = False
    compile_model(model, config["learning_rate"])
    model.summary()

    train_dataset = make_dataset(
        train_images,
        train_labels,
        config["batch_size"],
        True,
        config["input_size"],
        config["seed"],
    )
    val_dataset = make_dataset(
        val_images,
        val_labels,
        config["batch_size"],
        False,
        config["input_size"],
        config["seed"],
    )

    best_path = run_dir / "best.keras"
    callbacks = [
        MacroF1Callback(val_images, val_labels, config["batch_size"]),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=config["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        keras.callbacks.TerminateOnNaN(),
    ]

    started = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config["epochs"],
        callbacks=callbacks,
        verbose=2,
    )
    duration = time.time() - started
    model.save(run_dir / "last.keras")
    best_model = keras.models.load_model(best_path)
    val_inputs = (val_images.astype(np.float32) - 127.5) / 127.5
    val_probabilities = best_model.predict(val_inputs, batch_size=config["batch_size"], verbose=0)
    val_predictions = val_probabilities.argmax(axis=1)
    val_accuracy = float((val_predictions == val_labels).mean())
    val_macro_f1 = float(f1_score(val_labels, val_predictions, average="macro"))
    history_payload = {key: [float(value) for value in values] for key, values in history.history.items()}
    summary = {
        "run_id": config["run_id"],
        "config": config,
        "config_path": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "split_manifest": config["split_manifest"],
        "split_manifest_sha256": sha256_file(manifest_path),
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "train_label_counts": dict(Counter(row["label"] for row in train_rows)),
        "val_label_counts": dict(Counter(row["label"] for row in val_rows)),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "duration_seconds": round(duration, 2),
        "best_epoch": int(np.argmax(history.history["val_macro_f1"]) + 1),
        "best_val_macro_f1": val_macro_f1,
        "best_val_accuracy": val_accuracy,
        "final_val_loss": float(np.min(history.history["val_loss"])),
        "history": history_payload,
        "best_checkpoint": best_path.relative_to(PROJECT_ROOT).as_posix(),
        "best_checkpoint_sha256": sha256_file(best_path),
    }
    write_json(run_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
