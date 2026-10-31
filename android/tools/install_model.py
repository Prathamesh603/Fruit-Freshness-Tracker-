"""Validate and copy a training bundle into Android assets. Standard library only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

APP_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = ("freshness.tflite", "model.json", "labels.json", "MODEL_CARD.md", "evaluation.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_bundle(bundle: Path) -> dict:
    bundle = bundle.resolve(strict=True)
    require(bundle.is_dir(), "Bundle must be a directory.")
    for name in REQUIRED_FILES:
        file = bundle / name
        require(file.is_file() and not file.is_symlink(), f"Missing or symlinked bundle file: {name}")
        require(file.resolve().parent == bundle, f"File is outside the bundle: {name}")
        require(0 < file.stat().st_size <= (64 * 1024 * 1024 if name.endswith(".tflite") else 1024 * 1024),
                f"Empty or oversized bundle file: {name}")
    manifest = json.loads((bundle / "model.json").read_text(encoding="utf-8"))
    require(manifest.get("schemaVersion") == 1, "Unsupported schemaVersion.")
    require(isinstance(manifest.get("modelId"), str) and bool(manifest["modelId"].strip()), "Missing modelId.")
    require(manifest.get("modelFile") == "freshness.tflite", "modelFile must be freshness.tflite.")
    model = (bundle / "freshness.tflite").read_bytes()
    require(len(model) >= 8 and model[4:8] == b"TFL3", "File does not have a TFLite identifier.")
    require(hashlib.sha256(model).hexdigest() == manifest.get("sha256"), "Model SHA-256 mismatch.")
    settings = manifest.get("input", {})
    for key, expected in {"layout": "NHWC", "channels": 3, "colorSpace": "RGB", "resizeMode": "stretch"}.items():
        require(settings.get(key) == expected, f"Unsupported input.{key}; expected {expected}.")
    for key in ("width", "height"):
        require(type(settings.get(key)) is int and 16 <= settings[key] <= 1024, f"Invalid input.{key}.")
    require(settings.get("interpolation", "bilinear") in ("bilinear", "nearest"), "Unsupported input.interpolation.")
    for key in ("mean", "std"):
        values = settings.get(key)
        require(isinstance(values, list) and len(values) == 3, f"input.{key} must have three numbers.")
        require(all(type(v) in (int, float) and math.isfinite(v) for v in values), f"Invalid input.{key}.")
        if key == "std":
            require(all(v > 0 for v in values), "input.std must be positive.")
    require(manifest.get("output", {}).get("scoreType") in ("probabilities", "logits"), "Invalid output.scoreType.")
    labels = manifest.get("labels")
    require(isinstance(labels, list) and 2 <= len(labels) <= 100, "Expected 2-100 labels.")
    for label in labels:
        require(isinstance(label, dict) and all(isinstance(label.get(k), str) and bool(label[k].strip())
                for k in ("id", "fruit", "condition", "displayName")), "Malformed structured label.")
    require(len({v["id"] for v in labels}) == len(labels), "Duplicate label IDs.")
    require(json.loads((bundle / "labels.json").read_text(encoding="utf-8")) == labels,
            "labels.json must match the manifest label order and metadata exactly.")
    decision = manifest.get("decision", {})
    threshold = decision.get("minScore")
    require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1, "Invalid minScore.")
    require(type(decision.get("thresholdValidated")) is bool, "thresholdValidated must be Boolean.")
    require(isinstance(json.loads((bundle / "evaluation.json").read_text(encoding="utf-8")), dict),
            "evaluation.json must contain an object.")
    return manifest


def install_bundle(bundle: Path, assets: Path, replace: bool = False) -> Path:
    bundle = bundle.resolve(strict=True)
    manifest = validate_bundle(bundle)
    assets = assets.resolve()
    target = assets / "model"
    require(not target.exists() or replace, "A model already exists. Use --replace to preserve it as a backup and install this one.")
    require(not target.is_symlink(), "Refusing to replace a symlinked model directory.")
    assets.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".model-stage-", dir=assets)).resolve()
    try:
        for name in REQUIRED_FILES:
            shutil.copyfile(bundle / name, stage / name)
        validate_bundle(stage)
        if target.exists():
            # Backups stay outside packaged assets; no recursive deletion of prior work.
            backup_root = APP_ROOT / ".model-backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            target.rename(backup_root / stamp)
        stage.rename(target)
    finally:
        if stage.exists():
            # Verify the exact generated path before recursively cleaning up staging.
            require(stage.parent == assets and stage.name.startswith(".model-stage-"), "Invalid staging path.")
            shutil.rmtree(stage)
    print(f"Installed {manifest['modelId']} ({len(manifest['labels'])} classes) at {target}")
    print("This checks the bundle contract, not model accuracy, license clearance, or executable tensor parity.")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--replace", action="store_true", help="Preserve the installed model as a backup before replacing it")
    args = parser.parse_args()
    try:
        install_bundle(args.bundle, APP_ROOT / "app/src/main/assets", args.replace)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(1, f"Model was not installed: {error}\n")


if __name__ == "__main__":
    main()
