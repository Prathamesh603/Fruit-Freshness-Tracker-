from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
DATA_ROOT = ML_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
MANIFEST_ROOT = DATA_ROOT / "manifests"
REPORT_ROOT = ML_ROOT / "reports"
CONFIG_ROOT = ML_ROOT / "configs"
CHECKPOINT_ROOT = ML_ROOT / "checkpoints"
EXPORT_ROOT = PROJECT_ROOT / "model_exports"

SOURCE_ID = "mendeley_bdd69gyhv8_v1"
SOURCE_ROOT = RAW_ROOT / "mendeley_bdd69gyhv8"
ORIGINAL_ZIP = SOURCE_ROOT / "Original_Image.zip"
AUGMENTED_ZIP = SOURCE_ROOT / "Augmented_Image.zip"
EXTRACTED_ROOT = SOURCE_ROOT / "originals"

LABELS = [
    "fresh_apple",
    "fresh_banana",
    "fresh_orange",
    "rotten_apple",
    "rotten_banana",
    "rotten_orange",
]

LABEL_META = [
    {"id": "fresh_apple", "fruit": "apple", "condition": "fresh", "displayName": "Apple - No visible spoilage"},
    {"id": "fresh_banana", "fruit": "banana", "condition": "fresh", "displayName": "Banana - No visible spoilage"},
    {"id": "fresh_orange", "fruit": "orange", "condition": "fresh", "displayName": "Orange - No visible spoilage"},
    {"id": "rotten_apple", "fruit": "apple", "condition": "rotten", "displayName": "Apple - Visible spoilage"},
    {"id": "rotten_banana", "fruit": "banana", "condition": "rotten", "displayName": "Banana - Visible spoilage"},
    {"id": "rotten_orange", "fruit": "orange", "condition": "rotten", "displayName": "Orange - Visible spoilage"},
]

SOURCE_CLASS_MAP = {
    "FreshApple": "fresh_apple",
    "FreshBanana": "fresh_banana",
    "FreshOrange": "fresh_orange",
    "RottenApple": "rotten_apple",
    "RottenBanana": "rotten_banana",
    "RottenOrange": "rotten_orange",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def labels_payload() -> list[dict]:
    return [dict(item) for item in LABEL_META]


def label_index(label: str) -> int:
    return LABELS.index(label)


def normalization_mean_std() -> tuple[list[float], list[float]]:
    return [127.5, 127.5, 127.5], [127.5, 127.5, 127.5]


def normalize_rgb_uint8(image: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) - 127.5) / 127.5


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
