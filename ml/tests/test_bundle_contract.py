from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT_ROOT / "model_exports" / "mobilenetv3small-sixclass-v1"
MODEL = BUNDLE / "freshness.tflite"
SPEC = BUNDLE / "model.json"
LABELS_PATH = BUNDLE / "labels.json"
PARITY = BUNDLE / "parity" / "expected.json"
INSTALLER = PROJECT_ROOT / "android" / "tools" / "install_model.py"

EXPECTED_LABELS = [
    "fresh_apple",
    "fresh_banana",
    "fresh_orange",
    "rotten_apple",
    "rotten_banana",
    "rotten_orange",
]
EXPECTED_MODEL_SHA256 = "dd519afd78d27d731fc4cef0e9a02b96086e42122a51a768874b8b6fdfc9052e"
EXPECTED_THRESHOLD = 0.5833432078361511


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_installer():
    spec = importlib.util.spec_from_file_location("fruitlens_install_model", INSTALLER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_passes_android_validation() -> None:
    installer = load_installer()
    manifest = installer.validate_bundle(BUNDLE)
    assert manifest["modelId"] == "mobilenetv3small-sixclass-v1"
    assert manifest["sha256"] == EXPECTED_MODEL_SHA256
    assert manifest["decision"]["minScore"] == EXPECTED_THRESHOLD
    assert manifest["decision"]["thresholdValidated"] is True
    assert manifest["input"]["colorSpace"] == "RGB"
    assert manifest["input"]["layout"] == "NHWC"
    assert manifest["input"]["resizeMode"] == "stretch"
    assert manifest["input"]["mean"] == [127.5, 127.5, 127.5]
    assert manifest["input"]["std"] == [127.5, 127.5, 127.5]
    assert [label["id"] for label in manifest["labels"]] == EXPECTED_LABELS


def test_spec_and_labels_match_the_contract() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    assert spec["schemaVersion"] == 1
    assert spec["modelFile"] == "freshness.tflite"
    assert spec["sha256"] == EXPECTED_MODEL_SHA256
    assert spec["sha256"] == sha256_file(MODEL)
    assert spec["input"]["width"] == 224
    assert spec["input"]["height"] == 224
    assert spec["input"]["channels"] == 3
    assert spec["input"]["mean"] == [127.5, 127.5, 127.5]
    assert spec["input"]["std"] == [127.5, 127.5, 127.5]
    assert spec["output"]["scoreType"] == "probabilities"
    assert spec["output"]["tensorShape"] == [1, 6]
    assert spec["decision"]["minScore"] == EXPECTED_THRESHOLD
    assert spec["decision"]["thresholdValidated"] is True
    assert spec["redistribution"]["status"] == "unresolved"
    assert "CC BY 4.0" in spec["redistribution"]["notes"]
    assert "Apache-2.0" in spec["redistribution"]["notes"]
    assert [label["id"] for label in labels] == EXPECTED_LABELS
    assert spec["labels"] == labels


def test_tflite_tensor_contract() -> None:
    interpreter = Interpreter(model_path=str(MODEL))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    assert list(input_detail["shape"]) == [1, 224, 224, 3]
    assert input_detail["dtype"] == np.float32
    assert list(output_detail["shape"]) == [1, 6]
    assert output_detail["dtype"] == np.float32


def test_checksums_cover_every_bundle_file() -> None:
    expected = {}
    for line in (BUNDLE / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        expected[relative.strip()] = digest
    actual_files = {
        path.relative_to(BUNDLE).as_posix()
        for path in BUNDLE.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    assert set(expected) == actual_files
    for relative, digest in expected.items():
        assert sha256_file(BUNDLE / relative) == digest


def test_parity_fixtures_and_expected_scores() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    payload = json.loads(PARITY.read_text(encoding="utf-8"))
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    label_ids = [label["id"] for label in labels]
    assert payload["model_sha256"] == EXPECTED_MODEL_SHA256
    assert payload["labels"] == EXPECTED_LABELS
    assert payload["threshold"] == EXPECTED_THRESHOLD
    assert len(payload["fixtures"]) == 6

    mean = np.asarray(spec["input"]["mean"], dtype=np.float32)
    std = np.asarray(spec["input"]["std"], dtype=np.float32)
    interpreter = Interpreter(model_path=str(MODEL))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    seen = set()
    for fixture in payload["fixtures"]:
        path = BUNDLE / "parity" / "inputs" / fixture["file"]
        assert sha256_file(path) == fixture["sha256"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            assert image.size == (224, 224)
            pixels = np.asarray(image, dtype=np.float32)
        normalized = ((pixels - mean) / std)[None, ...]
        interpreter.set_tensor(input_detail["index"], normalized)
        interpreter.invoke()
        actual = interpreter.get_tensor(output_detail["index"])[0]
        expected = np.asarray(fixture["probabilities"], dtype=np.float32)
        np.testing.assert_allclose(actual, expected, atol=payload["absolute_tolerance"], rtol=0)
        assert abs(float(actual.sum()) - 1.0) < 1e-5
        top1 = label_ids[int(np.argmax(actual))]
        assert top1 == fixture["top1"] == fixture["label"]
        assert bool(float(actual.max()) >= EXPECTED_THRESHOLD) is fixture["accepted_at_bundle_threshold"]
        seen.add(top1)
    assert seen == set(EXPECTED_LABELS)
