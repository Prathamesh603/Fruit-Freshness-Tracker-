import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "install_model.py"
SPEC = importlib.util.spec_from_file_location("install_model", MODULE)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class BundleContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / "candidate"
        self.bundle.mkdir()
        # Synthetic identifier fixture tests contract validation only; never an inference model.
        model = b"\x00\x00\x00\x00TFL3fixture"
        self.labels = [{"id": "fresh_apple", "fruit": "apple", "condition": "fresh", "displayName": "Apple, no visible spoilage"},
                       {"id": "rotten_apple", "fruit": "apple", "condition": "rotten", "displayName": "Apple, visible spoilage"}]
        self.manifest = {
            "schemaVersion": 1, "modelId": "contract-test-fixture", "modelFile": "freshness.tflite",
            "sha256": hashlib.sha256(model).hexdigest(),
            "input": {"layout": "NHWC", "channels": 3, "colorSpace": "RGB", "resizeMode": "stretch",
                      "width": 224, "height": 224, "mean": [127.5] * 3, "std": [127.5] * 3},
            "output": {"scoreType": "probabilities"}, "labels": self.labels,
            "decision": {"minScore": 0.8, "thresholdValidated": False}}
        (self.bundle / "freshness.tflite").write_bytes(model)
        (self.bundle / "labels.json").write_text(json.dumps(self.labels), encoding="utf-8")
        (self.bundle / "MODEL_CARD.md").write_text("Synthetic contract test fixture, not a trained model.", encoding="utf-8")
        (self.bundle / "evaluation.json").write_text('{"fixture":true}', encoding="utf-8")
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.bundle / "model.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_valid_contract(self):
        self.assertEqual(installer.validate_bundle(self.bundle)["labels"], self.labels)

    def test_label_reordering_rejected(self):
        (self.bundle / "labels.json").write_text(json.dumps(list(reversed(self.labels))), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "label order"):
            installer.validate_bundle(self.bundle)

    def test_hash_mismatch_rejected_before_install(self):
        (self.bundle / "freshness.tflite").write_bytes(b"0000TFL3changed")
        assets = self.root / "assets"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            installer.install_bundle(self.bundle, assets)
        self.assertFalse(assets.exists())

    def test_path_escape_rejected(self):
        self.manifest["modelFile"] = "../../outside.tflite"
        self.save()
        with self.assertRaisesRegex(ValueError, "modelFile"):
            installer.validate_bundle(self.bundle)

    def test_invalid_normalization_rejected(self):
        self.manifest["input"]["std"][0] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, "positive"):
            installer.validate_bundle(self.bundle)

    def test_existing_model_preserved_without_replace(self):
        assets = self.root / "assets"
        target = assets / "model"
        target.mkdir(parents=True)
        marker = target / "existing.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            installer.install_bundle(self.bundle, assets)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
