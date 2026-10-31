"""Prepare the existing Smart Fruit classifier for LOCAL EVALUATION, not redistribution."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".tools/python"))
import tflite
from install_model import install_bundle

COMMIT = "200d8dffb48679567777a6ab5adc1d87eeea1c88"
BASE = f"https://raw.githubusercontent.com/JosephMoustaid/smart-fruit-models-notebooks/{COMMIT}/"


def main():
    bundle = ROOT / ".tools/reference-classifier"
    bundle.mkdir(parents=True, exist_ok=True)
    model_file = bundle / "freshness.tflite"
    if not model_file.exists():
        with urllib.request.urlopen(BASE + "ripeness_model.tflite", timeout=60) as response:
            model_file.write_bytes(response.read())
    model = tflite.Model.GetRootAsModel(model_file.read_bytes(), 0)
    graph = model.Subgraphs(0)
    if graph.InputsLength() != 1 or graph.OutputsLength() != 1:
        raise ValueError("Unexpected classifier signatures")
    input_tensor = graph.Tensors(graph.Inputs(0))
    output_tensor = graph.Tensors(graph.Outputs(0))
    dimensions = [input_tensor.Shape(i) for i in range(input_tensor.ShapeLength())]
    output = [output_tensor.Shape(i) for i in range(output_tensor.ShapeLength())]
    if dimensions != [1, 96, 96, 3] or output != [1, 9]:
        raise ValueError(f"Unexpected classifier tensor shapes: {dimensions}, {output}")
    if input_tensor.Type() != tflite.TensorType.FLOAT32 or output_tensor.Type() != tflite.TensorType.FLOAT32:
        raise ValueError("Reference IO differs from the expected float32 contract")
    with urllib.request.urlopen(BASE + "ripeness_labels.txt", timeout=30) as response:
        names = response.read().decode().splitlines()
    expected = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges",
                "unripe apple", "unripe banana", "unripe orange"]
    if names != expected:
        raise ValueError(f"Unexpected class order: {names}")
    labels = []
    for condition in ("fresh", "rotten", "unripe"):
        for fruit in ("apple", "banana", "orange"):
            description = {"fresh": "No visible spoilage", "rotten": "Visible spoilage", "unripe": "Unripe appearance"}[condition]
            labels.append({"id": f"{condition}_{fruit}", "fruit": fruit, "condition": condition,
                           "displayName": f"{fruit.title()} · {description}"})
    metadata = {
        "schemaVersion": 1, "modelId": "smart-fruit-mobilenetv2-reference-" + COMMIT[:8],
        "modelFile": "freshness.tflite", "sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        "input": {"width": 96, "height": 96, "channels": 3, "layout": "NHWC", "colorSpace": "RGB",
                  "resizeMode": "stretch", "interpolation": "nearest", "mean": [0, 0, 0], "std": [255, 255, 255]},
        "output": {"scoreType": "probabilities"}, "labels": labels,
        "decision": {"minScore": 0.8, "thresholdValidated": False},
        "redistribution": {"status": "unresolved", "notes": "Public repository has no visible license. Local evaluation candidate only."},
        "source": BASE + "ripeness_model.tflite", "sourceCommit": COMMIT,
    }
    (bundle / "model.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (bundle / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    (bundle / "evaluation.json").write_text(json.dumps({"status": "not_evaluated", "accuracy": None,
        "androidLatency": None, "tensorShapesVerified": True, "thresholdValidated": False}, indent=2), encoding="utf-8")
    (bundle / "MODEL_CARD.md").write_text(
        "# Smart Fruit reference classifier\n\n"
        "Local evaluation candidate. No independent accuracy evaluation or validated threshold.\n\n"
        f"Source revision: {COMMIT}\n\n"
        "The notebook rescales RGB pixels by 1/255 and trains a MobileNetV2-based classifier at 96x96. "
        "Its validation path/provenance requires independent auditing; repository claims are not app metrics. "
        "Output is nine combined fruit/condition categories. Color alone does not establish condition.\n\n"
        "Redistribution rights unresolved: do not publish a build containing these weights until permissions are established "
        "or replace this bundle with an appropriately licensed trained export.\n", encoding="utf-8")
    target = ROOT / "app/src/main/assets/model"
    if target.exists():
        print("An app model is already installed; reference bundle prepared without replacing it.")
    else:
        install_bundle(bundle, target.parent)
    print(f"Verified tensor shapes: {dimensions} -> {output}; {model_file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
