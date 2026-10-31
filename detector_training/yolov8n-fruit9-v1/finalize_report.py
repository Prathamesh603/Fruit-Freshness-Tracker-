"""Write the final model card from measured training/evaluation records."""
from pathlib import Path
import csv,json,hashlib,shutil

RUN=Path(__file__).resolve().parent


def main():
    def load(name):return json.loads((RUN/name).read_text(encoding='utf-8'))
    audit=load('reports/data_audit.json');evaluation=load('reports/test_evaluation.json')
    metadata=load('artifacts/model.json');environment=load('reports/environment.json')
    training=load('reports/training_summary.json');parity=load('reports/export_parity.json')
    with (RUN/'runs/train/results.csv').open(newline='',encoding='utf-8') as f:
        history=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f)]
    table='\n'.join(f"| {r['fruit']} | {r['test_images']} | {r['test_boxes']} | {100*r['precision']:.1f}% | {100*r['recall']:.1f}% | {100*r['AP50']:.1f}% | {100*r['AP50_95']:.1f}% |" for r in evaluation['per_class'])
    metrics=evaluation['metrics']
    files='\n'.join(f"| {name} | {v['bytes']:,} | `{v['sha256']}` |" for name,v in metadata['artifacts'].items())
    card=f'''# YOLOv8n nine-fruit detector: first experiment

## What this model does

Detects apple, banana, grape, lemon, mango, orange, pomegranate, strawberry and
tomato in an image. It outputs fruit identities and bounding boxes. It does NOT
predict freshness, internal quality, remaining shelf life, or food safety.

This is a completed experimental run, not a validated nine-fruit Android release.
Sparse categories and same-source testing limit what the reported scores establish.

## Training

- Architecture: YOLOv8n, initialized with official COCO-pretrained weights.
- Fruit classes: {len(metadata['names'])}; parameters: {metadata['parameters']:,}.
- Exact class order: `{', '.join(metadata['names'])}`.
- Training device: {environment['gpu']}; CUDA build {environment['cuda_build']}.
- PyTorch: {environment['torch']}; Ultralytics: {environment['packages']['ultralytics']}.
- Completed epochs: {len(history)}; maximum requested: 60; early-stopping patience: 12.
- Input size: 416 pixels; batch size: 8; optimizer: AdamW; initial LR: 0.001.
- Seed: 42; deterministic setting enabled; AMP disabled; disk/RAM image caching disabled.
- Only best/last checkpoints retained. The framework's validation fitness selected best.pt.
- Training invocation duration: {training['elapsed_seconds_this_invocation']/60:.2f} minutes.
- Full settings: reports/training_config.json; epoch history: runs/train/results.csv.

## Data preparation

The collected pool contained {audit['input_images']:,} images. After source filtering,
{audit['accepted_images']:,} remain. Exclusions: {audit['exclusion_reasons']}.

Final split: {audit['splits']['train']['images']:,} train,
{audit['splits']['val']['images']:,} validation, {audit['splits']['test']['images']:,} test.
There is no exact SHA256 or defined perceptual-group overlap between these splits.
Groups use 64-bit pHash, Hamming radius six, with eight rotation/reflection variants
and transitive union. Physical-fruit/session IDs are unavailable; these checks
cannot prove complete physical-fruit separation.

Split manifest SHA256: `{audit['manifest_sha256']}`.
The split was fixed before model training and test prediction. The held-out test
does not select epochs or deployment thresholds.

Prepared images are aspect-preserving RGB JPEG copies with the long side at most
640 pixels. Ultralytics letterboxes them to 416 for this experiment. The source
condition labels fresh/normal/rotten are collapsed to fruit identity only.
Source photos and existing classifier/app files are unchanged.

## Held-out test results

- Test images: {evaluation['images']}; labelled fruit boxes: {evaluation['boxes']}.
- mAP50: {100*metrics['metrics/mAP50(B)']:.2f}%.
- mAP50-95: {100*metrics['metrics/mAP50-95(B)']:.2f}%.
- Framework mean precision: {100*metrics['metrics/precision(B)']:.2f}%.
- Framework mean recall: {100*metrics['metrics/recall(B)']:.2f}%.

| Fruit | Test images | Test boxes | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|---:|
{table}

AP uses confidence floor 0.001, class-aware NMS IoU 0.7, and max_det 300.
Precision/recall use the framework's maximum-F1 operating point; they are not
performance at a separately validated app confidence threshold. The qualitative
sample sheet uses 0.25 solely as an illustrative prediction setting.

Mango precision may be reported as 1.0 when no predictions succeed; zero recall
and zero AP indicate failure, not successful detection. Pomegranate AP50 is based
on one test image/box and cannot establish reliability. Review the qualitative
sample sheet and reports/qualitative_review.json for observed wrong-class and
extra-box errors before considering deployment.

Confusion matrices and PR curves are in runs/test/. Read reports/test_evaluation.json
and reports/per_class_test.csv for the machine-readable results. The sample sheet
is reports/test_prediction_samples.jpg; it is not a complete failure analysis.

## Critical limitations

- Mango has one training photo, one validation photo and one test photo after
  conservative source filtering. Pomegranate also has only one test photo.
  A high or low AP for these classes does not establish generalization.
- Class/source imbalance is severe. Rare-class AP must be reported separately.
- Open Images includes cut fruit and prepared-food scenes. Detector outputs from
  such photos must not automatically be treated as valid freshness assessments.
- No independent phone-photo dataset, unsupported-fruit test, field study, battery
  benchmark, or Android latency measurement has been completed.
- The source annotation convention and image provenance have limitations listed
  in README.md and ../../datasets/fruit13/SOURCES.md.
- Broader or more independent evaluation, and more mango/pomegranate data, are
  required before product-readiness claims.

## Artifacts and export

| File | Bytes | SHA256 |
|---|---:|---|
{files}

artifacts/model.json specifies preprocessing, class IDs, and raw output layout.
TorchScript parity passed on {parity['images']} held-out sample images; maximum
absolute raw-output difference: {parity['max_absolute_raw_output_difference']:.6g}.

Desktop CPU raw-forward median: {parity['CPU_raw_forward_median_ms']:.2f} ms;
p95: {parity['CPU_raw_forward_p95_ms']:.2f} ms, with four CPU threads. This excludes
decode, preprocessing, NMS, classifier work, and rendering. It is NOT Android or
tap-to-result latency. CUDA validation timings are recorded separately in evaluation JSON.

## License and later expansion

Retain Ultralytics' upstream software/checkpoint license terms and all dataset
attribution. The source GitHub image provenance remains unresolved, so no public
redistribution clearance is claimed. This run performs no public upload.

For thirteen fruits, add reviewed guava/jujube/lulo/tamarillo boxes and retain data
for all original nine classes. Train a separate thirteen-class head/checkpoint;
do not overwrite this baseline or assume its class IDs equal the thirteen-class
taxonomy. Existing Fruit Lens Classic and the MobileNetV3 freshness model remain separate.
'''
    (RUN/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    for directory,description in [('runs','Framework training and held-out test outputs.'),('runs/train','Training curves, arguments and validation-selected weights.'),('runs/train/weights','best.pt is validation-selected. last.pt supports resuming this exact experiment.'),('runs/test','Held-out test metrics, confusion matrices, precision-recall curves and plotted batches.'),('runtime_config','Experiment-local library settings and caches.')]:
        folder=RUN/directory
        if folder.exists():(folder/'README.md').write_text(description+'\n',encoding='utf-8')
    (RUN/'reports/COMPLETION.json').write_text(json.dumps({'training_completed':True,'heldout_evaluation_completed':True,
        'torchscript_parity_passed':True,'android_integrated':False,'production_validated':False,
        'free_D_bytes':shutil.disk_usage(RUN).free},indent=2),encoding='utf-8')
    print('Final model card and completion record written.',flush=True)


if __name__=='__main__':main()
