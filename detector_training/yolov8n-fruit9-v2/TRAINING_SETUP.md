# Reproduce detector v2

Read existing MODEL_CARD.md for results and the regression on the unchanged v1
test. Existing README and weights are preserved.

Environment: Python3.10.11,PyTorch2.5.1+cu121,torchvision0.20.1+cu121,
Ultralytics8.3.40,GTX1650 CUDA. The local executable is recorded in
runtime_environment.json. Build a matching CUDA environment on another machine;
the recorded temporary path is machine-specific.

`run_pipeline.py` prepares grouped data, trains, evaluates, exports TorchScript,
checks hashes/parity and compares against the frozen v1test. `prepare_data.py`
combines v1prepared data with Bohni's matching8fruits, retaining lemon from v1.
Filename families and rotation/reflection pHash groups stay together. New groups
touching frozen holdouts are excluded. New held-out components use one view.
Physical-fruit/session and crop independence are not guaranteed.

Training: original COCO YOLOv8n initialization,416input,batch8,AdamW0.001,
60maximumepochs,patience12,seed42,AMPoff. Actual run43epochs, best31,44.6minutes.
Exact augmentations/optimizer settings: reports/training_config.json. Split:
3,761train,591validation,584test.9classes in order apple,banana,grape,lemon,mango,
orange,pomegranate,strawberry,tomato. Parameters3,012,603before fusion.

Losses: runs/train/results.csv. Logs: logs/train.log. Checkpoints:
runs/train/weights/best.pt,last.pt. Final exports: artifacts/. Separate deployment
conversion: tools/export_custom_detector.py, model registry NCNN folder. A new
training run must use a new version folder; do not overwrite this completed run.
