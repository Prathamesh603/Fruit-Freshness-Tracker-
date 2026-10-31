# Reproduce the nine-fruit classifier

Existing README and MODEL_CARD remain unchanged. Start at MODEL_CARD for measured
results and limitations; this file explains how training ran.

Python3.11.9,TensorFlowCPU2.21,Keras3.15.1; use the existing `ml/.venv`. Its pinned
dependencies are in `ml/requirements.txt` and `requirements-lock.txt`. Training
does not use the NVIDIA GPU because this native Windows TensorFlow environment
is CPU-only. Four intra-op threads, two inter-op threads, seed42.

Run from project root:

```powershell
ml/.venv/Scripts/python.exe -u classifier_training/mobilenetv3small-fruit9-v1/train_classifier.py
```

The script reads13,354nine-fruit candidates from the tracked inventory, verifies
original image hashes, keeps pHash groups together, and splits9,338train,
2,012validation,2,004test. Original data stays local and untouched. Label-conflict
groups and exact duplicate copies were already held out by collection preparation.
Unknown physical-fruit/session identity remains a limitation.

Model: ImageNet MobileNetV3Small, input160RGB normalized[-1,1], global average
pool, dropout0.2, Dense18softmax. Parameters949,506. Head training uses frozen
features, Adam0.001, up to30epochs, patience5, batch64, inverse-frequency class
weights. Fine tuning selects last40backbone layers while freezing BatchNorm,
Adam0.0001,batch32,12epochs,patience4. Best fine epoch11selected by validation
cross entropy. The head baseline remains a candidate if fine tuning fails to
improve validation loss. Training/preparation took405.6seconds on this computer.

Outputs: logs/head_epochs.csv and fine_epochs.csv, checkpoints/best.keras and
last.keras, reports/test_evaluation.json and per_fruit_test.csv, artifacts/ model
and exact labels/preprocessing/hashes. Export uses SavedModel then built-in-only
TFLite conversion; float32and float16exports were compared against Keras on90
held-out examples. Training arrays were removed after completion to recover
disk space; the full script regenerates them. `--finish` is an export recovery
option only when those arrays exist. Rerunning overwrites this experiment's
outputs: use a NEW version folder for a new experiment.
