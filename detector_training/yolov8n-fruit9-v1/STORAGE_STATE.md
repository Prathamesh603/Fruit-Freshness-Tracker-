# Storage state — 1 November 2026

Generated train/validation image and label copies were removed. Frozen test images/labels, original manifests, preparation/training/evaluation code, logs, best/last checkpoints and model exports remain.

Rebuild old data only when needed using `prepare_data.py` and retained raw sources. Preserve the recorded split manifest before rerunning preparation: reproduce the old membership rather than silently replacing the benchmark. See [cleanup and restore records](../../datasets/storage-cleanup-2026-11-01/README.md).
