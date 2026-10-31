# Environment report

Recorded 2026-10-30 for the reproducible training and export environment.

## Host

| Component | Value |
|---|---|
| OS | Windows 11 Home Single Language, version 10.0.26300, build 26300 |
| Machine | Lenovo 82K1 |
| CPU | 11th Gen Intel Core i5-11300H @ 3.10 GHz |
| CPU topology | 4 physical cores, 8 logical processors |
| RAM | 15.79 GiB usable |
| Integrated GPU | Intel Iris Xe Graphics, driver 31.0.101.4502 |
| Discrete GPU | NVIDIA GeForce GTX 1650, 4 GiB, driver 31.0.15.4633 (NVIDIA 546.33) |
| Free disk at completion | C: 4.17 GiB; D: 8.90 GiB |

The project lives on drive D:. TensorFlow checkpoints, the raw archives, and the training environment consume most of the free space.

## Python and ML packages

| Component | Version |
|---|---|
| Python | 3.11.9 (64-bit, MSC v.1938) |
| TensorFlow CPU | 2.21.0 |
| Keras | 3.15.1 |
| NumPy | 2.4.6 |
| Pillow | 13.0.0 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.1 |
| ai-edge-litert | 2.3.0 |
| pytest | 9.2.0 |

Full transitive pinning is in `ml/requirements-lock.txt`; direct dependencies are in `ml/requirements.txt`.

## Accelerator status

TensorFlow reports only:

```text
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
```

Native Windows GPU support is unavailable for TensorFlow versions at or above 2.11. The GTX 1650 was therefore not used. Training was CPU-only and completed without requiring a resume:

- Frozen-backbone run: 130.89 seconds.
- Fine-tuning run: 44.67 seconds.
- TFLite conversion and final bundle assembly were not separately timed.

No WSL2 or TensorFlow-DirectML path was used because the CPU runs completed quickly and changing the environment would weaken reproducibility.

## Benchmark caveat

`ml/reports/benchmark_mv3s_finetune_224_v1.json` was measured on this Windows host with two LiteRT CPU threads:

- Inference only: mean 1.492 ms, p50 1.287 ms, p95 2.824 ms.
- Desktop PIL preprocess plus inference: mean 43.867 ms, p50 36.590 ms, p95 108.031 ms.

These figures are not Android measurements. No phone or emulator benchmark was performed.

## Reproduce the environment

```powershell
py -3.11 -m venv ml\.venv
ml\.venv\Scripts\python.exe -m pip install --upgrade pip
ml\.venv\Scripts\python.exe -m pip install -r ml\requirements-lock.txt
ml\.venv\Scripts\python.exe -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
```

`requirements-lock.txt` was produced from `pip freeze` inside the working environment.
