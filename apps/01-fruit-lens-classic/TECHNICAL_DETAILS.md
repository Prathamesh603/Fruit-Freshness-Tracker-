# Classic edition — technical record

## 1. What the app does

Fruit Lens Classic is a native Android, offline, still-photo fruit scanner. It detects supported fruit regions, crops them from the decoded photo, and classifies the visible appearance of each crop. The live camera preview is continuous; model inference runs when a photo is captured or selected. It is not a continuous video detector or tracking system.

The original prototype's application ID is `com.fruitlens.app`. This preserved named edition uses `com.fruitlens.classic` and launcher name **Fruit Lens Classic**. Source namespace `com.fruitlens.app` is deliberately retained so the existing C++ JNI entry-point names remain valid. The future custom-model edition reserves ID `com.fruitlens.custom` and a different folder/name.

## 2. Technology stack

| Layer | Technology and role |
|---|---|
| Android application/UI | Kotlin, Jetpack Compose, Material 3 |
| Camera | CameraX 1.5.1, Camera2 backend, PreviewView, ImageCapture |
| Native detection | C++17 JNI library `fruitvision`, NCNN 20261020 |
| Classification | LiteRT 1.4.2 CPU Interpreter, XNNPACK requested |
| Image decoding/orientation | Android BitmapFactory, Matrix, ExifInterface 1.4.1 |
| Thread coordination | One dedicated Java executor worker for model initialization/inference/cleanup |
| Build | Gradle 8.13, Android Gradle plugin 8.13.2, Kotlin 2.2.21, JDK 17 |
| Android native build | NDK 28.2.13676358, CMake 3.22.1 |
| Android versions | Compile/target API 36; minimum API 26 |
| Packaging | Separate ABI APKs; optimized `internal` build uses R8 and resource shrinking |

Python is used only for development, model inspection, bundle validation, and build preparation. Python, training libraries, and datasets are not shipped inside the phone app. There is no OpenCV dependency, server, login, or runtime model download. The merged APK has camera permission and no Internet permission.

## 3. Detection model

**Model:** NanoDet-Plus-m, official existing 416-pixel NCNN export.

* Source: https://github.com/RangiLyu/nanodet
* Export: https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_ncnn.zip
* Model files: `source/android/app/src/main/assets/detector/nanodet.param` and `nanodet.bin`.
* Runtime and archive provenance: adjacent `provenance.json` and license texts.
* This baseline does not retrain the detector and does not substitute the older NanoDet-m checkpoint.

The model is a general COCO detector with 80 class scores; the app retains COCO zero-based classes **46 banana, 47 apple, 49 orange**. Other object types do not receive a freshness classification through this detector path.

### Preprocessing

The decoded Android bitmap is RGBA. NCNN converts it to BGR and resizes with preserved aspect ratio, padding to **416 × 416**. BGR channel means are `[103.53, 116.28, 123.675]`; normalization multipliers are `[1/57.375, 1/57.12, 1/58.395]`.

### Output processing

The parser expects `data` input and `output` output. The 416 model outputs **3,598 candidate rows**, each with 80 class scores and 32 box-regression values. Strides are 8, 16, 32, and 64. Four sides use eight-bin distribution decoding with softmax and expected distance.

The app applies a **0.4 detection score threshold**, keeps supported class predictions, maps coordinates back through the letterbox transform, clamps boxes, and rejects very small/invalid boxes. It sorts by score and applies class-aware NMS at **IoU 0.5**, retaining at most **five regions**.

This is region detection, not reliable fruit counting or spoilage segmentation. A banana bunch may become one region; a pile of oranges may receive a large group box plus overlapping individual boxes. Class-aware IoU NMS does not eliminate every contained or partially overlapping box. This behavior is preserved for comparison.

## 4. Freshness classifier

**Model:** existing Smart Fruit MobileNetV2-based classifier, not our newly trained model.

* Repository: https://github.com/JosephMoustaid/smart-fruit-models-notebooks
* Pinned source commit: `200d8dffb48679567777a6ab5adc1d87eeea1c88`.
* Original filename: `ripeness_model.tflite`.
* App filename: `source/android/app/src/main/assets/model/freshness.tflite`.
* Size: **2,673,592 bytes**, approximately 2.55 MiB.
* SHA-256: `e09307ca0aacdfc83666ca8ff25c02659d6323a1372409b390561619778b10a4`.
* Executable IO shapes were inspected: `[1,96,96,3]` to `[1,9]`, FLOAT32 input and output.

The author describes a frozen ImageNet-pretrained MobileNetV2 backbone with global average pooling, a 128-unit ReLU dense layer, dropout, and a nine-output softmax head. The notebook exports with default post-training optimization. This does not mean every operation executes in INT8 or that the app uses integer IO. The input/output tensors are FLOAT32; constant-tensor precision can be inspected in the bundled graph.

Inspection of the preserved graph found 125 FLOAT32 tensors, one INT32 tensor, and 50 INT8 tensors. Among tensors with nonempty constant buffers, 58 are FLOAT32, one is INT32, and 50 are INT8. This confirms mixed stored tensor types; it does not establish a single arithmetic precision for all runtime operations.

### Ordered classes

| Index | App label | Fruit | Condition category |
|---|---|---|---|
| 0 | `fresh_apple` | Apple | Fresh appearance |
| 1 | `fresh_banana` | Banana | Fresh appearance |
| 2 | `fresh_orange` | Orange | Fresh appearance |
| 3 | `rotten_apple` | Apple | Rotten appearance |
| 4 | `rotten_banana` | Banana | Rotten appearance |
| 5 | `rotten_orange` | Orange | Rotten appearance |
| 6 | `unripe_apple` | Apple | Unripe appearance |
| 7 | `unripe_banana` | Banana | Unripe appearance |
| 8 | `unripe_orange` | Orange | Unripe appearance |

The model has no separate overripe class. A dark banana peel does not establish the condition of its flesh. Labels are learned dataset categories; the app does not measure internal freshness, bacteria, chemical contamination, remaining shelf life, or food safety.

### Crop and input processing

Each detected box is cropped from the higher-resolution decoded photo, with about **6% padding per side** and image-boundary clipping. Crops under **24 pixels** in either dimension are skipped. Classifier input is stretched to **96 × 96**, using **nearest-neighbor resize** to match the reference notebook's directory-loader default. Input remains RGB and is normalized as `pixel / 255`, equivalent to manifest means `[0,0,0]`, standard deviations `[255,255,255]`.

Labels, normalization, dimensions, interpolation, output interpretation, and threshold metadata are in the bundle's `model.json`. Do not substitute ImageNet `[-1,1]` preprocessing for this checkpoint merely because its backbone is MobileNetV2.

### Prediction interpretation

The app interprets this checkpoint's output as probabilities and selects its highest score. It does not apply a second softmax. It checks agreement between the classifier's fruit type and the detector's fruit type.

The manifest's `minScore` is **0.8**, but `thresholdValidated` is **false**. Consequently, this reference's results are marked experimental; the app may display a top-class suggestion even below 0.8. The existing UI labels "No visible spoilage" and "Visible spoilage" are mappings of fresh/rotten categories, not certified observations. The disclaimer and exact baseline wording are preserved rather than retroactively changing the demonstrated app.

A model score of 0.980 is not "98% fresh". A displayed 1.000 is rounded model output, not proof of certainty. This model has not been independently calibrated or evaluated on a representative phone-photo holdout.

## 5. CPU versus GPU: precise execution record

**Both neural networks execute on the CPU in this edition.**

* NCNN is the non-Vulkan Android build. `net.opt.use_vulkan_compute = false` and `net.opt.num_threads = 2`.
* NCNN FP16 storage is enabled. This is a runtime/storage option, not a guarantee that every operation uses FP16 arithmetic on every CPU.
* LiteRT uses `Interpreter.Options().setNumThreads(2).setUseXNNPACK(true)`; there is no GPU, NNAPI, or vendor-NPU delegate.
* One worker orchestrates detection followed by crop classifications sequentially. "Two threads per runtime" does not mean both models run simultaneously on four dedicated cores.
* Android may still use the GPU normally to display the camera preview, draw Compose UI, and compose surfaces. **CPU-only inference does not mean the entire phone never uses its GPU.**

No Vulkan compute, Metal, CUDA, QNN/Hexagon, MediaTek APU/NeuroPilot, or cloud inference is used by this baseline. The app does not require GPU/NPU support to scan.

## 6. Processing pipeline

```text
CameraX photo / selected photo URI
  -> background bitmap decode and EXIF orientation correction
  -> decoded image kept to a bounded size (long side at most about 1600 px)
  -> NCNN NanoDet-Plus-m detection at 416 x 416
  -> supported-class filtering and NMS
  -> up to five padded crops from the decoded image
  -> LiteRT classifier at 96 x 96 for each retained crop
  -> score interpretation and fruit-type agreement check
  -> annotated still-photo preview and individual result cards
```

The green boxes locate fruit regions. They do not locate rot, mold, or damaged pixels. The detector cannot reliably count individual fruit in every bunch/pile. No tracking ID or motion model is used.

## 7. Techniques that make this edition small and responsive

| Technique actually implemented | What it does |
|---|---|
| Compact detector and 96-pixel classifier | Limits neural-network work per photo/crop |
| Native NCNN detector | Uses optimized native mobile CPU kernels |
| LiteRT/XNNPACK classifier path | Requests optimized CPU execution |
| Models initialized once | Avoids reopening weights for every scan |
| Warmup before enabling scans | Pays lazy first-execution kernel/allocation setup during preparation |
| One background inference worker | Keeps model work away from the main UI thread |
| Classifier buffer reuse | Reuses allocated input/output byte buffers |
| Region-of-interest classification | Processes detected crops rather than asking freshness questions about the full scene |
| Five-region cap and small-crop rejection | Bounds worst-case classification workload |
| Bounded image decode | Limits large source-photo memory use |
| Explicit lifecycle cleanup | Closes engines after queued inference and releases temporary bitmaps/files |
| R8/resource shrinking and ABI splits | Reduces the installation package; no universal multi-ABI APK is needed for the phone |
| No OpenCV/Python runtime/server SDK | Avoids unnecessary runtime dependencies |

This baseline does **not** implement LRU result caching, transformer KV caching, temporal classification caching, object tracking, custom zero-copy hardware-buffer import, custom persistent GPU pipeline caching, automatic backend benchmarking, or thermal-aware adaptive scheduling. Model/session reuse is not the same as caching predictions for similar-looking fruits.

## 8. Timing and evidence

The user supplied six screenshots of the original APK running on an Android phone. Reported processing times, in screenshot order, were **68, 40, 65, 101, 74, and 50 ms**. Minimum: **40 ms**; maximum: **101 ms**; arithmetic mean: **66.33 ms**; median: **66.5 ms**. Region counts were 1, 1, 2, 5, 2, and 1.

These are user-reported app readouts from a small convenience sample, not a controlled benchmark. The device model is not visible in those screenshots; the project originally targets Nord 2T. Several inputs are photographs of fruit pictures displayed on another screen, which introduces resizing, reflections, and screen patterns.

The app timer starts **after bitmap decode/orientation** and covers detector preprocessing/inference/postprocessing, crop preparation, and crop classifications. It excludes photo capture, file/image decoding, and final UI presentation. Do not call these numbers tap-to-result latency, sustained camera FPS, model-only latency, or energy measurements.

Existing automated evidence includes six Python bundle checks, five JVM numerical checks, and four Android emulator integration checks. Those execute models and exercise camera/bitmap/result paths; they are not a representative accuracy evaluation. See the copied `source/android/verification/` evidence for the original implementation. Classic's separate identity build has its own build/identity/hash checks recorded in `VERSION.json`.

The original APK SHA-256 is `dd9d75a84acf8ff8edf3dc500388bd8f6a49776d337ae8e0c92e229578ba52c9`. The Classic APK has a different package identity and therefore a different APK hash. Model byte identity is verified separately so rebranding cannot be mistaken for model retraining.

## 9. Rebuilding without modifying the baseline models

From `source/android/`, use the copied Gradle wrapper and the pinned toolchain. The included preparation scripts can provision the same official NCNN runtime and Android SDK locally if building on a new machine. Keep the already archived model assets intact; do not run a new-model installer into Classic.

On the current shared machine, the existing toolchain can be reused without copying gigabytes of SDK/cache files:

```powershell
$env:GRADLE_USER_HOME = 'D:\Projects\Fruit freshness detection\android\.tools\gradle-home'
$env:FRUITLENS_NCNN_ROOT = 'D:\Projects\Fruit freshness detection\android\.tools\ncnn\ncnn-20261020-android'
& 'D:\Projects\Fruit freshness detection\android\.tools\gradle-8.13\bin\gradle.bat' --offline :app:testDebugUnitTest :app:assembleInternal
```

Set `local.properties` to the installed SDK, or run the copied bootstrap. That machine-specific file is not part of the portable snapshot. `internal` builds use the local development signing key. There is no production signing configuration or claim of store release readiness.

## 10. Limitations and future comparison

The freshness model's real-world accuracy, unknown-input rejection, and confidence calibration are not established. Existing reference-weight redistribution rights remain unresolved; license texts for NCNN/NanoDet and source provenance must remain attached. Public model-containing distribution requires appropriate clearance or a cleared replacement.

Future fine-tuning belongs in **Fruit Lens Custom**, never this preserved edition. Compare both editions on the same held-out photos with independent labels. Report per-fruit errors and false-fresh predictions, accepted coverage, accuracy/F1, processing boundaries, memory, and energy where measured. A faster readout or higher softmax score alone does not prove a more accurate model.
