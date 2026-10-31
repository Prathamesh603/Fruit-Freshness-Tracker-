# Fruit Lens app editions

Each edition has its own folder, launcher name, Android application ID, model assets, APK, and documentation. Installing a later edition must not replace Classic.

| Order | Edition | Purpose | Launcher name | Application ID | Status |
|---|---|---|---|---|---|
| 1 | [Classic — Pretrained CPU](01-fruit-lens-classic/README.md) | Preserved existing-model baseline | Fruit Lens Classic | `com.fruitlens.classic` | Built separately from the original app |
| 2 | Custom | Our own fine-tuned classifier and subsequent improvements | Fruit Lens Custom | `com.fruitlens.custom` | Reserved; not integrated or built by this preservation task |

The original already-installed prototype is also retained inside Classic's `apk/original/` directory. Its application ID remains `com.fruitlens.app`. The renamed Classic APK uses a different ID, so it can be installed alongside both that original prototype and a future Custom app.

## Rules for future editions

1. Preserve `01-fruit-lens-classic/`; do not replace its models with trained exports.
2. Create `02-fruit-lens-custom/` for the custom-model app. Use a different launcher name and application ID.
3. Keep the Kotlin/native namespace unchanged if reusing the current JNI entry points, or update JNI names and test them deliberately. Application ID and source-code namespace are different concepts.
4. Record exact model hashes, preprocessing, labels, execution hardware/backend, build identity, and evaluation evidence for every edition.
5. Compare editions using the same original input images and matching measurement boundaries. Do not assume a trained replacement is more accurate or faster until measured.
6. Use separate APK names, for example `Fruit-Lens-Classic-CPU-v1-arm64.apk` and `Fruit-Lens-Custom-v1-arm64.apk`.

Training work remains in its existing directories. A training export does not automatically change an archived app.
