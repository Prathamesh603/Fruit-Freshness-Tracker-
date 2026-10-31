# Project helpers

`organize_models.py` creates the versioned catalog, copies existing weights with
hash checks, preserves existing READMEs, and stages the exact chosen model assets
inside the separate Custom app. It never replaces Classic. Run with Python3.10+.

`export_custom_detector.py` converts an owned YOLO checkpoint to NCNN in a new
folder using pinned local PyTorch/Ultralytics/PNNX tools. Conversion does not prove
accuracy; compare outputs and run Android tests. Native input/output rules are
recorded in the generated conversion metadata.

`verify_custom_app.py` checks the packaged model bundle, Classic preservation,
APK identity/signature, test evidence and release hashes. The resulting JSON is
execution evidence for a local demo, not a claim of independently validated
camera accuracy.
