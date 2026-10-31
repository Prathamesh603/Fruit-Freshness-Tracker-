# Phone evaluation input folder

No user phone photos were available during this iteration. Real-device and camera-to-result generalization remain unverified.

This folder is reserved for a later, independent phone evaluation set. Do not mix these images into training or threshold selection. If this set is used to choose a model or threshold, it is no longer an unbiased final holdout.

## Collection procedure

1. Recruit an authorized collector and obtain consent for the photos.
2. Capture one fruit per photo with the fruit reasonably centered and fully visible.
3. Use the app's intended rear camera and normal capture path.
4. Record at least 30 images per canonical class, across multiple fruits, sessions, backgrounds, and lighting conditions.
5. Include some difficult negatives: blur, low light, partial occlusion, multiple fruit, and a non-fruit scene. These do not provide a trained unknown class, but they expose closed-set failure modes.
6. Do not use AI-generated images, stock photos, or images downloaded from the training source.
7. Do not include people or private information in the frame.
8. Keep the original files; do not edit or crop them before the first evaluation.

## Suggested layout

```text
ml/data/phone_eval/inbox/
  fresh_apple/
  fresh_banana/
  fresh_orange/
  rotten_apple/
  rotten_banana/
  rotten_orange/
  unsupported/
```

The `unsupported/` folder is for failure analysis only. It is not an evaluation label unless an unknown-class model is trained and validated.

## Label meanings

- `fresh_*`: no visible spoilage in the photographed surface.
- `rotten_*`: visible spoilage in the source dataset's condition taxonomy.
- `unsupported`: no clearly supported single fruit, or the intended label cannot be determined.

Labels describe visible condition, not food safety, internal quality, chemical contamination, or shelf life.

## Metadata

Record at least:

- capture timestamp;
- phone model and camera;
- lighting/background;
- fruit or specimen identifier, if available;
- label and any uncertainty;
- whether the image is a negative/OOD example.

Store metadata separately as CSV/JSONL. Hash every file before evaluation and keep a frozen manifest.

## Release gate

Before using phone metrics in a release claim:

- freeze the collection and labels;
- audit duplicates and near-duplicates against the training manifest;
- evaluate the exact release bundle;
- report accepted coverage, accepted accuracy, false-fresh by fruit, unsupported-input behavior, and Android latency;
- obtain a new holdout if any phone image influences model or threshold choices.
