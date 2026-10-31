# Demonstrating Classic first

## Suggested introduction

"This is Fruit Lens Classic, our baseline Android app using existing pretrained fruit models. It detects apples, bananas, and oranges, then predicts their visible condition. Neural-network inference runs offline on the phone's CPU. The detector uses NCNN and the classifier uses LiteRT."

## Demonstration order

1. Open **Fruit Lens Classic**, showing the separate app name.
2. Capture a clear single supported fruit, or select a photo.
3. Explain that the green rectangle is a detected fruit region, not a map of damaged pixels.
4. Show the condition suggestion and explain that its score is a model output, not a percentage of freshness.
5. Show an unripe or visibly deteriorated example with a known label; discuss any mistakes rather than hiding them.
6. Show a multi-fruit scene, explaining that bunches and overlapping boxes can make counts imperfect.
7. Show an empty/unsupported scene and the unable-to-assess response.
8. Explain that the timing readout excludes capture, image decoding, and UI rendering.

## How to describe current speed

"In six user screenshots, the app reported 40–101 ms processing, averaging about 66 ms. This is encouraging initial execution evidence, not a formal phone benchmark or accuracy score."

The original measurements came from the original Fruit Lens APK, preserved alongside Classic. Classic has the same model files and inference settings; its renamed app identity is the change. Do not invent a new Classic benchmark without actually running it.

## Introducing a later custom-model edition

"Fruit Lens Custom is a separate app using our own trained classifier. We kept Classic unchanged so we can compare both on the same images."

Only say the custom model is better if the comparison supports it. Use this table after both apps have been evaluated:

| Measure | Classic | Custom |
|---|---|---|
| Model/checkpoint hashes | Recorded in this folder | Fill after integration |
| Fruit/condition labels | Nine combined classes | Record actual labels |
| Backend | CPU / NCNN + LiteRT | Record actual backend |
| Same-device processing latency | Measure on a fixed image set | Same set/procedure |
| Held-out accuracy / macro-F1 | Not yet independently measured | Measure independently |
| Spoiled examples predicted fresh | Not yet independently measured | Measure by fruit |
| Memory/energy | Unmeasured | Measure if making claims |
| App package size | See VERSION.json | Measure built APK |

## Practical presentation notes

* Classic and Custom should have different launcher names and application IDs, allowing side-by-side installation.
* Keep original test images unchanged between app comparisons; direct photo imports avoid screen-photo artifacts where possible.
* The existing UI's freshness terms are experimental model categories. A photo cannot confirm internal condition or food safety.
* This preserved build is for local demonstration/testing; unresolved reference-weight redistribution terms are documented in the technical record.
