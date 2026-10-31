# Instructions for the next agent

## What has been done

Original classification images were collected for 13 target fruits. A boxed
three-fruit dataset and a bounded Open Images subset were gathered separately.
Every image inventory records origin and hashes. See `reports/STATUS.md` for
measured counts and failures, rather than relying on a publisher's headline.

No new model was trained, no final split was frozen, and no model was installed in
Classic or Custom. Existing `ml/` and `model_exports/` outputs remain unchanged.

## Reproduce the collection

Run from `D:\Projects\Fruit freshness detection` with Python 3.10 or newer.
Install the packages listed in `scripts/requirements.txt` in an isolated environment
if they are not already available. The scripts use Pillow, NumPy and SciPy; the
download step otherwise uses Python's standard library.

```powershell
python datasets/fruit13/scripts/download_sources.py
python datasets/fruit13/scripts/prepare_collection.py
python datasets/fruit13/scripts/collect_openimages.py
python datasets/fruit13/scripts/document_collection.py
python datasets/fruit13/scripts/verify_collection.py
```

Download commands need Internet access. Successfully downloaded files are reused;
failed images can be retried by rerunning the collector. The extraction script
checks existing images against ZIP CRCs rather than replacing changed files.
The Open Images collector checks readable images and retained source licenses.
The original eight-fruit ZIP is intentionally reused from the older training
folder and must exist at the path listed in `SOURCES.md`.

The classification audit uses JPEG draft decoding to keep the inventory fast.
It checks readability, original dimensions, exact hashes, and perceptual groups;
it is not exhaustive full-resolution visual inspection. Raw images are unchanged.

## Required work before training

1. Read the review guide and resolve conflicting/ambiguous labels.
2. Establish physical-fruit/session groups where possible. Review residual leakage.
3. Draw/check boxes for the missing four fruit types. Add many more independent
   scenes for underrepresented types and review existing boxes.
4. Decide whether the GitHub boxed image provenance is adequate for the intended
   use. Keep it out of a redistributable release until this is established.
5. Deduplicate across detection/classification sources and any derived crops.
6. Freeze group-aware splits and record per-fruit, per-condition counts.
7. Train a 13-class detector and a 26-class freshness classifier as separate
   experiments. Record architecture, preprocessing, labels, parameters, size,
   precision, optimizer, seed, metrics, and model hashes.
8. Evaluate per-fruit detection AP and freshness errors on held-out data. Track
   spoiled-as-fresh mistakes, abstentions, unknown inputs, and phone conditions.
9. Benchmark the complete Android pipeline on real phones; desktop inference
   time is not Android latency. Retain Classic as the unchanged comparison.

The requested 13-fruit collection is a starting scope, not all fruits in the world.
Do not infer that a universal fresh/spoiled classifier has been validated for an
unseen fruit. Additional fruit types need labels, boxes, and evaluation of their own.
