# Public YOLO11m reference benchmark

Source: https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector
Revision: `82c9865261175e778ba5641cf8560b5d10439ec6`. SHA256: `499ccee2169731363de26146d9989942c27f3849f64509e0b5190cbd1c679c19`.
Original file: 40,565,868 bytes. Actual checkpoint has 35 classes.
It was loaded with restricted globals and its state tensors transferred strictly
to an architecture constructed by the installed Ultralytics library.

## Scope and results

Evaluated on the existing frozen 483-image test pool at input size 416, matching
the first experiment's size and metric settings. This pool is held out from our
nine-fruit training, but overlap with the reference's training data is UNKNOWN.
It is not a new independent evaluation set. Mango and pomegranate each have only
one test photo; their AP cannot establish generalization.

The reference covers eight targets and has no lemon class. Green_Orange and
Orange scores are merged before NMS. Unsupported class predictions are excluded
from this target-scope benchmark. Original 35-class names are retained in metadata.

| Fruit | Test images | Reference AP50 | Reference AP50-95 |
|---|---:|---:|---:|
| apple | 219 | 76.0% | 70.7% |
| banana | 267 | 96.2% | 70.6% |
| grape | 2 | 41.3% | 24.6% |
| lemon | 8 | 0.0% | 0.0% |
| mango | 1 | 0.0% | 0.0% |
| orange | 240 | 86.3% | 83.5% |
| pomegranate | 1 | 0.0% | 0.0% |
| strawberry | 6 | 28.8% | 19.4% |
| tomato | 7 | 31.6% | 18.4% |

For the eight shared classes, reference mAP50 is 45.02%
and mAP50-95 is 35.89%.
Our trained YOLOv8n scores on the same eight-class scope are
59.55% and
48.66% respectively.
These are descriptive results, not proof of model superiority on independent phones.

Timings in comparison.json are desktop GPU validation measurements. They do not
establish Android performance. No app weights, completed run, or raw source image
was replaced. This model detects identities, not freshness.

## Bohni dataset access

The exact v13 page returned HTTP403 and the normal public API returned HTTP401.
An authorized Roboflow export or local ZIP is needed before image-level auditing.
No dataset images were downloaded in this step. Keep the dataset export separate
until provenance, counts, boxes, augmentation lineage and overlap have been checked.

Follow-up: the user has now supplied the ZIP. See ../../datasets/bohni-v13/reports/STATUS.md
for the completed archive audit and compact mapped subset. The access errors above
are historical; no further ZIP is requested.
