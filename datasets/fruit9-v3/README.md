# New nine-fruit detection data

Target IDs: 0 apple, 1 banana, 2 grape, 3 lemon, 4 mango, 5 orange, 6 pomegranate, 7 strawberry, 8 tomato.

| Location | Purpose |
|---|---|
| `links.csv` | Dataset links received from the user, selected versions, format and review status |
| `incoming/` | New source ZIPs with original images and bounding-box annotations; not tracked by Git |
| `prepared/` | Reviewed images and mapped YOLO labels; generated only after audit, not tracked by Git |
| `reports/` | Source counts, SHA256 hashes, label checks, duplicate checks and final split manifest |

No new links have been supplied yet. No data has been declared training-ready. Prefer unaugmented originals, preserve source class maps, verify boxes, and group related captures/augmentations before splitting. A classification-only dataset is not an object-detection dataset. Keep the phone evaluation set out of training.

Old useful sources remain separately documented under `datasets/fruit13`, `datasets/bohni-v13`, and `datasets/zenodo-fruit9-v1`. Read their latest storage-state notes before assuming an archive is still on disk. Do not merge everything merely to increase counts.
