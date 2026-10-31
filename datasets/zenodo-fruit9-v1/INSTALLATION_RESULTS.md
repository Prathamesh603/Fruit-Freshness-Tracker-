# Installed dataset results — 31 October 2026

The publisher checksum passed. The active nine-fruit subset contains **9,640 photos and 100,794 boxes**. The source splits are retained: 7,565 training, 1,329 validation, 746 test photos. These are staging splits, not verified independent evaluation data.

| Fruit | Train photos | Validation photos | Test photos | Training boxes |
|---|---:|---:|---:|---:|
| apple | 3317 | 493 | 272 | 29352 |
| banana | 14 | 5 | 4 | 26 |
| grape | 81 | 19 | 9 | 484 |
| lemon | 215 | 58 | 25 | 1242 |
| mango | 149 | 31 | 28 | 724 |
| orange | 2001 | 437 | 210 | 36598 |
| pomegranate | 3 | 3 | 4 | 22 |
| strawberry | 43 | 6 | 3 | 74 |
| tomato | 1748 | 279 | 192 | 9875 |

## Important change to the initial recommendation

Metadata established that all nine names existed; actual extraction shows that this collection is **not balanced enough to use alone for our nine fruits**. It is useful mainly as supplementary detection data. In particular, banana, strawberry and pomegranate need stronger data coverage. Existing useful nine-fruit data and the freshness photos were retained during cleanup.

The installer removed 3,445 repeated image copies and rejected 134 source label files with invalid geometry/format. Among retained duplicate groups, **627 photos had conflicting target annotations**; those photos and their provisional labels were moved out of the active splits into `quarantine/conflicting-annotations/`. There were 865 conflict events across the duplicate copies, not 865 unique quarantined photos.

No training was started. More source-family/perceptual duplicate checks and visual annotation review are needed before a trustworthy detector experiment. This source has fruit-identity boxes and cannot supply freshness targets for classifier training.

See `reports/installation-verification.json`, `reports/class-coverage.csv`, and `reports/conflict-quarantine.json` for measured evidence. The source raw row count was 218,004, compared with 222,301 annotations claimed on the publication page. This discrepancy remains unresolved; raw rows are not automatically valid boxes. Free space at verification was approximately 6.56 GB in decimal units.
