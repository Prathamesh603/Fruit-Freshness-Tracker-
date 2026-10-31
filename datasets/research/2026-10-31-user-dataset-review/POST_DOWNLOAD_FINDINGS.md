# Findings after the smaller Zenodo archive was downloaded

This follow-up supersedes the metadata-only recommendation where actual data quality is concerned. The initial review established class-name coverage; it did not establish balance or visual annotation quality.

The pinned archive passed its publisher MD5 checksum. Full label-file reading found **218,004 nonempty annotation rows**, rather than the publication's claimed 222,301. The difference is unresolved. All 30 listed classes occur, but the original archive has only **46 banana boxes and 83 strawberry boxes across all splits**, before project filtering. It cannot serve as a balanced standalone nine-fruit training source.

Preparing the nine-fruit subset removed 3,445 exact duplicate copies and rejected 134 source label files for invalid geometry/format. There were 865 duplicate-copy events with conflicting target annotations, affecting 627 selected photos. Those 627 photos and provisional labels were quarantined rather than left in active splits.

The active subset contains **9,640 photos and 100,794 boxes**. Its training split has just 14 banana photos, 43 strawberry photos, and 3 pomegranate photos after conflict quarantine. Apple, orange, and tomato account for about 96.3% of active boxes. Existing useful v1 detection data and freshness photos were therefore retained during cleanup.

Updated recommendation: use this collection only as supplementary detection data after visual/source-family/perceptual checks. Strengthen scarce classes with the existing sources and suitable additional real-world data. Do not start a nine-fruit freshness classifier from these identity-only detection labels.

Detailed table and evidence: `datasets/zenodo-fruit9-v1/INSTALLATION_RESULTS.md`, `reports/class-coverage.csv`, and `reports/installation-verification.json` from the project root. No training or app-model replacement occurred.
