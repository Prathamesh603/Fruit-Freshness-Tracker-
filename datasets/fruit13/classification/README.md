# Freshness classification inventory

images.csv lists each original source photo once, with its fruit and condition,
source folder, license, original dimensions, SHA-256, pHash, candidate duplicate
group, baseline overlap, and review status. Paths are relative to fruit13/.

`candidates.csv` withholds exact duplicate copies and candidate conflicting groups.
`../review/held_images.csv` explains every hold; raw files remain available. Candidates
still need human review and are not an approved training split.

All project splits are unassigned. source labels are not yet fully reviewed.
exact_duplicate marks an identical byte copy. group_has_label_conflict marks
candidate groups with different labels; it does not automatically correct them.
This table contains full source frames, not guaranteed tight single-fruit crops.
Review composition and create derived crops only after boxes and splits are settled.
