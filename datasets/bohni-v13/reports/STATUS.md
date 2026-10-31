# Bohni v13: measured audit results

## What was received

- User-supplied Roboflow YOLOv8 ZIP: 709,892,684 bytes.
- SHA256: `e53de22bd94867b4c7034506423249ad823d8c3878481485f498e613b2ad4f1c`.
- Actual image/label pairs: **6,730**; source classes: **35**; source boxes: **20,592**.
- All image members passed the JPEG draft readability check and ZIP CRC reads;
  this does not constitute exhaustive full-resolution pixel or semantic review.
- Malformed/out-of-bounds source box records: **0**.
- Source splits: **6,573 train / 147 validation / 10 test images**.
- The exporter README says 6,805 images, but the archive actually contains 6,730. The reason for the difference is not established.
- Publisher metadata declares CC BY 4.0, project bohni-tech/fruits-and-vegi, version 13.

## Useful fruit subset

Prepared **2,330 image/label pairs** with **6,801 boxes**, covering
eight matching fruit identities. This uses **165.3 MiB** of file
content instead of extracting every 1280-pixel image. Original bytes remain in the ZIP.

| Fruit | Source image files | Filename families | Boxes |
|---|---:|---:|---:|
| apple | 427 | 147 | 1232 |
| banana | 277 | 97 | 478 |
| grape | 425 | 145 | 833 |
| lemon | 0 | 0 | 0 |
| mango | 277 | 101 | 1098 |
| orange | 290 | 102 | 893 |
| pomegranate | 343 | 119 | 885 |
| strawberry | 235 | 81 | 791 |
| tomato | 150 | 58 | 591 |

Filename families group augmented views of a source filename. They are NOT proof
of independent physical fruits or capture sessions. Mango and pomegranate coverage
is much larger than our first run, but labels and source independence still matter.

## Problems that must be handled before training

1. **Source test split is unsuitable for our fruit task:** none of its ten images
   has any of the eight matching target fruit classes. Do not report fruit accuracy
   on this split or assume the original YAML provides a useful held-out test.
2. **Augmented data:** the exporter describes three views per source image, with
   random cropping and brightness reduction. There are 2,282
   source-filename families across the whole archive and 802
   among selected fruit files. Keep complete families together in new project splits.
3. **Existing split leakage:** 11 source-filename
   families already cross the publisher's split folders, including ten target-fruit
   families. Rebuild splits after grouping rather than reuse the source YAML.
4. **Duplicates:** 1 exact duplicate group(s), with
   1 additional copy/copies, were found in the archive.
5. **Potential overlap with the previous experiment:** 4 canonical pHash
   candidate pairs involving 3 new images; 1
   source family/families touch its validation/test pool. The candidate table holds
   all members of those families. Perceptual similarity is not identity proof, and
   this check does not resolve all crops/rotations or physical-fruit overlap.
6. **Sample annotation concerns:** the preview contains stock photos, composites,
   cut fruit, possibly ambiguous citrus/berry labels, and apparently unboxed small
   fruit. These need review. Labels were not silently corrected from visual guesses.
   Grapes/banana bunches also need a consistent region-versus-individual counting policy.

After automatic holds, **2,326 candidates** remain and **4
images are held**, with reasons in held_images.csv. No project training split has
been assigned and no second detector has been trained from this source yet.

## Next preparation step

Combine accepted data with the prior nine-class dataset, retain lemon examples
from the prior source, group all source families/near-duplicates across both,
and protect the existing test pool. Build adequate new per-fruit validation/test
coverage using source groups; use one representative view per augmented family
for headline held-out support counts. Then train a separately named version 2.

Guava, jujube, lulo and tamarillo remain absent. This dataset does not label
freshness. Existing Classic, the first detector, and the freshness model are unchanged.
