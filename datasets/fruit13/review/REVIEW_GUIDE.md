# How to review the data

## First: review fruit and condition labels

Open the sample sheets. They are quick previews, not proof that every image is
correct. Review every conflict group in `label_conflicts.csv`; the group joins
exact copies or perceptually similar pictures. A similarity match is a warning,
not proof of a wrong label. Preserve raw photos and write decisions in a separate
review table with image_id, decision, corrected_fruit, corrected_condition,
reviewer, date, and reason.

Inspect repeated-looking photos together. If possible, identify the same physical
fruit and capture session, including photos taken as it spoiled. Store those IDs.
The automated pHash groups use 64-bit hashes with Hamming distance at most six,
joined transitively. This reduces obvious duplication but cannot establish true
fruit/session independence. `exact_duplicate` means identical SHA-256 bytes.

## Second: draw the missing detection boxes

`missing_detection_boxes.csv` lists source photos for guava, jujube, lulo, and
tamarillo. The blank box fields are an annotation queue, not real annotations.
Use a local tool such as CVAT or Label Studio. No photos have been uploaded to an
external annotation service.

- Draw a tight rectangular box around each visible fruit; include its visible body.
- Label fruit identity only, using the 13 canonical names in `LABELS.md`.
- Annotate every target fruit in the photo, including partly visible ones when
  identity is clear. Mark occlusion/truncation separately.
- Record whether a banana bunch or grape bunch is one region or many individual
  fruits. Choose a consistent policy before training or reporting counts.
- Do not use the whole image as a fake fruit box.
- Do not draw a box around a rotten patch and call it the fruit detector target.
- Keep original width, height, rotation, and pixel coordinates with the box.
- Have another person check difficult images and a sample of ordinary images.

The other nine fruits still need box inspection and more diverse examples. Having
13 mango boxes or 18 pomegranate boxes in this Open Images subset is not enough
to call those categories robust.

## Third: freeze splits, then augment

Deduplicate across ALL sources and both tasks, including any classifier crops
made from detection photos. Keep a physical fruit, session, video, and duplicate
group in one split. Avoid putting related frames into training and test.

After review, create fixed group-aware train/validation/test manifests, retain the
seed and hashes, and inspect per-class counts. Do not generate augmented images
until the split is fixed; augment training only. Do not repeatedly change the test
split after seeing results.

The old three-fruit test data is already known to the project. It cannot be used
as independent evidence that a new model works on unseen phone photos.

## Training gate

Do not mark the collection training-ready until labels, box quality, source
provenance, duplicate/session grouping, and final split checks are recorded.
Initial public-data experiments may be useful, but must be labelled as such and
must not be presented as externally validated app performance.
