# Official starting weights

`yolo26s.pt` is the official COCO-pretrained detection checkpoint downloaded from Ultralytics, not an already fine-tuned nine-fruit model. Its size and SHA256 match the publisher release metadata. See `provenance.json` for exact identity and verification boundaries.

The `.pt` binary is excluded from Git by the existing training-data policy. To restore it, download the recorded source URL and verify its size and SHA256. Keep this source checkpoint separate from any future trained `best.pt`/`last.pt` outputs.
