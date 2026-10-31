# Download and preparation evidence

`status.json` shows the current stage. While downloading it records received and expected bytes. When preparation finishes it records the measured subset counts and remaining quality checks.

`download.json` contains the original URL, byte size, MD5 and SHA256. `class-map.json` maps the source classes to our nine fruits. `images.csv` connects each extracted photo and label to their original archive members. `annotation-audit.json` records rejected rows, exact duplicates and source class counts. `summary.json` gives final images, boxes and split/class totals.

These reports describe detection data, not freshness labels or measured model accuracy. Original source splits are retained for staging; perceptual/source-family overlap and visual annotations must be reviewed before trusting a training benchmark.
