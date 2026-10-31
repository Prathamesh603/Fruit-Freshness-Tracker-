# New detection-source search

This folder records the follow-up search on Roboflow Universe, Mendeley Data,
GitHub, Hugging Face and Zenodo on 2026-10-30. These are candidate sources, not
newly accepted training data. Read ../../reports/DETECTION_SOURCE_SEARCH.md.

mango_mendeley.json contains the publisher's download metadata for a mango RAR
archive. The archive itself has not been downloaded. tamarillo_github.json records
the research repository tree and commit. tamarillo_repository_attachment.zip was
downloaded and inspected as an archive: it contains three Excel result files and
one pickle file, with no JPEG images or bounding-box text files. The pickle has
not been loaded or executed. This attachment is not the tamarillo image dataset.

Roboflow export pages and Zenodo returned HTTP 403 to the command-line client.
Public indexed project pages were available for research. Export archives, counts,
box geometry and source permissions still need inspection after normal authorized
download. Do not count these candidates in the existing local image inventories.
