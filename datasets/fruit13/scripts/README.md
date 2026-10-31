# Reproduce data preparation

download_sources.py downloads source archives and annotation metadata.
prepare_collection.py safely extracts, hashes, groups, and inventories source images.
collect_openimages.py downloads the bounded detection subset and keeps attribution.
document_collection.py writes folder guides, coverage tables and annotation queues.
verify_collection.py checks inventories against actual files and expected coverage.
requirements.txt lists Python dependencies. See ../HANDOFF.md for command order.

Downloads reuse existing files. Extraction detects changed raw files instead of
overwriting them. The scripts regenerate machine inventories/reports; they do not
modify review/decisions.csv or train any model. Do not edit a raw file in place.
