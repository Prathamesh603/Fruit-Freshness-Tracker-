# Storage state — 1 November 2026

The original 30-category ZIP was removed as a redundant download. The active nine-fruit subset (9,640 photos), selected original label text, quarantined conflicts and measured reports remain on disk.

To restore the exact ZIP without rebuilding or overwriting the prepared subset:

```powershell
python datasets/zenodo-fruit9-v1/prepare_dataset.py --download-only
```

The download URL, bytes, SHA256 and publisher MD5 are saved in `reports/download.json` and [the cleanup restore record](../storage-cleanup-2026-11-01/README.md). Training has not started on this imbalanced subset.
