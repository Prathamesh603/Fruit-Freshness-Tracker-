from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "ml" / "data" / "manifests" / "splits_v1.csv"
SUMMARY = PROJECT_ROOT / "ml" / "data" / "manifests" / "split_summary_v1.json"
AUDIT = PROJECT_ROOT / "ml" / "data" / "manifests" / "data_audit.json"

LABELS = [
    "fresh_apple",
    "fresh_banana",
    "fresh_orange",
    "rotten_apple",
    "rotten_banana",
    "rotten_orange",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_manifest_hashes_are_stable() -> None:
    assert sha256_file(MANIFEST) == "47cfa8f903e04a59b9a1a059ff4d5f2267bd2010104e7f115a2a48121e8e1cd7"
    assert sha256_file(AUDIT) == "ee4d6e50c921b388a3f23aa2a79b341af3d59fa8c4ef5f887b026038a40f4656"


def test_split_counts_are_exact() -> None:
    rows = load_rows()
    assert len(rows) == 1197
    assert Counter(row["split"] for row in rows) == {
        "train": 837,
        "val": 180,
        "test": 180,
    }
    for split in ("val", "test"):
        assert Counter(row["label"] for row in rows if row["split"] == split) == {
            label: 30 for label in LABELS
        }


def test_conflict_group_is_absent() -> None:
    rows = load_rows()
    group_ids = {row["group_id"] for row in rows}
    assert "g00463" not in group_ids
    paths = {row["relative_path"] for row in rows}
    assert not any("FreshOrange (166).jpg" in path for path in paths)
    assert not any("FreshOrange (61).jpg" in path for path in paths)
    assert not any("RottenOrange (108).jpg" in path for path in paths)


def test_groups_and_exact_hashes_do_not_cross_splits() -> None:
    rows = load_rows()
    groups: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], set()).add(row["split"])
        hashes.setdefault(row["sha256"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert all(len(splits) == 1 for splits in hashes.values())


def test_summary_matches_manifest() -> None:
    rows = load_rows()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["total_images"] == len(rows)
    assert summary["counts"] == {
        "train": {
            "fresh_apple": 140,
            "fresh_banana": 140,
            "fresh_orange": 138,
            "rotten_apple": 140,
            "rotten_banana": 140,
            "rotten_orange": 139,
        },
        "val": {label: 30 for label in LABELS},
        "test": {label: 30 for label in LABELS},
    }
    assert summary["group_counts"] == {
        "train": 817,
        "val": 176,
        "test": 173,
    }
    assert summary["validation"]["no_group_overlap"] is True
    assert summary["validation"]["no_exact_hash_overlap"] is True
    assert summary["split_manifest_sha256"] == sha256_file(MANIFEST)


def test_audit_evidence_matches_the_selected_data() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["image_count"] == 1200
    assert audit["near_duplicate_distance"] == 6
    assert len(audit["conflict_groups"]) == 1
    assert audit["conflict_groups"][0]["group_id"] == "g00463"
    assert sorted(audit["conflict_groups"][0]["labels"]) == ["fresh_orange", "rotten_orange"]
