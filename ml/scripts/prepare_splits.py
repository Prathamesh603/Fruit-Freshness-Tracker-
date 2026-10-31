from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from common import LABELS, LABEL_META, MANIFEST_ROOT, read_json, sha256_file, write_json

SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def validate_group_splits(rows: list[dict]) -> None:
    group_splits = defaultdict(set)
    hashes = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
        hashes[row["sha256"]].add(row["split"])
    leaked_groups = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    leaked_hashes = {key: sorted(value) for key, value in hashes.items() if len(value) > 1}
    if leaked_groups:
        raise RuntimeError(f"groups cross splits: {list(leaked_groups.items())[:5]}")
    if leaked_hashes:
        raise RuntimeError(f"hashes cross splits: {list(leaked_hashes.items())[:5]}")
    for split in SPLITS:
        present = {row["label"] for row in rows if row["split"] == split}
        missing = set(LABELS) - present
        if missing:
            raise RuntimeError(f"{split} missing labels: {sorted(missing)}")


def assign_class_groups(groups: list[list[dict]], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    total = sum(len(group) for group in shuffled)
    targets = {split: int(round(total * ratio)) for split, ratio in SPLIT_RATIOS.items()}
    targets["train"] = total - targets["val"] - targets["test"]
    counts = {split: 0 for split in SPLITS}
    assignments = {}
    for group in shuffled:
        split = max(SPLITS, key=lambda name: (targets[name] - counts[name]) / max(targets[name], 1))
        assignments[group[0]["group_id"]] = split
        counts[split] += len(group)
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_path = MANIFEST_ROOT / f"splits_{args.version}.csv"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} already exists; pass --force to replace it")

    audit = read_json(MANIFEST_ROOT / "data_audit.json")
    excluded = read_json(MANIFEST_ROOT / "excluded_conflict_groups.json")
    excluded_groups = {group["group_id"] for group in excluded["groups"]}
    entries = [entry for entry in audit["images"] if entry["group_id"] not in excluded_groups]

    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["group_id"]].append(entry)
    groups_by_label = defaultdict(list)
    for group_id, group in grouped.items():
        labels = {entry["label"] for entry in group}
        if len(labels) != 1:
            raise RuntimeError(f"unresolved label conflict in {group_id}: {sorted(labels)}")
        groups_by_label[next(iter(labels))].append(group)

    assignments = {}
    for index, label in enumerate(LABELS):
        class_assignments = assign_class_groups(groups_by_label[label], args.seed + index)
        assignments.update(class_assignments)

    metadata_by_label = {item["id"]: item for item in LABEL_META}
    rows = []
    for entry in entries:
        meta = metadata_by_label[entry["label"]]
        rows.append(
            {
                "relative_path": entry["relative_path"],
                "label": entry["label"],
                "fruit": meta["fruit"],
                "condition": meta["condition"],
                "source_class": entry["source_class"],
                "sha256": entry["sha256"],
                "group_id": entry["group_id"],
                "width": entry["width"],
                "height": entry["height"],
                "split": assignments[entry["group_id"]],
            }
        )
    rows.sort(key=lambda row: (SPLITS.index(row["split"]), LABELS.index(row["label"]), row["relative_path"]))
    validate_group_splits(rows)

    fieldnames = [
        "relative_path",
        "label",
        "fruit",
        "condition",
        "source_class",
        "sha256",
        "group_id",
        "width",
        "height",
        "split",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {split: {label: 0 for label in LABELS} for split in SPLITS}
    group_counts = {split: set() for split in SPLITS}
    for row in rows:
        counts[row["split"]][row["label"]] += 1
        group_counts[row["split"]].add(row["group_id"])

    summary = {
        "version": args.version,
        "seed": args.seed,
        "source_id": audit["source_id"],
        "audit_path": "ml/data/manifests/data_audit.json",
        "audit_sha256": sha256_file(MANIFEST_ROOT / "data_audit.json"),
        "split_manifest": output_path.relative_to(MANIFEST_ROOT.parents[2]).as_posix(),
        "split_manifest_sha256": sha256_file(output_path),
        "total_images": len(rows),
        "excluded_conflict_groups": sorted(excluded_groups),
        "excluded_images": len(audit["images"]) - len(rows),
        "counts": counts,
        "group_counts": {split: len(groups) for split, groups in group_counts.items()},
        "ratios": {
            split: round(sum(counts[split].values()) / len(rows), 4)
            for split in SPLITS
        },
        "validation": {
            "no_group_overlap": True,
            "no_exact_hash_overlap": True,
            "all_labels_present": True,
        },
    }
    write_json(MANIFEST_ROOT / f"split_summary_{args.version}.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
