from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fftpack import dct

from common import (
    EXTRACTED_ROOT,
    LABELS,
    MANIFEST_ROOT,
    ORIGINAL_ZIP,
    REPORT_ROOT,
    SOURCE_CLASS_MAP,
    SOURCE_ID,
    iter_images,
    relative_to_project,
    sha256_file,
    write_json,
)

PHASH_SIZE = 32
PHASH_LOW_FREQ = 8


def extract_originals() -> None:
    if not ORIGINAL_ZIP.exists():
        raise FileNotFoundError(ORIGINAL_ZIP)
    with zipfile.ZipFile(ORIGINAL_ZIP) as archive:
        names = archive.namelist()
        for source_class in SOURCE_CLASS_MAP:
            target = EXTRACTED_ROOT / source_class
            target.mkdir(parents=True, exist_ok=True)
            prefix = f"Original Image/{source_class}/"
            members = [
                name
                for name in names
                if name.startswith(prefix)
                and not name.endswith("/")
                and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            existing = {path.name for path in target.glob("*") if path.is_file()}
            for member in members:
                output = target / Path(member).name
                if output.name in existing:
                    continue
                with archive.open(member) as source, output.open("wb") as destination:
                    destination.write(source.read())
    print(f"extracted originals to {EXTRACTED_ROOT}")


def perceptual_hash(image: Image.Image) -> int:
    gray = image.convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.float32)
    coefficients = dct(dct(values, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coefficients[:PHASH_LOW_FREQ, :PHASH_LOW_FREQ]
    threshold = np.median(low[1:, :])
    bits = (low > threshold).flatten()
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def popcount(value: int) -> int:
    return int(value).bit_count()


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def make_grid(entries: list[dict], output: Path) -> None:
    thumb = 180
    label_height = 28
    columns = 5
    rows = len(LABELS)
    canvas = Image.new("RGB", (thumb * columns, (thumb + label_height) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for row, label in enumerate(LABELS):
        label_entries = [entry for entry in entries if entry["label"] == label][:columns]
        for column in range(columns):
            x = column * thumb
            y = row * (thumb + label_height)
            if column >= len(label_entries):
                continue
            entry = label_entries[column]
            with Image.open(entry["path"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = ImageOps.fit(image, (thumb, thumb), method=Image.Resampling.LANCZOS)
                canvas.paste(image, (x, y))
            draw.text((x + 4, y + thumb + 5), f"{label} {Path(entry['path']).name[:20]}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def audit(near_duplicate_distance: int) -> dict:
    classes = list(SOURCE_CLASS_MAP.keys())
    entries = []
    failures = []
    for source_class in classes:
        class_root = EXTRACTED_ROOT / source_class
        if not class_root.exists():
            raise FileNotFoundError(class_root)
        paths = list(iter_images(class_root))
        for path in paths:
            try:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image)
                    width, height = image.size
                    mode = image.mode
                    image = image.convert("RGB")
                    phash = perceptual_hash(image)
            except Exception as error:
                failures.append({"path": relative_to_project(path), "error": repr(error)})
                continue
            entries.append(
                {
                    "path": path,
                    "relative_path": relative_to_project(path),
                    "source_class": source_class,
                    "label": SOURCE_CLASS_MAP[source_class],
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "phash": phash,
                }
            )

    if failures:
        raise RuntimeError(f"failed to read {len(failures)} images")

    exact_groups = defaultdict(list)
    for index, entry in enumerate(entries):
        exact_groups[entry["sha256"]].append(index)

    union = UnionFind(len(entries))
    hashes = np.asarray([entry["phash"] for entry in entries], dtype=np.uint64)
    distances = np.bitwise_count(hashes[:, None] ^ hashes[None, :]).astype(np.uint16)
    upper_mask = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    distance_counts = {
        str(threshold): int(np.count_nonzero((distances <= threshold) & upper_mask))
        for threshold in (0, 2, 4, 6, 8, 10, 12)
    }
    pair_indices = np.argwhere((distances <= near_duplicate_distance) & upper_mask)
    for left, right in pair_indices:
        union.union(int(left), int(right))

    grouped = defaultdict(list)
    for index in range(len(entries)):
        grouped[union.find(index)].append(index)

    group_id_by_index = {}
    conflict_groups = []
    group_sizes = []
    for group_number, indices in enumerate(sorted(grouped.values(), key=lambda value: min(value))):
        group_id = f"g{group_number:05d}"
        labels = sorted({entries[index]["label"] for index in indices})
        if len(labels) > 1:
            conflict_groups.append(
                {
                    "group_id": group_id,
                    "labels": labels,
                    "paths": [entries[index]["relative_path"] for index in indices],
                }
            )
        group_sizes.append(len(indices))
        for index in indices:
            group_id_by_index[index] = group_id

    for entry, index in zip(entries, range(len(entries))):
        entry["group_id"] = group_id_by_index[index]
        entry["phash"] = f"{entry['phash']:016x}"
        entry["path"] = entry["relative_path"]

    exact_duplicate_groups = [
        {
            "sha256": digest,
            "count": len(indices),
            "labels": sorted({entries[index]["label"] for index in indices}),
            "paths": [entries[index]["relative_path"] for index in indices],
        }
        for digest, indices in exact_groups.items()
        if len(indices) > 1
    ]

    size_widths = [entry["width"] for entry in entries]
    size_heights = [entry["height"] for entry in entries]
    label_counts = Counter(entry["label"] for entry in entries)
    group_label_counts = Counter()
    for indices in grouped.values():
        labels = {entries[index]["label"] for index in indices}
        if len(labels) == 1:
            group_label_counts[next(iter(labels))] += 1

    audit_payload = {
        "source_id": SOURCE_ID,
        "archive": relative_to_project(ORIGINAL_ZIP),
        "archive_sha256": sha256_file(ORIGINAL_ZIP),
        "selected_source_classes": list(SOURCE_CLASS_MAP.keys()),
        "near_duplicate_distance": near_duplicate_distance,
        "image_count": len(entries),
        "label_counts": dict(label_counts),
        "group_counts": dict(group_label_counts),
        "size": {
            "width_min": min(size_widths),
            "width_max": max(size_widths),
            "height_min": min(size_heights),
            "height_max": max(size_heights),
        },
        "distance_pair_counts": distance_counts,
        "exact_duplicate_groups": exact_duplicate_groups,
        "conflict_groups": conflict_groups,
        "largest_groups": sorted(group_sizes, reverse=True)[:20],
        "images": [
            {
                "relative_path": entry["relative_path"],
                "source_class": entry["source_class"],
                "label": entry["label"],
                "width": entry["width"],
                "height": entry["height"],
                "mode": entry["mode"],
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
                "phash": entry["phash"],
                "group_id": entry["group_id"],
            }
            for entry in entries
        ],
    }

    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST_ROOT / "data_audit.json", audit_payload)
    write_json(
        MANIFEST_ROOT / "excluded_conflict_groups.json",
        {"groups": conflict_groups, "count": len(conflict_groups)},
    )
    make_grid(entries, REPORT_ROOT / "sample_grid.png")

    print(json.dumps(
        {
            "images": len(entries),
            "labels": dict(label_counts),
            "groups": len(grouped),
            "single_image_groups": group_sizes.count(1),
            "largest_group": max(group_sizes),
            "exact_duplicate_groups": len(exact_duplicate_groups),
            "conflict_groups": len(conflict_groups),
            "distance_pair_counts": distance_counts,
        },
        indent=2,
    ))
    return audit_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--near-duplicate-distance", type=int, default=6)
    args = parser.parse_args()
    if not args.skip_extract:
        extract_originals()
    audit(args.near_duplicate_distance)


if __name__ == "__main__":
    main()
