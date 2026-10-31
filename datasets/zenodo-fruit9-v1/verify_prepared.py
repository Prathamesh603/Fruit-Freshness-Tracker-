"""Check installed image hashes, matching labels, counts and exact split leakage."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, math, shutil, zipfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
NAMES = ['apple','banana','grape','lemon','mango','orange','pomegranate','strawberry','tomato']


def main():
    report = ROOT / 'reports'
    with (report/'images.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    errors, hashes, expected = [], set(), set()
    boxes, photos, splits = Counter(), Counter(), Counter()
    heldout = set()
    for version in ('v1','v2'):
        manifest = PROJECT/f'detector_training/yolov8n-fruit9-{version}/reports/split_manifest.csv'
        with manifest.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                if row['split'] in ('val','valid','test'):
                    for key in ('source_sha256','prepared_sha256'):
                        if row.get(key): heldout.add(row[key])
    for row in rows:
        image, label = ROOT/row['image'], ROOT/row['label']
        expected.add(image.resolve()); expected.add(label.resolve())
        sha = hashlib.sha256(image.read_bytes()).hexdigest()
        if sha != row['sha256']: errors.append(f'Image hash changed: {image}')
        if sha in hashes: errors.append(f'Duplicate retained image: {image}')
        if sha in heldout: errors.append(f'Exact old holdout overlap: {image}')
        hashes.add(sha)
        classes = set(); count = 0
        for line in label.read_text().splitlines():
            if not line.strip(): continue
            fields = line.split(); cls = int(fields[0]); values = list(map(float, fields[1:]))
            if len(values) != 4 or cls not in range(9) or not all(math.isfinite(x) for x in values):
                errors.append(f'Invalid label: {label}'); continue
            boxes[row['split'], NAMES[cls]] += 1; classes.add(NAMES[cls]); count += 1
        if count != int(row['boxes']): errors.append(f'Box count mismatch: {label}')
        for name in classes: photos[row['split'], name] += 1
        splits[row['split']] += 1
    actual = {p.resolve() for part in ('images','labels') for p in (ROOT/'prepared'/part).rglob('*') if p.is_file()}
    if actual != expected: errors.append(f'Manifest/file inventory mismatch: {len(actual ^ expected)}')
    source_rows = Counter(); source_box_count = 0
    with zipfile.ZipFile(ROOT/'downloads/AgricultureDataset.zip') as z:
        config = json.loads((report/'class-map.json').read_text())
        source_names = config['source_names']
        for member in z.namelist():
            if '/labels/' not in member or not member.endswith('.txt'): continue
            for line in z.read(member).decode('utf-8-sig').splitlines():
                if not line.strip(): continue
                source_box_count += 1
                fields = line.split()
                if fields and fields[0].isdigit() and int(fields[0]) < len(source_names):
                    source_rows[source_names[int(fields[0])]] += 1
    summary = json.loads((report/'summary.json').read_text())
    if len(rows) != summary['images'] or sum(boxes.values()) != summary['boxes']:
        errors.append('Summary counts mismatch')
    verification = dict(passed=not errors, errors=errors, images_checked=len(rows),
        boxes_checked=sum(boxes.values()), no_exact_duplicates_in_active_splits=len(hashes)==len(rows),
        exact_old_holdout_matches=sum(r['sha256'] in heldout for r in rows),
        source_nonempty_annotation_rows=source_box_count, source_raw_class_rows=dict(source_rows),
        source_rows_count_note='Raw nonempty rows, including any geometrically invalid boxes rejected during preparation',
        free_disk_bytes=shutil.disk_usage(ROOT).free,
        near_duplicate_and_visual_audit_complete=False)
    (report/'installation-verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    with (report/'class-coverage.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['fruit','train_images','val_images','test_images','train_boxes','val_boxes','test_boxes'])
        for name in NAMES:
            writer.writerow([name]+[photos[s,name] for s in ('train','val','test')]+[boxes[s,name] for s in ('train','val','test')])
    table = '\n'.join('| '+n+' | '+' | '.join(str(photos[s,n]) for s in ('train','val','test'))+' | '+str(boxes['train',n])+' |' for n in NAMES)
    text=f'''# Installed dataset results — 31 October 2026

The publisher checksum passed. The active nine-fruit subset contains **{len(rows):,} photos and {sum(boxes.values()):,} boxes**. The source splits are retained: {splits['train']:,} training, {splits['val']:,} validation, {splits['test']:,} test photos. These are staging splits, not verified independent evaluation data.

| Fruit | Train photos | Validation photos | Test photos | Training boxes |
|---|---:|---:|---:|---:|
{table}

## Important change to the initial recommendation

Metadata established that all nine names existed; actual extraction shows that this collection is **not balanced enough to use alone for our nine fruits**. It is useful mainly as supplementary detection data. In particular, banana, strawberry and pomegranate need stronger data coverage. Existing useful nine-fruit data and the freshness photos were retained during cleanup.

The installer removed 3,445 repeated image copies and rejected 134 source label files with invalid geometry/format. Among retained duplicate groups, **627 photos had conflicting target annotations**; those photos and their provisional labels were moved out of the active splits into `quarantine/conflicting-annotations/`. There were 865 conflict events across the duplicate copies, not 865 unique quarantined photos.

No training was started. More source-family/perceptual duplicate checks and visual annotation review are needed before a trustworthy detector experiment. This source has fruit-identity boxes and cannot supply freshness targets for classifier training.

See `reports/installation-verification.json`, `reports/class-coverage.csv`, and `reports/conflict-quarantine.json` for measured evidence. The source raw row count was {source_box_count:,}, compared with 222,301 annotations claimed on the publication page. This discrepancy remains unresolved; raw rows are not automatically valid boxes. Free space at verification was approximately {shutil.disk_usage(ROOT).free/1e9:.2f} GB in decimal units.
'''
    (ROOT/'INSTALLATION_RESULTS.md').write_text(text,encoding='utf-8')
    print(json.dumps(verification,indent=2))
    if errors: raise SystemExit(1)


if __name__ == '__main__': main()
