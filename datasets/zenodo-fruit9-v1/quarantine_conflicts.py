"""Keep all photos with conflicting duplicate annotations out of active splits."""
from pathlib import Path
from collections import Counter
import csv, json

ROOT = Path(__file__).resolve().parent


def main():
    reports = ROOT / 'reports'
    audit = json.loads((reports / 'annotation-audit.json').read_text(encoding='utf-8'))
    conflicts = {row['kept'] for row in audit['exact_duplicates'] if row['label_conflict']}
    manifest = reports / 'images.csv'
    with manifest.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    retained, quarantined = [], []
    boxes, photos = Counter(), Counter()
    for row in rows:
        if row['source_image'] in conflicts:
            moves = []
            for key in ('image', 'label'):
                source = (ROOT / row[key]).resolve()
                destination = (ROOT / 'quarantine/conflicting-annotations' / row[key]).resolve()
                # Both absolute paths must stay inside this exact dataset folder.
                source.relative_to(ROOT.resolve())
                destination.relative_to(ROOT.resolve())
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                moves.append({'from': row[key], 'to': destination.relative_to(ROOT).as_posix()})
            quarantined.append({'source_image': row['source_image'], 'sha256': row['sha256'], 'moves': moves})
            continue
        retained.append(row)
        photos[row['split']] += 1
        for line in (ROOT / row['label']).read_text().splitlines():
            if line.strip(): boxes[row['split'], int(line.split()[0])] += 1
    with manifest.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(retained)
    output = reports / 'conflict-quarantine.json'
    if quarantined:
        output.write_text(json.dumps(quarantined, indent=2), encoding='utf-8')
    summary = json.loads((reports / 'summary.json').read_text())
    names = ['apple', 'banana', 'grape', 'lemon', 'mango', 'orange', 'pomegranate', 'strawberry', 'tomato']
    summary.update(images=len(retained), splits=dict(photos), boxes=sum(boxes.values()),
        per_split_class_boxes={s: {n: boxes[s, i] for i,n in enumerate(names)} for s in ('train','val','test')},
        conflict_images_quarantined=len(json.loads(output.read_text())) if output.exists() else 0,
        active_conflicting_exact_duplicates=0)
    summary['remaining_checks'] = ['source-family and perceptual overlap within source and against old holdouts',
                                   'visual completeness and class imbalance; review quarantine before restoring',
                                   'independent phone holdout']
    for filename in ('summary.json', 'status.json'):
        (reports / filename).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
