"""Download the pinned Zenodo archive and prepare a nine-class detection subset.

Freshness labels are never inferred. Source split names are retained; the output
is an audited staging dataset, not an independently validated benchmark.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, hashlib, io, json, math, shutil, time, zipfile
import requests
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
META = PROJECT / 'datasets/research/2026-10-31-user-dataset-review/zenodo20216.json'
NAMES = ['apple', 'banana', 'grape', 'lemon', 'mango', 'orange',
         'pomegranate', 'strawberry', 'tomato']


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def download():
    metadata = json.loads(META.read_text(encoding='utf-8'))
    source = next(x for x in metadata['files'] if x['key'] == 'AgricultureDataset.zip')
    expected_md5 = source['checksum'].split(':', 1)[1]
    archive = ROOT / 'downloads/AgricultureDataset.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        if archive.stat().st_size != source['size'] or digest(archive, 'md5') != expected_md5:
            raise ValueError('Existing archive fails publisher checksum; retained for inspection')
        return archive
    partial = archive.with_suffix('.zip.partial')
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > source['size']:
            raise ValueError('Partial archive larger than expected')
        if offset == source['size']:
            break
        if shutil.disk_usage(ROOT).free < (source['size'] - offset) + 2_000_000_000:
            raise RuntimeError('Insufficient download/extraction headroom; refusing to fill disk')
        try:
            headers = {'User-Agent': 'FruitLens-DatasetAudit/1.0', 'Accept-Encoding': 'identity'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            with requests.get(source['links']['self'], headers=headers, stream=True, timeout=(30, 90)) as r:
                r.raise_for_status()
                if offset and (r.status_code != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    raise RuntimeError('Server did not honor resume offset; partial file retained')
                if not offset and r.status_code not in (200, 206):
                    raise RuntimeError(f'Unexpected HTTP status {r.status_code}')
                if not offset and r.status_code == 206 and not r.headers.get('Content-Range', '').startswith('bytes 0-'):
                    raise RuntimeError('Wrong initial byte range')
                total = offset
                last = 0
                with partial.open('ab' if offset else 'wb') as out:
                    for block in r.iter_content(1024 * 1024):
                        if not block:
                            continue
                        total += len(block)
                        if total > source['size']:
                            raise ValueError('Download exceeds publisher size')
                        out.write(block)
                        if time.monotonic() - last > 15:
                            save_json(ROOT / 'reports/status.json', {'stage': 'downloading', 'bytes': total, 'expected_bytes': source['size']})
                            print(f'Download {total / source["size"]:.1%} ({total:,} bytes)', flush=True)
                            last = time.monotonic()
            if partial.stat().st_size != source['size']:
                raise ValueError('Short download; retrying with resume')
            break
        except (requests.RequestException, ValueError) as exc:
            if attempt == 4:
                raise
            print(f'Download retry {attempt + 1}: {exc}', flush=True)
            time.sleep(2)
    actual_md5 = digest(partial, 'md5')
    if actual_md5 != expected_md5:
        raise ValueError(f'Publisher MD5 mismatch: {actual_md5}; partial retained')
    partial.replace(archive)
    save_json(ROOT / 'reports/download.json', dict(url=source['links']['self'],
              record='https://zenodo.org/records/18904638', bytes=archive.stat().st_size,
              md5=actual_md5, sha256=digest(archive), publisher_checksum_verified=True))
    return archive


def prepare(archive):
    report_dir = ROOT / 'reports'
    report_dir.mkdir(exist_ok=True)
    rows, rejected, duplicates = [], [], []
    all_boxes, selected_boxes = Counter(), Counter()
    source_photos = defaultdict(Counter)
    hashes = {}
    existing_holdout = set()
    for manifest in [PROJECT / 'detector_training/yolov8n-fruit9-v1/reports/split_manifest.csv',
                     PROJECT / 'detector_training/yolov8n-fruit9-v2/reports/split_manifest.csv']:
        if manifest.exists():
            with manifest.open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    if row.get('split') in ('test', 'val', 'valid'):
                        for field in ('source_sha256', 'prepared_sha256', 'sha256'):
                            if row.get(field): existing_holdout.add(row[field])
    with zipfile.ZipFile(archive) as z:
        configs = [x for x in z.infolist() if x.filename.endswith('/data.yaml')]
        source_yaml = z.read(configs[0])
        config = yaml.safe_load(source_yaml)
        original_names = config['names']
        if isinstance(original_names, dict):
            original_names = [original_names[i] for i in range(len(original_names))]
        mapping = {i: NAMES.index(n.lower()) for i, n in enumerate(original_names) if n.lower() in NAMES}
        if len(mapping) != 9:
            raise ValueError('Pinned archive does not contain expected nine names')
        (ROOT / 'source-data.yaml').write_bytes(source_yaml)
        save_json(report_dir / 'class-map.json', {'source_names': original_names,
                  'target_names': NAMES, 'source_to_target': mapping})
        members = {x.filename: x for x in z.infolist() if not x.is_dir()}
        label_members = [n for n in members if '/labels/' in n and n.endswith('.txt')]
        # Keep a duplicate from test before validation before train, rather than
        # leaking a source's held-out image into training.
        label_members.sort(key=lambda n: (0 if '/test/' in n else 1 if '/valid/' in n else 2, n))
        for index, member in enumerate(label_members):
            text = z.read(member).decode('utf-8-sig')
            mapped, error = [], None
            present = set()
            for number, line in enumerate(text.splitlines(), 1):
                if not line.strip(): continue
                fields = line.split()
                try:
                    if len(fields) != 5: raise ValueError('Not five-field YOLO detection row')
                    cls = int(fields[0]); box = list(map(float, fields[1:]))
                    if cls < 0 or cls >= len(original_names): raise ValueError('Class out of range')
                    if not all(math.isfinite(v) for v in box): raise ValueError('Nonfinite box')
                    x, y, w, h = box
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError('Invalid normalized box')
                    if min(x-w/2, y-h/2) < -0.0001 or max(x+w/2, y+h/2) > 1.0001:
                        raise ValueError('Box extends outside image')
                    all_boxes[original_names[cls]] += 1
                    present.add(cls)
                    if cls in mapping:
                        mapped.append((mapping[cls], box))
                except ValueError as exc:
                    error = f'line {number}: {exc}'
                    break
            source_split = next(s for s in ('test', 'valid', 'train') if f'/{s}/' in member)
            for cls in present: source_photos[source_split][original_names[cls]] += 1
            if error:
                rejected.append({'member': member, 'reason': error}); continue
            if not mapped: continue
            stem = member.replace('/labels/', '/images/').rsplit('.', 1)[0]
            image_member = next((stem+ext for ext in ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.PNG') if stem+ext in members), None)
            if image_member is None:
                rejected.append({'member': member, 'reason': 'Missing matching image'}); continue
            payload = z.read(image_member)  # ZIP member CRC is checked here.
            sha = hashlib.sha256(payload).hexdigest()
            if sha in existing_holdout:
                rejected.append({'member': image_member, 'reason': 'Matches old holdout source SHA256'}); continue
            canonical = tuple(sorted((cls, *[round(v, 6) for v in box]) for cls, box in mapped))
            if sha in hashes:
                duplicates.append({'member': image_member, 'kept': hashes[sha][0],
                                   'label_conflict': hashes[sha][1] != canonical})
                continue
            try:
                with Image.open(io.BytesIO(payload)) as im:
                    im.load()
                    width, height = im.size
                    if width < 2 or height < 2: raise ValueError('Image too small')
            except Exception as exc:
                rejected.append({'member': image_member, 'reason': f'Decode: {exc}'}); continue
            split = 'val' if source_split == 'valid' else source_split
            filename = sha[:24]
            extension = Path(image_member).suffix.lower()
            image_out = ROOT / f'prepared/images/{split}/{filename}{extension}'
            label_out = ROOT / f'prepared/labels/{split}/{filename}.txt'
            original_out = ROOT / f'original-labels/{filename}.txt'
            for path in (image_out, label_out, original_out): path.parent.mkdir(parents=True, exist_ok=True)
            image_out.write_bytes(payload)
            label_out.write_text('\n'.join(f'{cls} '+ ' '.join(f'{v:.8f}' for v in box) for cls, box in mapped)+'\n', encoding='utf-8')
            original_out.write_text(text, encoding='utf-8')
            hashes[sha] = (image_member, canonical)
            for cls, box in mapped: selected_boxes[(split, NAMES[cls])] += 1
            rows.append(dict(image=image_out.relative_to(ROOT).as_posix(),
                        label=label_out.relative_to(ROOT).as_posix(), source_image=image_member,
                        source_label=member, source_split=source_split, split=split, sha256=sha,
                        width=width, height=height, boxes=len(mapped),
                        fruits=';'.join(sorted({NAMES[cls] for cls, box in mapped}))))
            if index % 1000 == 0:
                print(f'Inspected {index}/{len(label_members)} labels; selected {len(rows)} images', flush=True)
        save_json(report_dir / 'annotation-audit.json', dict(source_label_files=len(label_members),
                  source_box_counts=dict(all_boxes), source_images_per_class_by_split={s: dict(c) for s,c in source_photos.items()},
                  rejected=rejected, exact_duplicates=duplicates))
    if not rows: raise ValueError('No matching images')
    with (report_dir / 'images.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Paths are deliberately project-relative when possible; resolve using this
    # YAML's absolute dataset root for Ultralytics training from any directory.
    out_config = dict(path=(ROOT / 'prepared').as_posix(), train='images/train',
                      val='images/val', test='images/test', nc=9, names=NAMES)
    (ROOT / 'data.yaml').write_text(yaml.safe_dump(out_config, sort_keys=False), encoding='utf-8')
    totals = Counter(row['split'] for row in rows)
    summary = dict(stage='prepared', task='fruit_identity_detection_only', images=len(rows),
                   splits=dict(totals), boxes=sum(selected_boxes.values()),
                   per_split_class_boxes={s: {n: selected_boxes[s,n] for n in NAMES} for s in ('train','val','test')},
                   rejected=len(rejected), exact_duplicate_images_removed=len(duplicates),
                   duplicate_label_conflicts=sum(x['label_conflict'] for x in duplicates),
                   freshness_labels_available=False, independent_phone_validation=False,
                   source_splits_retained=True, near_duplicate_and_source_family_audit_complete=False,
                   training_ready=False,
                   remaining_checks=['source-family and perceptual overlap within source and against old holdouts',
                                     'visual box completeness and label-conflict review', 'independent phone holdout'],
                   free_disk_bytes=shutil.disk_usage(ROOT).free)
    if any(sum(selected_boxes[s,n] for s in ('train','val','test')) == 0 for n in NAMES):
        raise ValueError('At least one target fruit has no retained boxes')
    save_json(report_dir / 'summary.json', summary)
    save_json(report_dir / 'status.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    source_archive = download()
    if not args.download_only:
        prepare(source_archive)
        from quarantine_conflicts import main as quarantine_conflicts
        quarantine_conflicts()
