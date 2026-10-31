"""Extract originals, audit files, and inventory labels. Does not train a model."""
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
import csv
import concurrent.futures
import hashlib
import json
import zipfile
import zlib
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
FRUITS = ['apple','banana','grape','guava','jujube','lemon','lulo','mango','orange','pomegranate','strawberry','tamarillo','tomato']
IMAGE_EXT = {'.jpg', '.jpeg', '.png'}


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rel(path):
    return path.relative_to(ROOT).as_posix()


def extract(archive, output, extensions=None):
    """Only regular members, with zip-slip/symlink/duplicate destination checks."""
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if '\\' in member.filename or '..' in parts or PurePosixPath(member.filename).is_absolute():
                raise ValueError('Unsafe archive name: ' + member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive contains a symlink')
            suffix = PurePosixPath(member.filename).suffix.lower()
            if extensions is not None and suffix not in extensions:
                continue
            target = output.joinpath(*parts[1:]).resolve()
            target.relative_to(output.resolve())
            key = str(target).casefold()
            if key in seen:
                raise ValueError('Duplicate archive path')
            seen.add(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != member.file_size or zlib.crc32(target.read_bytes()) != member.CRC:
                    raise ValueError('Existing extraction differs: ' + str(target))
            else:
                # ZipFile.read checks CRC before a new file is written.
                data = z.read(member)
                target.write_bytes(data)


def inspect_image(path):
    data = path.read_bytes()
    with Image.open(path) as im:
        width, height = im.size
        # JPEG draft decoding keeps this inventory fast; originals stay unchanged.
        # This is a readability check, not exhaustive full-resolution pixel QA.
        im.draft('RGB',(64,64))
        im.load()
        gray = np.asarray(ImageOps.exif_transpose(im).convert('L').resize((32,32)), dtype=np.float32)
    low = dctn(gray, type=2, norm='ortho')[:8,:8]
    bits = low > np.median(low.flatten()[1:])
    phash = 0
    for bit in bits.flatten():
        phash = (phash << 1) | int(bit)
    return {'width': width, 'height': height, 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(), 'phash': f'{phash:016x}'}


def group_rows(rows):
    """BK-tree radius-6 pHash search, plus explicit SHA256 grouping."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a,b):
        parent[find(b)] = find(a)
    tree = None
    exact = {}
    for i,row in enumerate(rows):
        if not row.get('phash'):
            continue
        if row['sha256'] in exact:
            union(i,exact[row['sha256']])
        else:
            exact[row['sha256']] = i
        value = int(row['phash'],16)
        if tree is None:
            tree = [value,i,{}]
            continue
        pending = [tree]
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= 6:
                union(i,node[1])
            pending.extend(child for edge,child in node[2].items() if distance-6 <= edge <= distance+6)
        node = tree
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                break
            if distance not in node[2]:
                node[2][distance] = [value,i,{}]
                break
            node = node[2][distance]
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    conflicts = []
    exact_seen = set()
    for number,indices in enumerate(groups.values()):
        group = f'phash_{number:06d}'
        labels = {rows[i]['label'] for i in indices if rows[i]['decode_ok']}
        for i in indices:
            row = rows[i]
            row['group_id'] = group
            row['group_has_label_conflict'] = len(labels)>1
            row['exact_duplicate'] = row['sha256'] in exact_seen
            exact_seen.add(row['sha256'])
            row['review_status'] = ('unreadable' if not row['decode_ok'] else
                                    'review_label_conflict' if len(labels)>1 else
                                    'duplicate_candidate' if row['exact_duplicate'] else 'awaiting_visual_review')
            # No final split until labels/session groups have been reviewed.
            row['split'] = 'unassigned'
            if len(labels)>1:
                conflicts.append({'group_id':group,'image_id':row['image_id'],'path':row['path'],'label':row['label']})
    return conflicts


def sample_grid(rows, output):
    labels = sorted({r['label'] for r in rows})
    canvas = Image.new('RGB',(800,len(labels)*155),'#f5f5f5')
    draw = ImageDraw.Draw(canvas)
    for y,label in enumerate(labels):
        candidates = [r for r in rows if r['label']==label and r['decode_ok']]
        candidates.sort(key=lambda r:r['sha256'])
        draw.text((5,y*155+3),label,fill='black')
        for x,row in enumerate(candidates[:5]):
            with Image.open(ROOT/row['path']) as im:
                tile = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(150,125))
                canvas.paste(tile,(x*160,y*155+22))
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output)


def classification():
    existing = PROJECT/'ml/data/raw/mendeley_bdd69gyhv8/Original_Image.zip'
    expected = 'f89d67d4c4b24810bcd8406db877db6b1ac9834cea0b0cb56d65acebe59e66ff'
    hasher = hashlib.sha256()
    with existing.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):hasher.update(chunk)
    if hasher.hexdigest()!=expected:
        raise ValueError('Original eight-fruit archive checksum changed')
    extract(existing,ROOT/'raw/mendeley8/images',IMAGE_EXT)
    extract(ROOT/'raw/fruit16k/publisher.zip',ROOT/'raw/fruit16k/images',IMAGE_EXT)
    rows=[]
    for source in ['mendeley8','fruit16k']:
        base=ROOT/'raw'/source/'images'
        for path in sorted(base.rglob('*')):
            if path.suffix.lower() not in IMAGE_EXT:continue
            folder=path.parent.name
            if source=='mendeley8':
                prefix='Fresh' if folder.startswith('Fresh') else 'Rotten'
                fruit=folder[len(prefix):].lower()
                condition='fresh' if prefix=='Fresh' else 'spoiled'
            else:
                prefix,fruit=folder.split('_',1);fruit=fruit.lower()
                if prefix not in ['F','S']:raise ValueError(folder)
                condition='fresh' if prefix=='F' else 'spoiled'
            if fruit not in FRUITS:raise ValueError('Unmapped fruit '+fruit)
            row={'image_id':source+'_'+path.parent.name+'_'+path.stem,'source':source,
                 'path':rel(path),'source_folder':folder,'fruit':fruit,'condition':condition,
                 'label':condition+'_'+fruit,'label_origin':'publisher_folder','license':'CC-BY-4.0',
                 'physical_fruit_id':'','capture_session_id':'','width':'','height':'','bytes':path.stat().st_size,
                 'sha256':'','phash':'','decode_ok':True,'decode_error':'',
                 'baseline_overlap':'not_checked'}
            rows.append(row)
    def inspect_row(row):
        try:row.update(inspect_image(ROOT/row['path']))
        except Exception as error:row.update(decode_ok=False,decode_error=str(error))
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        audited=[]
        for row in executor.map(inspect_row,rows):
            audited.append(row)
            if len(audited)%2000==0:print('Audited classification',len(audited),flush=True)
        rows=audited
    # Mark all images previously used by the completed 3-fruit experiment.
    old=PROJECT/'ml/data/manifests/splits_v1.csv'
    with old.open(encoding='utf-8',newline='') as f:
        prior=list(csv.DictReader(f))
    old_hashes={r.get('sha256','') for r in prior}
    if not old_hashes- {''}:
        # Manifest formats can change. Hash recorded source paths rather than guess.
        old_hashes=set()
        for r in prior:
            name=r.get('path') or r.get('image_path')
            if name:
                p=Path(name)
                if not p.is_absolute():p=PROJECT/p
                if p.exists():old_hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
    for row in rows:row['baseline_overlap']=row['sha256'] in old_hashes
    conflicts=group_rows(rows)
    write_csv(ROOT/'classification/images.csv',rows)
    write_csv(ROOT/'review/label_conflicts.csv',conflicts,['group_id','image_id','path','label'])
    sample_grid(rows,ROOT/'review/classification_samples.jpg')
    counts=Counter((r['fruit'],r['condition']) for r in rows)
    summary={'images':len(rows),'decode_failures':sum(not r['decode_ok'] for r in rows),
             'fruits':len({r['fruit'] for r in rows}),'classes':len(counts),
             'exact_duplicate_copies':sum(r['exact_duplicate'] for r in rows),
             'conflict_images':len(conflicts),'conflict_groups':len({r['group_id'] for r in conflicts}),
             'baseline_overlap_images':sum(r['baseline_overlap'] for r in rows),
             'counts':{f'{condition}_{fruit}':n for (fruit,condition),n in sorted(counts.items())},
             'split_status':'unassigned; review source labels and capture groups before freezing splits',
             'training_ready':False,'archive_sha256':hasher.hexdigest()}
    (ROOT/'reports/classification_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Classification summary',json.dumps(summary),flush=True)


def detection_boxes():
    extract(ROOT/'raw/fruit_freshness_boxes/publisher.zip',ROOT/'raw/fruit_freshness_boxes/extracted')
    base=ROOT/'raw/fruit_freshness_boxes/extracted'
    rows=[];issues=[];images=[]
    for xml in sorted((base/'Annotations').glob('*.xml')):
        doc=ET.parse(xml).getroot()
        path=base/'JPEGImages'/Path(doc.findtext('filename')).name
        record={'image_id':'boxes_'+xml.stem,'path':rel(path),'source':'fruit_freshness_boxes',
                'annotation_path':rel(xml),'license':'Apache-2.0 repository; image provenance needs review',
                'review_status':'awaiting_visual_and_provenance_review','split':'unassigned'}
        try:record.update(inspect_image(path))
        except Exception as error:issues.append({'path':rel(path),'error':str(error)});continue
        images.append(record)
        w,h=record['width'],record['height']
        for obj in doc.findall('object'):
            label=obj.findtext('name');condition,fruit=label.split('_',1)
            if fruit not in FRUITS:raise ValueError(label)
            bbox=obj.find('bndbox')
            coords=[float(bbox.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
            valid=0<=coords[0]<coords[2]<=w and 0<=coords[1]<coords[3]<=h
            if not valid:issues.append({'path':rel(xml),'error':'invalid box '+str(coords)})
            rows.append({'image_id':record['image_id'],'path':rel(path),'fruit':fruit,
                         'source_condition':condition,'source_label':label,
                         'xmin':coords[0],'ymin':coords[1],'xmax':coords[2],'ymax':coords[3],
                         'width':w,'height':h,'coordinate_convention':'publisher VOC; review inclusive endpoints before conversion',
                         'box_valid':valid,'difficult':obj.findtext('difficult','0'),
                         'annotation_origin':'publisher_manual','split':'unassigned'})
    write_csv(ROOT/'detection/boxed_images.csv',images)
    write_csv(ROOT/'detection/publisher_boxes.csv',rows)
    (ROOT/'reports/detection_boxes_audit.json').write_text(json.dumps({
        'images':len(images),'boxes':len(rows),'fruit_counts':dict(Counter(r['fruit'] for r in rows)),
        'source_label_counts':dict(Counter(r['source_label'] for r in rows)),
        'issues':issues,'training_ready':False},indent=2),encoding='utf-8')
    print('Detection boxes',len(images),len(rows),'issues',len(issues),flush=True)


if __name__=='__main__':
    classification()
    detection_boxes()
