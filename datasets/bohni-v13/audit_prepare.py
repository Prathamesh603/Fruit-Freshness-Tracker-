"""Audit the user-supplied Roboflow ZIP and prepare a compact mapped subset."""
from pathlib import Path,PurePosixPath
from collections import Counter,defaultdict
import zipfile,hashlib,csv,json,math,io,concurrent.futures
import numpy as np
from scipy.fft import dctn
from PIL import Image,ImageDraw,ImageOps
import yaml

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
ARCHIVE=ROOT.parent/'Fruits and Vegi.v13-verison8.yolov8.zip'
TARGETS=['apple','banana','grape','lemon','mango','orange','pomegranate','strawberry','tomato']


def csv_write(name,rows,fields=None):
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    ROOT.mkdir(exist_ok=True)
    for name in ['metadata','prepared','reports','review']:(ROOT/name).mkdir(exist_ok=True)
    digest=hashlib.sha256()
    with ARCHIVE.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
    with zipfile.ZipFile(ARCHIVE) as z:
        members=[m for m in z.infolist() if not m.is_dir()]
        for m in members:
            p=PurePosixPath(m.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in m.filename:raise ValueError('Unsafe ZIP path')
        assert len({m.filename.casefold() for m in members})==len(members)
        for name in ['data.yaml','README.dataset.txt','README.roboflow.txt']:
            (ROOT/'metadata'/name).write_bytes(z.read(name))
        cfg=yaml.safe_load(z.read('data.yaml'));names=cfg['names'];assert cfg['nc']==len(names)==35
        source_to_target={i:TARGETS.index('orange' if name=='Green_Orange' else name.lower())
                          for i,name in enumerate(names) if name.lower() in TARGETS or name=='Green_Orange'}
        images=sorted(m.filename for m in members if m.filename.lower().endswith('.jpg'))
        labels={m.filename for m in members if '/labels/' in m.filename and m.filename.endswith('.txt')}
        assert {n.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt' for n in images}==labels
        parsed={};geometry_errors=[];family_splits=defaultdict(set);class_counts=Counter();class_images=Counter();family_classes=defaultdict(set)
        for name in images:
            label=name.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'
            entries=[];seen=set()
            for line in z.read(label).decode('utf-8').splitlines():
                values=line.split()
                try:
                    assert len(values)==5
                    c=int(values[0]);x,y,w,h=map(float,values[1:]);assert 0<=c<35
                    assert all(math.isfinite(v) for v in [x,y,w,h])
                    assert 0<x<1 and 0<y<1 and 0<w<=1 and 0<h<=1
                    assert x-w/2>=-1e-6 and x+w/2<=1+1e-6 and y-h/2>=-1e-6 and y+h/2<=1+1e-6
                    entries.append((c,x,y,w,h));seen.add(c);class_counts[name.split('/')[0],c]+=1
                except Exception:geometry_errors.append({'member':label,'line':line})
            family=Path(name).stem.split('.rf.')[0]
            family_splits[family].add(name.split('/')[0]);family_classes[family].update(seen)
            for c in seen:class_images[name.split('/')[0],c]+=1
            parsed[name]=(entries,family)
        # Compare raw exact hashes and canonical pHashes to the previous experiment.
        baseline=PROJECT/'detector_training/yolov8n-fruit9-v1/reports/split_manifest.csv'
        with baseline.open(encoding='utf-8',newline='') as f:prior=list(csv.DictReader(f))
        prior_sha=defaultdict(list);prior_phash=defaultdict(list)
        for row in prior:
            prior_sha[row['source_sha256']].append((row['image_id'],row['split']))
            prior_phash[row['phash']].append((row['image_id'],row['split']))
        def inspect_one(name):
            data=z.read(name) # ZIP CRC is checked by zipfile.
            entries,family=parsed[name]
            with Image.open(io.BytesIO(data)) as im:
                width,height=im.size;orientation=im.getexif().get(274,1)
                im.draft('RGB',(64,64));im.load()
                a=np.asarray(im.convert('L').resize((32,32)),dtype=np.float32)
            low=dctn(a,type=2,norm='ortho')[:8,:8];value=0
            for bit in (low>np.median(low.flatten()[1:])).flatten():value=(value<<1)|int(bit)
            sha=hashlib.sha256(data).hexdigest();phash=f'{value:016x}'
            exact=prior_sha.get(sha,[]);perceptual=prior_phash.get(phash,[])
            target_boxes=[(source_to_target[c],x,y,w,h) for c,x,y,w,h in entries if c in source_to_target]
            record={'archive_member':name,'source_split':name.split('/')[0],'source_family':family,
                    'sha256':sha,'phash':phash,'width':width,'height':height,'orientation':orientation,
                    'source_box_count':len(entries),'target_box_count':len(target_boxes),
                    'target_fruits':'|'.join(sorted({TARGETS[b[0]] for b in target_boxes})),
                    'baseline_exact_matches':'|'.join(i+':'+s for i,s in exact),
                    'baseline_exact_phash_matches':'|'.join(i+':'+s for i,s in perceptual),
                    'project_split':'unassigned','label_review':'publisher_labels_not_fully_reviewed'}
            if target_boxes:
                # Short stable names avoid Windows MAX_PATH failures on long web filenames.
                stem=hashlib.sha256(name.encode('utf-8')).hexdigest()[:24]
                image_path=ROOT/'prepared/images'/f'{name.split("/")[0]}__{stem}.jpg'
                label_path=ROOT/'prepared/labels'/f'{name.split("/")[0]}__{stem}.txt'
                image_path.parent.mkdir(exist_ok=True);label_path.parent.mkdir(exist_ok=True)
                with Image.open(io.BytesIO(data)) as im:
                    im=im.convert('RGB');im.thumbnail((640,640),Image.Resampling.LANCZOS);im.save(image_path,quality=92,subsampling=0)
                label_path.write_text('\n'.join(f'{c} {x:.8f} {y:.8f} {w:.8f} {h:.8f}' for c,x,y,w,h in target_boxes)+'\n',encoding='utf-8')
                record.update(prepared_image=image_path.relative_to(ROOT).as_posix(),prepared_label=label_path.relative_to(ROOT).as_posix())
            else:record.update(prepared_image='',prepared_label='')
            return record
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            records=[]
            for record in executor.map(inspect_one,images):
                records.append(record)
                if len(records)%1000==0:print('Audited',len(records),'of',len(images),flush=True)
        csv_write('reports/images.csv',records)
        all_classes=[]
        for c,name in enumerate(names):
            all_classes.append({'source_class_id':c,'source_class':name,
                'mapped_target':TARGETS[source_to_target[c]] if c in source_to_target else '',
                **{f'{split}_images':class_images[split,c] for split in ['train','valid','test']},
                **{f'{split}_boxes':class_counts[split,c] for split in ['train','valid','test']},
                'source_families':sum(c in cs for cs in family_classes.values())})
        csv_write('reports/class_counts.csv',all_classes)
        duplicates=defaultdict(list)
        for r in records:duplicates[r['sha256']].append(r['archive_member'])
        exact_groups=[v for v in duplicates.values() if len(v)>1]
        overlaps={k:sorted(v) for k,v in family_splits.items() if len(v)>1}
        (ROOT/'reports/duplicate_groups.json').write_text(json.dumps(exact_groups,indent=2))
        (ROOT/'reports/family_split_overlaps.json').write_text(json.dumps(overlaps,indent=2))
        (ROOT/'reports/geometry_errors.json').write_text(json.dumps(geometry_errors,indent=2))
        selected=[r for r in records if r['target_box_count']]
        summary={'archive':str(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,'archive_sha256':digest.hexdigest(),
                 'actual_images':len(images),'publisher_readme_images':6805,
                 'source_split_images':dict(Counter(r['source_split'] for r in records)),
                 'source_classes':35,'source_boxes':sum(class_counts.values()),'geometry_errors':len(geometry_errors),
                 'source_families':len(family_splits),'families_crossing_source_splits':len(overlaps),
                 'exact_duplicate_groups':len(exact_groups),'exact_duplicate_copies':sum(len(v)-1 for v in exact_groups),
                 'target_images':len(selected),'target_boxes':sum(r['target_box_count'] for r in selected),
                 'target_source_families':len({r['source_family'] for r in selected}),
                 'target_classes':sorted({f for r in selected for f in r['target_fruits'].split('|')}),
                 'missing_nine_class_target':['lemon'],'missing_thirteen_class_targets':['guava','jujube','lemon','lulo','tamarillo'],
                 'baseline_exact_overlap_images':sum(bool(r['baseline_exact_matches']) for r in records),
                 'baseline_exact_phash_overlap_images':sum(bool(r['baseline_exact_phash_matches']) for r in records),
                 'training_ready':False,'note':'All project splits unassigned. Augmented family grouping, near-duplicate/crop overlap checks, and label review required.'}
        (ROOT/'reports/audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        # Preview every target class without claiming full manual review.
        samples=[]
        for fruit in sorted({f for r in selected for f in r['target_fruits'].split('|')}):
            options=sorted((r for r in selected if fruit in r['target_fruits'].split('|')),key=lambda r:r['sha256'])
            families=set()
            for r in options:
                if r['source_family'] in families:continue
                samples.append((fruit,r));families.add(r['source_family'])
                if len(families)==3:break
        canvas=Image.new('RGB',(1200,((len(samples)+3)//4)*245),'white');draw=ImageDraw.Draw(canvas)
        for i,(fruit,r) in enumerate(samples):
            with Image.open(ROOT/r['prepared_image']) as im:tile=ImageOps.contain(im.convert('RGB'),(295,215))
            d=ImageDraw.Draw(tile)
            for c,x,y,w,h in parsed[r['archive_member']][0]:
                if c in source_to_target:d.rectangle(((x-w/2)*tile.width,(y-h/2)*tile.height,(x+w/2)*tile.width,(y+h/2)*tile.height),outline='lime',width=2)
            x=(i%4)*300;y=(i//4)*245;canvas.paste(tile,(x,y+25));draw.text((x+3,y+3),fruit+' '+r['source_split'],fill='black')
        canvas.save(ROOT/'review/annotation_samples.jpg')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
