"""Combine frozen v1 splits and family-grouped Bohni data using compact prepared copies."""
from pathlib import Path
from collections import defaultdict, Counter
import csv,json,hashlib,sys,os,importlib.util,shutil
import numpy as np
from PIL import Image
from scipy.fft import dctn
import yaml
RUN=Path(__file__).resolve().parent
PROJECT=RUN.parents[1]
OLD=RUN.parent/'yolov8n-fruit9-v1'
BOHNI=PROJECT/'datasets/bohni-v13'
NAMES=['apple','banana','grape','lemon','mango','orange','pomegranate','strawberry','tomato']
sys.path.insert(0,str(PROJECT/'datasets/fruit13/scripts'))
from prepare_collection import group_rows

def read(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    (RUN/'reports').mkdir(exist_ok=True)
    old=read(OLD/'reports/split_manifest.csv');new=read(BOHNI/'reports/candidates.csv')
    rows=[]
    for r in old:
        rows.append(dict(r,origin='v1',family='old:'+r['group_id'],image=str(OLD/r['image_path']),label=str(OLD/r['label_path'])))
    for r in new:
        rows.append(dict(image_id='bohni_'+Path(r['prepared_image']).stem,source='bohni-v13',source_path=r['archive_member'],source_sha256=r['sha256'],phash=r['phash'],origin='bohni',family='bohni:'+r['source_family'],split='unassigned',fruits=r['target_fruits'],box_count=r['target_box_count'],image=str(BOHNI/r['prepared_image']),label=str(BOHNI/r['prepared_label'])))
    # Hash every rotation/reflection. SHA links all eight views back to an image.
    variants=[]
    for i,r in enumerate(rows):
        with Image.open(r['image']) as im:gray=np.asarray(im.convert('L').resize((32,32)),dtype=np.float32)
        for flip in [False,True]:
            for rotation in range(4):
                low=dctn(np.rot90(np.fliplr(gray) if flip else gray,rotation),type=2,norm='ortho')[:8,:8]
                value=0
                for bit in (low>np.median(low.flatten()[1:])).flatten():value=(value<<1)|int(bit)
                variants.append(dict(sha256=r['source_sha256'],phash=f'{value:016x}',label='detection',decode_ok=True))
    group_rows(variants)
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(b)]=find(a)
    seen={}
    for i,r in enumerate(rows):
        for key in [r['family'],'visual:'+variants[i*8]['group_id']]:
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(r)
    fresh=[];accepted=[];excluded=[]
    for root,items in groups.items():
        prior={r['split'] for r in items if r['origin']=='v1'}
        # Original v1 grouping was computed before resizing; preserve its frozen assignments.
        for r in items:
            if r['origin']=='v1':accepted.append(r)
        additions=[r for r in items if r['origin']=='bohni']
        if prior & {'val','test'}:
            excluded.extend(dict(image_id=r['image_id'],reason='family_or_transform_overlap_with_frozen_holdout') for r in additions)
            continue
        for r in additions:r['group_id']='bohni_component_'+str(root)
        if prior:
            for r in additions:r['split']='train';r['group_id']=next(r0['group_id'] for r0 in items if r0['origin']=='v1')
            accepted.extend(additions)
        else:fresh.extend(additions)
    spec=importlib.util.spec_from_file_location('oldprep',OLD/'prepare_data.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.NAMES=[n for n in NAMES if n!='lemon']
    for r in fresh:r['fruits']=r['fruits'].split('|')
    module.split_groups(fresh)
    bygroup=defaultdict(list)
    for r in fresh:bygroup[r['group_id']].append(r)
    for items in bygroup.values():
        items.sort(key=lambda r:r['source_sha256'])
        for j,r in enumerate(items):
            r['fruits']='|'.join(r['fruits'])
            if r['split']!='train' and j:
                excluded.append(dict(image_id=r['image_id'],reason='extra_augmented_view_of_heldout_component'));continue
            accepted.append(r)
    manifest=[]
    for r in accepted:
        image=RUN/'data/images'/r['split']/(r['image_id']+'.jpg');label=RUN/'data/labels'/r['split']/(r['image_id']+'.txt')
        for src,dest in [(r['image'],image),(r['label'],label)]:
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(src,dest)
        with Image.open(image) as im:w,h=im.size
        manifest.append({k:r[k] for k in ['image_id','source','source_path','source_sha256','phash','group_id','split','fruits','box_count']}|dict(image_path=image.relative_to(RUN).as_posix(),label_path=label.relative_to(RUN).as_posix(),width=w,height=h,prepared_sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
    write(RUN/'reports/split_manifest.csv',manifest)
    if excluded:write(RUN/'reports/excluded_images.csv',excluded)
    for a in ['train','val','test']:
        for b in ['train','val','test']:
            if a==b:continue
            for key in ['group_id','source_sha256']:assert not ({r[key] for r in manifest if r['split']==a}&{r[key] for r in manifest if r['split']==b})
    config=dict(path=str(RUN/'data'),train='images/train',val='images/val',test='images/test',names=dict(enumerate(NAMES)))
    (RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    audit=dict(names=NAMES,accepted_images=len(manifest),excluded_images=len(excluded),splits=dict(Counter(r['split'] for r in manifest)),manifest_sha256=hashlib.sha256((RUN/'reports/split_manifest.csv').read_bytes()).hexdigest(),frozen_v1_splits_preserved=True,heldout_bohni_views_per_component=1,phash_radius=6,rotation_reflection_variants=8,storage='Compact copies of existing prepared files',limitations=['Physical fruit/session IDs unknown; pHash does not guarantee crop independence','Publisher annotations used as supplied; incomplete labels and stock/composite images occur','Same-source evaluation, not independent phone evaluation','Lemon support remains sparse'])
    (RUN/'reports/data_audit.json').write_text(json.dumps(audit,indent=2))
    for folder in [RUN,RUN/'reports',RUN/'logs',RUN/'pretrained',RUN/'runs',RUN/'artifacts',RUN/'data',RUN/'data/images',RUN/'data/labels']:
        folder.mkdir(exist_ok=True)
        (folder/'README.md').write_text('YOLOv8n nine-fruit detector, experiment v2. '+{'reports':'Dataset audits, evaluation and verification.','logs':'Live training and pipeline logs.','pretrained':'Original COCO starting weights, reused from v1.','runs':'Epoch losses, charts and best/last checkpoints.','artifacts':'Final detector weights, export and class/preprocessing metadata.','data':'Compact prepared data copies; original data unchanged.','images':'RGB images; train/val/test folders keep evaluation separate.','labels':'YOLO TXT: class, normalized center x/y, width and height.'}.get(folder.name,'Bohni eight-fruit subset plus frozen nine-fruit v1 data. CPU/GPU settings and measured results are recorded in reports. This detector does not predict freshness.'))
    for folder in (RUN/'data').glob('*/*'):
        if folder.is_dir():(folder/'README.md').write_text('This split contains copied prepared data. Training learns from train only; val selects best checkpoint; test is used after training. Original source files remain unchanged.')
    print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()

