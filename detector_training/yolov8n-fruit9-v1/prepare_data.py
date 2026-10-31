"""Prepare a compact nine-fruit detection experiment without changing raw data."""
from pathlib import Path
from collections import defaultdict, Counter
import csv, json, hashlib, sys, random, concurrent.futures
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import yaml
from scipy.fft import dctn

RUN=Path(__file__).resolve().parent
PROJECT=RUN.parents[1]
SOURCE=PROJECT/'datasets/fruit13'
NAMES=['apple','banana','grape','lemon','mango','orange','pomegranate','strawberry','tomato']
sys.path.insert(0,str(SOURCE/'scripts'))
from prepare_collection import group_rows


def read(name):
    with (SOURCE/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def dump_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def split_groups(images):
    groups=defaultdict(list)
    for r in images:groups[r['group_id']].append(r)
    keys=sorted(groups)
    vectors=np.array([[sum(fruit in r['fruits'] for r in groups[k]) for fruit in NAMES]+[len(groups[k])] for k in keys],float)
    totals=vectors.sum(0)
    targets=np.array([.70,.15,.15])[:,None]*totals
    best=None
    frequencies=(vectors[:,:9]>0).sum(0)
    if np.any(frequencies<3):
        raise ValueError('Fewer than three independent hash groups for a class: '+str(dict(zip(NAMES,frequencies.tolist()))))
    for attempt in range(100):
        rng=random.Random(42+attempt)
        assigned={};used=np.zeros_like(targets)
        # Ensure each class occurs in each split without breaking any hash group.
        for c in np.argsort(frequencies):
            for s in [1,2,0]:
                if used[s,c]>0:continue
                options=[i for i in range(len(keys)) if i not in assigned and vectors[i,c]>0]
                rng.shuffle(options)
                if not options:break
                i=min(options,key=lambda i:vectors[i,-1])
                assigned[i]=s;used[s]+=vectors[i]
        order=[i for i in range(len(keys)) if i not in assigned]
        rng.shuffle(order)
        order.sort(key=lambda i:(min(frequencies[c] for c in range(9) if vectors[i,c]>0),-vectors[i,-1]))
        for i in order:
            costs=[]
            for s in range(3):
                before=((used[s]-targets[s])**2/np.maximum(targets[s],1)).sum()
                after=((used[s]+vectors[i]-targets[s])**2/np.maximum(targets[s],1)).sum()
                costs.append(after-before)
            s=int(np.argmin(costs));assigned[i]=s;used[s]+=vectors[i]
        score=(((used-targets)/np.maximum(targets,1))**2).sum()+1e6*np.sum(used[:,:9]==0)
        if best is None or score<best[0]:best=(score,assigned.copy(),used.copy())
    mapping={keys[i]:['train','val','test'][s] for i,s in best[1].items()}
    for row in images:row['split']=mapping[row['group_id']]
    return {split:{'images_with_fruit':dict(zip(NAMES,best[2][s,:9].astype(int).tolist())),
                   'images':int(best[2][s,-1])} for s,split in enumerate(['train','val','test'])},dict(zip(NAMES,frequencies.astype(int).tolist()))


def group_transforms(images):
    variants=[]
    for row in images:
        with Image.open(SOURCE/row['path']) as im:
            im.draft('RGB',(64,64))
            gray=np.asarray(im.convert('L').resize((32,32)),dtype=np.float32)
        for mirrored in [False,True]:
            for rotation in range(4):
                transformed=np.rot90(np.fliplr(gray) if mirrored else gray,rotation)
                low=dctn(transformed,type=2,norm='ortho')[:8,:8]
                value=0
                for bit in (low>np.median(low.flatten()[1:])).flatten():value=(value<<1)|int(bit)
                variants.append({'image_id':row['image_id'],'path':row['path'],'label':'detection',
                                 'decode_ok':True,'sha256':row['sha256'],'phash':f'{value:016x}'})
    group_rows(variants)
    for i,row in enumerate(images):row['group_id']=variants[i*8]['group_id']
    print('Rotation/reflection-aware groups',len({r['group_id'] for r in images}),flush=True)


def main():
    (RUN/'reports').mkdir(exist_ok=True)
    inventory=read('detection/boxed_images.csv')+read('detection/openimages_images.csv')
    index={r['image_id']:r for r in inventory}
    boxes=defaultdict(list);excluded=[]
    oid=read('detection/openimages_boxes.csv')
    bad={r['image_id'] for r in oid if r['is_group_of']=='1' or r['is_depiction']=='1'}
    for r in inventory:
        if r['image_id'] in bad:excluded.append({'image_id':r['image_id'],'reason':'source_image_has_target_group_or_depiction_box'})
    for r in read('detection/publisher_boxes.csv'):
        if r['box_valid']!='True' or r['difficult']=='1':raise ValueError('Unexpected invalid/difficult VOC box')
        w,h=float(r['width']),float(r['height'])
        coords=[float(r[k])/dim for k,dim in [('xmin',w),('ymin',h),('xmax',w),('ymax',h)]]
        boxes[r['image_id']].append((NAMES.index(r['fruit']),*coords))
    for r in oid:
        if r['image_id'] in bad:continue
        boxes[r['image_id']].append((NAMES.index(r['fruit']),*[float(r[k]) for k in ['xmin_normalized','ymin_normalized','xmax_normalized','ymax_normalized']]))
    candidates=[]
    for image_id in sorted(boxes):
        r=dict(index[image_id]);r['boxes']=sorted(set(boxes[image_id]));r['fruits']=sorted({NAMES[b[0]] for b in r['boxes']})
        r.update(label='detection',decode_ok=True)
        candidates.append(r)
    # Identical pixels with inconsistent annotations must not silently become truth.
    exact=defaultdict(list)
    for r in candidates:exact[r['sha256']].append(r)
    images=[]
    for items in exact.values():
        signatures={tuple(tuple(round(v,4) if isinstance(v,float) else v for v in b) for b in r['boxes']) for r in items}
        if len(signatures)>1:
            excluded.extend({'image_id':r['image_id'],'reason':'exact_pixels_with_inconsistent_annotations'} for r in items)
        else:
            images.append(items[0])
            excluded.extend({'image_id':r['image_id'],'reason':'exact_duplicate_copy'} for r in items[1:])
    images.sort(key=lambda r:r['image_id'])
    group_transforms(images)
    split_summary,frequencies=split_groups(images)
    def process(r):
        p=SOURCE/r['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        with Image.open(p) as im:
            if im.getexif().get(274,1)!=1:raise ValueError('Unresolved EXIF box orientation')
            if im.size!=(int(r['width']),int(r['height'])):raise ValueError('Dimension mismatch')
            im=im.convert('RGB');im.thumbnail((640,640),Image.Resampling.LANCZOS)
            dest=RUN/'data/images'/r['split']/(r['image_id']+'.jpg');dest.parent.mkdir(parents=True,exist_ok=True)
            im.save(dest,quality=93,subsampling=0)
            new_w,new_h=im.size
        labels=RUN/'data/labels'/r['split']/(r['image_id']+'.txt');labels.parent.mkdir(parents=True,exist_ok=True)
        lines=[]
        for c,x1,y1,x2,y2 in r['boxes']:
            assert 0<=x1<x2<=1 and 0<=y1<y2<=1
            lines.append(f'{c} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}')
        labels.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return {'image_id':r['image_id'],'source':r['source'],'source_path':r['path'],
                'source_sha256':r['sha256'],'phash':r['phash'],'group_id':r['group_id'],'split':r['split'],
                'fruits':'|'.join(r['fruits']),'box_count':len(r['boxes']),
                'image_path':dest.relative_to(RUN).as_posix(),'label_path':labels.relative_to(RUN).as_posix(),
                'width':new_w,'height':new_h,'prepared_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:manifest=list(executor.map(process,images))
    dump_csv(RUN/'reports/split_manifest.csv',manifest)
    dump_csv(RUN/'reports/excluded_images.csv',excluded,['image_id','reason'])
    config={'path':str((RUN/'data').resolve()),'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(NAMES))}
    (RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    boxcounts={split:dict(Counter(NAMES[b[0]] for r in images if r['split']==split for b in r['boxes'])) for split in ['train','val','test']}
    for split in ['train','val','test']:
        for other in ['train','val','test']:
            if other==split:continue
            for key in ['group_id','source_sha256']:
                assert not ({r[key] for r in manifest if r['split']==split}&{r[key] for r in manifest if r['split']==other})
    report={'names':NAMES,'input_images':len(inventory),'accepted_images':len(images),'excluded_images':len(excluded),
            'exclusion_reasons':dict(Counter(r['reason'] for r in excluded)),'splits':split_summary,'boxes_by_split':boxcounts,
            'hash_groups_per_class':frequencies,'seed':42,'phash_radius':6,'rotation_reflection_variants':8,
            'no_sha256_overlap':True,'no_phash_group_overlap':True,
            'prepared_image_long_side':640,'manifest_sha256':hashlib.sha256((RUN/'reports/split_manifest.csv').read_bytes()).hexdigest(),
            'limitations':['Physical fruit/session IDs unavailable','Same-source test, not independent phone evaluation',
                           'Mango and pomegranate have very few independent images','GitHub source image provenance unresolved',
                           'Source VOC endpoints treated as continuous supplied coordinates, no blanket one-pixel shift']}
    (RUN/'reports/data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Inspect representative boxes before allowing training.
    chosen=[]
    for fruit in NAMES:
        subset=[r for r in images if fruit in r['fruits']]
        chosen.extend(subset[:2])
    canvas=Image.new('RGB',(1000,((len(chosen)+3)//4)*230),'white');draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(chosen):
        with Image.open(SOURCE/r['path']) as im:tile=ImageOps.contain(im.convert('RGB'),(245,200))
        d=ImageDraw.Draw(tile)
        for c,x1,y1,x2,y2 in r['boxes']:
            d.rectangle((x1*tile.width,y1*tile.height,x2*tile.width,y2*tile.height),outline='lime',width=2)
        x=(i%4)*250;y=(i//4)*230;canvas.paste(tile,(x,y+25));draw.text((x+2,y+2),r['image_id']+' '+','.join(r['fruits']),fill='black')
    canvas.save(RUN/'reports/annotation_samples.jpg')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
