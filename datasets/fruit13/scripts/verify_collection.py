"""Verify file integrity and inventory consistency, without claiming label accuracy."""
from collections import Counter
from pathlib import Path
import concurrent.futures
import csv
import hashlib
import json
from prepare_collection import ROOT, FRUITS


def load(name):
    with (ROOT/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def main():
    classification=load('classification/images.csv')
    github=load('detection/boxed_images.csv')
    oid=load('detection/openimages_images.csv')
    rows=classification+github+oid
    assert len(classification)==19200,'Unexpected original classification image count'
    assert {r['fruit'] for r in classification}==set(FRUITS)
    assert len({r['label'] for r in classification})==26
    assert len({r['image_id'] for r in rows})==len(rows),'Duplicate inventory ID'
    assert len({r['path'] for r in rows})==len(rows),'Repeated physical path'
    assert all(r['split']=='unassigned' for r in rows),'Final splits must not be silently created'
    by_id={r['image_id']:r for r in rows}
    failures=[]
    def verify_image(row):
        path=(ROOT/row['path']).resolve()
        path.relative_to(ROOT.resolve())
        assert path.is_file(),'Missing file '+str(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],'Changed image '+str(path)
        assert int(row['width'])>0 and int(row['height'])>0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results=[ex.submit(verify_image,row) for row in rows]
        for future in results:
            try:future.result()
            except Exception as error:failures.append(str(error))
    box_rows=load('detection/publisher_boxes.csv')+load('detection/openimages_boxes.csv')
    for box in box_rows:
        assert box['image_id'] in by_id
        assert box['path']==by_id[box['image_id']]['path']
        assert box['fruit'] in FRUITS
    queue=load('review/missing_detection_boxes.csv')
    assert {r['fruit'] for r in queue}=={'guava','jujube','lulo','tamarillo'}
    assert all(not r[k] for r in queue for k in ['xmin','ymin','xmax','ymax'])
    undocumented=[]
    for folder in [ROOT]+[p for p in ROOT.rglob('*') if p.is_dir() and '__pycache__' not in p.parts]:
        if not (folder/'README.md').exists() and not (folder/'FOLDER_GUIDE.md').exists():undocumented.append(str(folder))
    assert not undocumented,'Undocumented folders: '+str(undocumented)
    inventory_names=['classification/images.csv','detection/boxed_images.csv','detection/publisher_boxes.csv',
                     'detection/openimages_images.csv','detection/openimages_boxes.csv','labels.json']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inventory_names}
    # Evidence-only cross-task overlap audit: actual hashes, not filename guesses.
    owners={}
    cross_task=[]
    for row in rows:
        task='classification' if row['image_id'].startswith(('fruit16k_','mendeley8_')) else 'detection'
        previous=owners.get(row['sha256'])
        if previous and previous[0]!=task:cross_task.append({'image_id':row['image_id'],'same_as':previous[1]})
        else:owners[row['sha256']]=(task,row['image_id'])
    (ROOT/'reports/cross_task_exact_overlaps.json').write_text(json.dumps(cross_task,indent=2),encoding='utf-8')
    result={'inventory_integrity_passed':not failures,'training_ready':False,
            'verified_images':len(rows),'classification_images':len(classification),
            'github_detection_images':len(github),'openimages_detection_images':len(oid),
            'source_boxes':len(box_rows),'missing_box_queue_images':len(queue),
            'image_sha256_failures':failures,'undocumented_folders':undocumented,
            'cross_task_exact_overlap_records':len(cross_task),'inventory_sha256':hashes,
            'limitations':['Human label/box review is incomplete','Physical-fruit/session grouping is incomplete',
                           'No final project splits','Four detection classes lack boxes',
                           'No independent phone evaluation','Cross-task near-duplicate grouping remains required']}
    (ROOT/'reports/verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
