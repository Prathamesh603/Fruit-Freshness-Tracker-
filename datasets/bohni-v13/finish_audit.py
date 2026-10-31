"""Finish overlap flags, candidate inventory, documentation and integrity checks."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,hashlib,shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]


def write_csv(path,rows,fields=None):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    rows=list(csv.DictReader((ROOT/'reports/images.csv').open(encoding='utf-8',newline='')))
    prior=list(csv.DictReader((PROJECT/'detector_training/yolov8n-fruit9-v1/reports/split_manifest.csv').open(encoding='utf-8',newline='')))
    values=np.array([int(r['phash'],16) for r in prior],dtype=np.uint64)
    flags=[]
    for row in rows:
        if not int(row['target_box_count']):continue
        dist=np.bitwise_count(values ^ np.uint64(int(row['phash'],16)))
        for i in np.flatnonzero(dist<=6):
            flags.append({'source_member':row['archive_member'],'source_family':row['source_family'],
                          'baseline_image_id':prior[i]['image_id'],'baseline_split':prior[i]['split'],'hamming_distance':int(dist[i])})
    write_csv(ROOT/'reports/baseline_near_overlap_candidates.csv',flags,
              ['source_member','source_family','baseline_image_id','baseline_split','hamming_distance'])
    holdout_families={r['source_family'] for r in flags if r['baseline_split'] in ['val','test']}
    overlap={'canonical_phash_radius':6,'candidate_pairs':len(flags),'candidate_images':len({r['source_member'] for r in flags}),
             'families_touching_baseline_holdouts':sorted(holdout_families),
             'note':'Potential visual overlap, not confirmed identity. This canonical pHash check does not establish crop/rotation independence.'}
    (ROOT/'reports/baseline_near_overlap.json').write_text(json.dumps(overlap,indent=2),encoding='utf-8')
    selected=[r for r in rows if int(r['target_box_count'])]
    candidates=[];held=[];seen=set()
    for r in selected:
        reason='potential_baseline_holdout_overlap_family' if r['source_family'] in holdout_families else 'exact_duplicate_copy' if r['sha256'] in seen else ''
        seen.add(r['sha256'])
        if reason:held.append(dict(r,hold_reason=reason))
        else:candidates.append(r)
    write_csv(ROOT/'reports/candidates.csv',candidates)
    write_csv(ROOT/'reports/held_images.csv',held,list(selected[0])+['hold_reason'])
    # Validate actual prepared files against the recorded image/label inventory.
    hashes=[];boxes=0
    for r in selected:
        image=ROOT/r['prepared_image'];label=ROOT/r['prepared_label']
        assert image.is_file() and label.is_file()
        lines=label.read_text().splitlines();assert len(lines)==int(r['target_box_count'])
        for line in lines:
            c,x,y,w,h=map(float,line.split());assert c.is_integer() and 0<=c<9
            assert 0<x<1 and 0<y<1 and 0<w<=1 and 0<h<=1
        boxes+=len(lines)
        for p in [image,label]:hashes.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    assert len(list((ROOT/'prepared/images').glob('*.jpg')))==len(selected)
    assert len(list((ROOT/'prepared/labels').glob('*.txt')))==len(selected)
    write_csv(ROOT/'reports/prepared_file_hashes.csv',hashes)
    a=json.loads((ROOT/'reports/audit.json').read_text())
    assert len(rows)==a['actual_images'] and boxes==a['target_boxes']
    classes=list(csv.DictReader((ROOT/'reports/class_counts.csv').open(encoding='utf-8',newline='')))
    table=[]
    for fruit in ['apple','banana','grape','lemon','mango','orange','pomegranate','strawberry','tomato']:
        rr=[r for r in selected if fruit in r['target_fruits'].split('|')]
        count=sum(int(r[f'{s}_boxes']) for r in classes if r['mapped_target']==fruit for s in ['train','valid','test'])
        table.append(f"| {fruit} | {len(rr)} | {len({r['source_family'] for r in rr})} | {count} |")
    prepared_bytes=sum(r['bytes'] for r in hashes)
    status=f'''# Bohni v13: measured audit results

## What was received

- User-supplied Roboflow YOLOv8 ZIP: {a['archive_bytes']:,} bytes.
- SHA256: `{a['archive_sha256']}`.
- Actual image/label pairs: **{len(rows):,}**; source classes: **35**; source boxes: **{a['source_boxes']:,}**.
- All image members passed the JPEG draft readability check and ZIP CRC reads;
  this does not constitute exhaustive full-resolution pixel or semantic review.
- Malformed/out-of-bounds source box records: **{a['geometry_errors']}**.
- Source splits: **6,573 train / 147 validation / 10 test images**.
- The exporter README says 6,805 images, but the archive actually contains 6,730. The reason for the difference is not established.
- Publisher metadata declares CC BY 4.0, project bohni-tech/fruits-and-vegi, version 13.

## Useful fruit subset

Prepared **{len(selected):,} image/label pairs** with **{boxes:,} boxes**, covering
eight matching fruit identities. This uses **{prepared_bytes/2**20:.1f} MiB** of file
content instead of extracting every 1280-pixel image. Original bytes remain in the ZIP.

| Fruit | Source image files | Filename families | Boxes |
|---|---:|---:|---:|
{chr(10).join(table)}

Filename families group augmented views of a source filename. They are NOT proof
of independent physical fruits or capture sessions. Mango and pomegranate coverage
is much larger than our first run, but labels and source independence still matter.

## Problems that must be handled before training

1. **Source test split is unsuitable for our fruit task:** none of its ten images
   has any of the eight matching target fruit classes. Do not report fruit accuracy
   on this split or assume the original YAML provides a useful held-out test.
2. **Augmented data:** the exporter describes three views per source image, with
   random cropping and brightness reduction. There are {a['source_families']:,}
   source-filename families across the whole archive and {a['target_source_families']}
   among selected fruit files. Keep complete families together in new project splits.
3. **Existing split leakage:** {a['families_crossing_source_splits']} source-filename
   families already cross the publisher's split folders, including ten target-fruit
   families. Rebuild splits after grouping rather than reuse the source YAML.
4. **Duplicates:** {a['exact_duplicate_groups']} exact duplicate group(s), with
   {a['exact_duplicate_copies']} additional copy/copies, were found in the archive.
5. **Potential overlap with the previous experiment:** {len(flags)} canonical pHash
   candidate pairs involving {overlap['candidate_images']} new images; {len(holdout_families)}
   source family/families touch its validation/test pool. The candidate table holds
   all members of those families. Perceptual similarity is not identity proof, and
   this check does not resolve all crops/rotations or physical-fruit overlap.
6. **Sample annotation concerns:** the preview contains stock photos, composites,
   cut fruit, possibly ambiguous citrus/berry labels, and apparently unboxed small
   fruit. These need review. Labels were not silently corrected from visual guesses.
   Grapes/banana bunches also need a consistent region-versus-individual counting policy.

After automatic holds, **{len(candidates):,} candidates** remain and **{len(held)}
images are held**, with reasons in held_images.csv. No project training split has
been assigned and no second detector has been trained from this source yet.

## Next preparation step

Combine accepted data with the prior nine-class dataset, retain lemon examples
from the prior source, group all source families/near-duplicates across both,
and protect the existing test pool. Build adequate new per-fruit validation/test
coverage using source groups; use one representative view per augmented family
for headline held-out support counts. Then train a separately named version 2.

Guava, jujube, lulo and tamarillo remain absent. This dataset does not label
freshness. Existing Classic, the first detector, and the freshness model are unchanged.
'''
    (ROOT/'reports/STATUS.md').write_text(status,encoding='utf-8')
    verification={'passed':True,'archive_images':len(rows),'prepared_pairs':len(selected),'prepared_boxes':boxes,
                  'prepared_bytes':prepared_bytes,'candidates':len(candidates),'held_images':len(held),
                  'training_ready':False,'project_splits_assigned':False,'new_training_started':False,
                  'D_free_GiB':shutil.disk_usage('D:/').free/2**30}
    (ROOT/'reports/verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    (ROOT/'prepared/labels.json').write_text(json.dumps({'names':['apple','banana','grape','lemon','mango','orange','pomegranate','strawberry','tomato'],
        'missing_in_this_source':['lemon'],'source_Green_Orange_maps_to':'orange'},indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':main()
