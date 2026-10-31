"""Build plain-language folder guides and coverage tables from actual inventories."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
from prepare_collection import ROOT, FRUITS, write_csv


def read_csv(name):
    with (ROOT/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def guide(folder, text):
    path=ROOT/folder
    path.mkdir(parents=True,exist_ok=True)
    dest=path/'README.md'
    if folder=='raw/fruit_freshness_boxes/extracted' and dest.exists():
        dest=path/'FOLDER_GUIDE.md'
    dest.write_text(text.strip()+'\n',encoding='utf-8')


def main():
    rows=read_csv('classification/images.csv')
    boxes=read_csv('detection/publisher_boxes.csv')
    oid=read_csv('detection/openimages_boxes.csv')
    summary=json.loads((ROOT/'reports/classification_audit.json').read_text())
    bsummary=json.loads((ROOT/'reports/detection_boxes_audit.json').read_text())
    osummary=json.loads((ROOT/'reports/openimages_audit.json').read_text())
    candidates=[];held=[]
    for row in rows:
        reason=('unreadable' if row['decode_ok']!='True' else
                'candidate_label_conflict' if row['group_has_label_conflict']=='True' else
                'exact_duplicate_copy' if row['exact_duplicate']=='True' else '')
        if reason:held.append(dict(row,hold_reason=reason))
        else:candidates.append(row)
    write_csv(ROOT/'classification/candidates.csv',candidates)
    write_csv(ROOT/'review/held_images.csv',held)
    labels={'schema_version':1,'fruit_names':FRUITS,
            'detection_classes':[{'id':i,'name':fruit} for i,fruit in enumerate(FRUITS)],
            'classification_classes':[{'id':i,'name':condition+'_'+fruit,'fruit':fruit,'condition':condition}
              for i,(condition,fruit) in enumerate((c,f) for c in ['fresh','spoiled'] for f in FRUITS)],
            'state':'planned taxonomy; not a deployed model','condition_labels_need_review':True}
    (ROOT/'labels.json').write_text(json.dumps(labels,indent=2),encoding='utf-8')
    missing=set(osummary['missing_target_fruits'])-set(bsummary['fruit_counts'])
    queue=[{'image_id':r['image_id'],'path':r['path'],'fruit':r['fruit'],
            'publisher_condition':r['condition'],'xmin':'','ymin':'','xmax':'','ymax':'',
            'physical_fruit_id':'','capture_session_id':'','reviewer':'','review_status':'needs_real_boxes'}
           for r in rows if r['fruit'] in missing]
    write_csv(ROOT/'review/missing_detection_boxes.csv',queue)
    review_fields=['image_id','path','decision','corrected_fruit','corrected_condition','physical_fruit_id','capture_session_id','reviewer','date','reason']
    decisions=ROOT/'review/decisions.csv'
    if not decisions.exists():write_csv(decisions,[],review_fields)
    counts=Counter((r['fruit'],r['condition']) for r in rows)
    bc=Counter(r['fruit'] for r in boxes)
    oc=Counter(r['fruit'] for r in oid)
    coverage=[{'fruit':fruit,'fresh_source_images':counts[fruit,'fresh'],'spoiled_source_images':counts[fruit,'spoiled'],
               'github_source_boxes':bc[fruit],'openimages_source_boxes':oc[fruit],
               'detection_status':'missing_boxes' if fruit in missing else 'boxes_collected_need_review',
               'training_ready':False} for fruit in FRUITS]
    write_csv(ROOT/'reports/coverage.csv',coverage)
    widths=Counter((r['source'],r['width'],r['height']) for r in rows)
    dimensions={source:{'minimum_width':min(int(r['width']) for r in rows if r['source']==source and r['decode_ok']=='True'),
                        'maximum_width':max(int(r['width']) for r in rows if r['source']==source and r['decode_ok']=='True'),
                        'minimum_height':min(int(r['height']) for r in rows if r['source']==source and r['decode_ok']=='True'),
                        'maximum_height':max(int(r['height']) for r in rows if r['source']==source and r['decode_ok']=='True')}
                for source in sorted({r['source'] for r in rows})}
    (ROOT/'reports/image_dimensions.json').write_text(json.dumps(dimensions,indent=2),encoding='utf-8')
    table='\n'.join(f"| {r['fruit']} | {r['fresh_source_images']} | {r['spoiled_source_images']} | {r['github_source_boxes']} | {r['openimages_source_boxes']} |" for r in coverage)
    status=f'''# Collection status

This is a collected and inventoried staging dataset. It is NOT a training-ready
release and no 13-fruit model has been trained from it.

## Measured files

- Classification: **{summary['images']:,} source images**, **{summary['fruits']} fruit names**, **{summary['classes']} fruit-condition combinations**.
- Classification unreadable images: **{summary['decode_failures']}** (JPEG draft readability check).
- Exact duplicate copies in classification inventory: **{summary['exact_duplicate_copies']}**.
- Perceptual candidate label conflicts: **{summary['conflict_groups']} groups / {summary['conflict_images']} images**. These need review, not automatic relabelling.
- After withholding duplicate copies and candidate conflicting groups: **{len(candidates):,} classification candidates**, with **{len(held):,} images held for review**. This is automatic staging, not human approval.
- Images overlapping the prior three-fruit experiment by SHA-256: **{summary['baseline_overlap_images']}**.
- Boxed GitHub source: **{bsummary['images']:,} readable images / {bsummary['boxes']:,} boxes**, across three fruit types; **{len(bsummary['issues'])} recorded geometry/readability issues**.
- Open Images: **{osummary['downloaded_images']} downloaded images / {osummary['boxes']:,} boxes**, across nine fruit types; **{len(osummary['failures'])} download/box failures**.
- Counts are source-file counts before deduplication, filtering, crop generation, or final split selection. They are not counts of independent physical fruit.
- No trained model, final project split, independent phone test, or all-fruit support is claimed.

## Coverage

| Fruit | Fresh images | Spoiled images | GitHub boxes | Open Images boxes |
|---|---:|---:|---:|---:|
{table}

Different sources can overlap. Do not add these columns together and call them
independent training examples. Box counts also include source group/depiction
flags that must be filtered according to a written annotation policy.

## Remaining work

1. Review condition labels and candidate conflict groups. Keep decisions separate from raw data.
2. Establish same-fruit/session groups where possible; pHash cannot prove independence.
3. Draw real fruit boxes for **{', '.join(sorted(missing))}**. The work queue contains {len(queue):,} candidate source photos with blank box fields.
4. Add independent detection photos for sparse classes; the small Open Images subset is a seed collection.
5. Resolve GitHub boxed image provenance and review the original VOC coordinate convention.
6. Deduplicate across detection, classification, and any derived crops before making fixed splits.
7. Collect phone evaluation photos; that folder is currently a documented empty collection area.

See `classification_audit.json`, `detection_boxes_audit.json`, `openimages_audit.json`,
`downloads.json`, `coverage.csv`, and `image_dimensions.json` for machine-readable evidence.
'''
    (ROOT/'reports/STATUS.md').write_text(status,encoding='utf-8')
    guide('raw','''# Original source files

These are the downloaded source archives and extracted photos. Do not edit,
resize, rename, or relabel images here. Write corrections in review/decisions.csv
and create processed copies in a separately documented release later.

fruit16k and mendeley8 contain classification originals. fruit_freshness_boxes
contains photos plus original XML boxes. openimages contains a bounded natural-
scene detection subset. None is independent phone evaluation data.
''')
    for source in ['fruit16k','mendeley8']:
        subset=[r for r in rows if r['source']==source]
        guide('raw/'+source,f'''# {source}: original classification photos

This source contributes {len(subset):,} photos in the current inventory. Original
folder labels are preserved. classification/images.csv maps every image to a
canonical fruit name, publisher condition, dimensions, checksum and review status.
No object boxes are supplied for these photos. No augmentation is applied.
Read SOURCES.md two folders above for authors, version, license and limitations.
''')
        guide('raw/'+source+'/images',f'''# Extracted {source} images

Each child folder keeps a publisher class. There are {len(subset):,} photos here.
Fresh/Rotten prefixes in mendeley8 and F_/S_ prefixes in fruit16k are source labels,
not independently verified food-safety judgments. The parent source guide and
classification inventory explain the mapping. Keep raw files unchanged.
''')
        for folder in sorted({r['source_folder'] for r in subset}):
            items=[r for r in subset if r['source_folder']==folder]
            item=items[0]
            guide(f'raw/{source}/images/{folder}',f'''# Publisher class: {folder}

This folder contains {len(items)} original photos from {source}.
Canonical fruit: {item['fruit']}. Inventory condition: {item['condition']}.
The condition comes from the publisher folder and still requires review.

These images teach fruit appearance and visible condition. They do not contain
object bounding-box labels. Several photos may show the same physical fruit.
Do not randomly split these files before checking the groups in the inventory.
Do not change the images to correct a label; record a decision in review/decisions.csv.
''')
    guide('raw/fruit_freshness_boxes','''# Downloaded boxed fruit source

publisher.zip is the pinned GitHub archive. extracted/ keeps its original images,
Pascal VOC XML annotations, README and Apache-2.0 license. Detection inventory
tables live in detection/. Original source conditions are not normalized into
binary freshness training labels. Image provenance and box quality need review.
''')
    guide('raw/fruit_freshness_boxes/extracted','''# Extracted publisher repository

JPEGImages/ contains original photos. Annotations/ contains paired XML boxes.
The publisher README and LICENSE remain unchanged. This extra guide explains
their local role; see detection/ for searchable CSV inventories and audit findings.
''')
    guide('raw/fruit_freshness_boxes/extracted/JPEGImages','''# Photos belonging to the XML boxes

These are the publisher's original detection images. Match filename stems to XML
files in ../Annotations. Read detection/boxed_images.csv for hashes/dimensions.
Do not assume this source is independent of our other downloaded datasets.
''')
    guide('raw/fruit_freshness_boxes/extracted/Annotations','''# Original Pascal VOC box annotations

Each XML gives a matching image filename, size, one or more fruit-condition names,
and pixel box endpoints. publisher_boxes.csv preserves these values. The detector
uses fruit identity; the source word normal is not automatically fresh. Review
endpoint conventions and invalid boxes before exporting YOLO or COCO labels.
''')
    guide('raw/openimages',f'''# Open Images source photos

images/ contains {osummary['downloaded_images']} downloaded photos from the source
validation split. Original bounding boxes are indexed in detection/openimages_boxes.csv.
Author and license records are in sources/openimages/selected_image_attribution.csv.
No freshness labels have been inferred. This is a small detection seed dataset.
''')
    guide('raw/openimages/images','''# Natural-scene detection images

Filenames are Open Images IDs. The detection inventory connects each photo to its
fruit boxes, source URL, author, license, dimensions, and SHA-256. Multiple target
fruits can appear together. Group and depiction flags require review. These
images are not freshness training examples without separate condition annotation.
''')
    guide('sources','''# Source records and attribution

These files explain where images and boxes came from. Read ../SOURCES.md for the
human-readable source register. Metadata-only folders do not imply that their
images were downloaded or used. Public endpoint file pages may be partial lists;
the actual downloaded-image inventories are classification/ and detection/.
''')
    guide('sources/openimages','''# Open Images original metadata

class-descriptions-boxable.csv maps machine IDs to names. validation-annotations-
bbox.csv is the original source box table, not just our selected fruits.
validation-images-with-rotation.csv records source photo information.
selected_image_attribution.csv contains author/license records for selected photos.
Only downloaded images counted in reports/openimages_audit.json are locally present.
''')
    for sourceid,title in [('6ps7gtp2wg','FRUIT-16K'),('bdd69gyhv8','eight-fruit Mendeley source'),('42m5tb7yv9','AgriFreshNET candidate')]:
        guide('sources/'+sourceid,f'''# Metadata: {title}

This directory holds publisher metadata gathered during collection. files.json or
files_first_page.json can be a partial file listing and is not proof of a complete
image download. folders.json, when present, must be an actual source folder list.
Read ../../SOURCES.md and ../../reports/STATUS.md for what was actually included.
AgriFreshNET images were not added to the collection.
''')
    guide('sources/fruit_freshness_boxes','''# Pinned GitHub source tree

tree.json records the repository tree and commit used for download. It is provenance,
not an image dataset by itself. The archive and extracted photos live in raw/.
The original source license remains in the extracted repository.
''')
    guide('classification','''# Freshness classification inventory

images.csv lists each original source photo once, with its fruit and condition,
source folder, license, original dimensions, SHA-256, pHash, candidate duplicate
group, baseline overlap, and review status. Paths are relative to fruit13/.

`candidates.csv` withholds exact duplicate copies and candidate conflicting groups.
`../review/held_images.csv` explains every hold; raw files remain available. Candidates
still need human review and are not an approved training split.

All project splits are unassigned. source labels are not yet fully reviewed.
exact_duplicate marks an identical byte copy. group_has_label_conflict marks
candidate groups with different labels; it does not automatically correct them.
This table contains full source frames, not guaranteed tight single-fruit crops.
Review composition and create derived crops only after boxes and splits are settled.
''')
    guide('detection','''# Fruit detection inventories

boxed_images.csv inventories the GitHub source photos. publisher_boxes.csv preserves
its original pixel boxes and condition names. openimages_images.csv records local
photos and attribution. openimages_boxes.csv stores normalized box endpoints and
source flags. No missing-fruit boxes have been fabricated.

All paths are relative to fruit13/. These CSV files are staging tables, not YOLO
training labels. Preserve every target box in selected images. Confirm rotation,
coordinate conventions, group policy, box validity and source rights before export.
The sources do not provide comprehensive freshness labels for these natural scenes.
''')
    guide('review','''# Human review work

REVIEW_GUIDE.md explains what to do. classification_samples.jpg and fruit16k_samples.jpg
are small sample sheets. label_conflicts.csv lists candidate cross-label groups.
missing_detection_boxes.csv is a queue with intentionally blank coordinates for
the four missing fruit types. Blank coordinates must never become training boxes.
decisions.csv is a blank decision log; no human decisions are invented.
''')
    guide('reports','''# Measured collection evidence

Start with STATUS.md. JSON audit files contain actual counts and issues; coverage.csv
shows per-fruit coverage. image_dimensions.json records original size ranges.
downloads.json records downloaded archive/metadata URLs, byte sizes and SHA-256.
verification.json is written by the final consistency/hash verifier. A passed
inventory check does not mean labels are accurate or the dataset is training-ready.
''')
    guide('scripts','''# Reproduce data preparation

download_sources.py downloads source archives and annotation metadata.
prepare_collection.py safely extracts, hashes, groups, and inventories source images.
collect_openimages.py downloads the bounded detection subset and keeps attribution.
document_collection.py writes folder guides, coverage tables and annotation queues.
verify_collection.py checks inventories against actual files and expected coverage.
requirements.txt lists Python dependencies. See ../HANDOFF.md for command order.

Downloads reuse existing files. Extraction detects changed raw files instead of
overwriting them. The scripts regenerate machine inventories/reports; they do not
modify review/decisions.csv or train any model. Do not edit a raw file in place.
''')
    # Remove only transient files created during endpoint discovery, not source data.
    for name in ['sources/mendeley_bundle.js','sources/bdd69gyhv8/folders.json','sources/42m5tb7yv9/folders.json']:
        path=ROOT/name
        if path.exists():path.unlink()
    page=ROOT/'sources/6ps7gtp2wg/files.json'
    if page.exists():
        listing=json.loads(page.read_text())
        if len(listing)<16000:
            page.rename(page.with_name('files_first_page.json'))
    print('Documentation generated; missing-box queue',len(queue),flush=True)


if __name__=='__main__':main()
