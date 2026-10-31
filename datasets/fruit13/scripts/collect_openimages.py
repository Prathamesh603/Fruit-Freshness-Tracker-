"""Collect a bounded Open Images validation subset with original box labels."""
from collections import Counter, defaultdict
from pathlib import Path
import concurrent.futures
import csv
import json
from download_sources import download
from prepare_collection import ROOT, FRUITS, inspect_image, write_csv, rel


def main():
    source=ROOT/'sources/openimages'
    with (source/'class-descriptions-boxable.csv').open(encoding='utf-8',newline='') as f:
        labels={mid:name.lower() for mid,name in csv.reader(f) if name.lower() in FRUITS}
    with (source/'validation-annotations-bbox.csv').open(encoding='utf-8',newline='') as f:
        boxes=[r for r in csv.DictReader(f) if r['LabelName'] in labels]
    ids={r['ImageID'] for r in boxes}
    with (source/'validation-images-with-rotation.csv').open(encoding='utf-8',newline='') as f:
        metadata={r['ImageID']:r for r in csv.DictReader(f) if r['ImageID'] in ids}
    write_csv(source/'selected_image_attribution.csv',list(metadata.values()))
    images=[];failures=[]
    def fetch(image_id):
        if image_id not in metadata:raise ValueError('Missing source attribution')
        row=metadata[image_id]
        if 'creativecommons.org/licenses/by/2.0' not in row['License']:
            raise ValueError('Unexpected image license: '+row['License'])
        path=ROOT/'raw/openimages/images'/f'{image_id}.jpg'
        url=f'https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg'
        download(url,path)
        record={'image_id':'oid_'+image_id,'source_image_id':image_id,'source':'openimages',
                'path':rel(path),'download_url':url,'original_landing_url':row['OriginalLandingURL'],
                'author':row['Author'],'license':row['License'],'source_split':'validation',
                'split':'unassigned','review_status':'awaiting_visual_review',
                'condition':'not_annotated','source_rotation':row['Rotation']}
        record.update(inspect_image(path))
        return record
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures={ex.submit(fetch,image_id):image_id for image_id in sorted(ids)}
        for future in concurrent.futures.as_completed(futures):
            image_id=futures[future]
            try:images.append(future.result())
            except Exception as error:failures.append({'image_id':image_id,'error':str(error)})
            if (len(images)+len(failures))%50==0:print('Open Images',len(images),'downloaded',len(failures),'failed',flush=True)
    images.sort(key=lambda r:r['image_id'])
    write_csv(ROOT/'detection/openimages_images.csv',images)
    available={r['source_image_id']:r for r in images}
    records=[]
    for box in boxes:
        image=available.get(box['ImageID'])
        if not image:continue
        xmin,xmax,ymin,ymax=[float(box[k]) for k in ['XMin','XMax','YMin','YMax']]
        if not (0<=xmin<xmax<=1 and 0<=ymin<ymax<=1):
            failures.append({'image_id':box['ImageID'],'error':'Invalid normalized box'});continue
        record={'image_id':image['image_id'],'path':image['path'],'fruit':labels[box['LabelName']],
                'xmin_normalized':xmin,'ymin_normalized':ymin,'xmax_normalized':xmax,'ymax_normalized':ymax,
                'width':image['width'],'height':image['height'],'source_label_id':box['LabelName'],
                'is_group_of':box['IsGroupOf'],'is_depiction':box['IsDepiction'],
                'is_occluded':box['IsOccluded'],'is_truncated':box['IsTruncated'],
                'annotation_origin':box['Source'],'source_split':'validation','split':'unassigned',
                'condition':'not_annotated'}
        records.append(record)
    write_csv(ROOT/'detection/openimages_boxes.csv',records)
    summary={'expected_images':len(ids),'downloaded_images':len(images),'boxes':len(records),
             'fruit_box_counts':dict(Counter(r['fruit'] for r in records)),
             'missing_target_fruits':sorted(set(FRUITS)-set(labels.values())),
             'failures':failures,'training_ready':False,
             'note':'Public validation subset staged as raw data; no final project split. No freshness labels.'}
    (ROOT/'reports/openimages_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
