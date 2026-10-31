"""Final evidence checks for this experiment, not an accuracy guarantee."""
from pathlib import Path
from collections import Counter
import csv,json,hashlib

RUN=Path(__file__).resolve().parent
PROJECT=RUN.parents[1]


def main():
    def load(name):return json.loads((RUN/name).read_text(encoding='utf-8'))
    with (RUN/'reports/split_manifest.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    env=load('reports/environment.json');audit=load('reports/data_audit.json')
    manifest_hash=hashlib.sha256((RUN/'reports/split_manifest.csv').read_bytes()).hexdigest()
    assert manifest_hash==env['split_manifest_sha256']==audit['manifest_sha256'],'Split changed since training'
    counts=Counter()
    for row in rows:
        assert hashlib.sha256((RUN/row['image_path']).read_bytes()).hexdigest()==row['prepared_sha256']
        lines=(RUN/row['label_path']).read_text().splitlines()
        assert len(lines)==int(row['box_count'])
        for line in lines:
            values=line.split();assert len(values)==5
            c=int(values[0]);x,y,w,h=map(float,values[1:]);assert 0<=c<9
            assert 0<x<1 and 0<y<1 and 0<w<=1 and 0<h<=1
            assert x-w/2>=-1e-7 and x+w/2<=1+1e-7 and y-h/2>=-1e-7 and y+h/2<=1+1e-7
            counts[(row['split'],c)]+=1
    for first in ['train','val','test']:
        for second in ['train','val','test']:
            if first==second:continue
            for key in ['source_sha256','group_id']:
                assert not ({r[key] for r in rows if r['split']==first}&{r[key] for r in rows if r['split']==second})
    metadata=load('artifacts/model.json');evaluation=load('reports/test_evaluation.json')
    assert len(metadata['names'])==9 and len(evaluation['per_class'])==9
    assert evaluation['images']==sum(r['split']=='test' for r in rows)
    assert evaluation['boxes']==sum(v for (split,c),v in counts.items() if split=='test')
    for name,record in metadata['artifacts'].items():
        path=RUN/'artifacts'/name
        assert path.stat().st_size==record['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    assert load('reports/export_parity.json')['passed']
    protected={
      'apps/01-fruit-lens-classic/source/android/app/src/main/assets/model/freshness.tflite':'e09307ca0aacdfc83666ca8ff25c02659d6323a1372409b390561619778b10a4',
      'model_exports/mobilenetv3small-sixclass-v1/freshness.tflite':'dd519afd78d27d731fc4cef0e9a02b96086e42122a51a768874b8b6fdfc9052e'}
    for name,expected in protected.items():assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==expected
    report={'passed':True,'verified_prepared_images':len(rows),'all_nine_classes_evaluated':True,
            'manifest_unchanged_since_training':True,'sha256_and_transform_group_split_disjoint':True,
            'artifact_hashes_passed':True,'TorchScript_parity_passed':True,
            'existing_classifier_hashes_unchanged':True,'production_accuracy_validated':False}
    (RUN/'reports/final_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
