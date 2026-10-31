"""Run preparation, training, evaluation and integrity verification in sequence."""
from pathlib import Path
import subprocess,sys,json,time,csv
RUN=Path(__file__).resolve().parent
def main():
    start=time.time()
    for script in ['prepare_data.py','train_detector.py','evaluate_detector.py','verify_experiment.py']:
        (RUN/'pipeline_status.json').write_text(json.dumps(dict(stage=script,completed=False)))
        with (RUN/'logs'/('train.log' if script=='train_detector.py' else Path(script).stem+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u',str(RUN/script)],stdout=log,stderr=subprocess.STDOUT,cwd=RUN.parents[1])
        if result.returncode:raise RuntimeError(script+' failed; see logs')
    # Compare both checkpoints on the unchanged original test set.
    import os
    os.environ['YOLO_CONFIG_DIR']=str(RUN/'runtime_config')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model=YOLO(str(RUN/'runs/train/weights/best.pt'))
    metrics=model.val(data=str(RUN.parent/'yolov8n-fruit9-v1/data.yaml'),split='test',imgsz=416,batch=8,device=0,workers=2,half=False,conf=.001,iou=.7,max_det=300,plots=False,project=str(RUN/'runs'),name='original_test',exist_ok=True)
    old=json.loads((RUN.parent/'yolov8n-fruit9-v1/reports/test_evaluation.json').read_text())
    comparison=dict(note='Same original frozen test set, input416. New Bohni groups touching original holdouts were excluded by family/transform hash checks. Crop/physical-fruit independence is not guaranteed.',v1=old['metrics'],v2={k:float(v) for k,v in metrics.results_dict.items()})
    (RUN/'reports/original_test_comparison.json').write_text(json.dumps(comparison,indent=2))
    e=json.loads((RUN/'reports/test_evaluation.json').read_text());m=json.loads((RUN/'artifacts/model.json').read_text());a=json.loads((RUN/'reports/data_audit.json').read_text())
    history=list(csv.DictReader((RUN/'runs/train/results.csv').open()))
    table='\n'.join('| '+r['fruit']+' | '+str(r['test_images'])+' | '+str(r['test_boxes'])+' | '+f"{r['AP50']*100:.2f}%"+' |' for r in e['per_class'])
    text=f'''# YOLOv8n nine-fruit detector v2

Detects nine fruit identities and boxes. Freshness is a separate model.
Trained on NVIDIA GTX 1650 CUDA from original COCO YOLOv8n weights, not from the v1 fine-tuned model. Input416, batch8, up to60 epochs, patience12, seed42. See reports/training_config.json for exact settings.

Prepared splits: {a['splits']}. Original v1 splits frozen. Bohni filename families and rotation/reflection pHash components kept together. One representative view per new held-out component. Raw annotations have not all been manually verified. Physical-fruit and crop independence are not established.

Completed epochs: {len(history)}. Combined test: {e['images']} images, {e['boxes']} boxes.
Metrics: {json.dumps(e['metrics'])}

| Fruit | Test images | Test boxes | AP50 |
|---|---:|---:|---:|
{table}

Original frozen test comparison: {json.dumps(comparison)}

Final weights and export: artifacts/. Hashes, labels, size and preprocessing: artifacts/model.json. Parameters: {m['parameters']}.
Logs: logs/train.log. Losses and epoch metrics: runs/train/results.csv. Best and latest checkpoints: runs/train/weights/best.pt and last.pt.

This is same-source experimental evaluation. Desktop timings are not Android latency. No Android app was changed. Classic and v1 preserved. Lemon remains sparse; evaluate real camera photos before deployment. Source attribution: datasets/bohni-v13/metadata and datasets/fruit13/sources. Existing source image provenance limitations still apply.
'''
    (RUN/'MODEL_CARD.md').write_text(text,encoding='utf-8')
    with (RUN/'logs/classic_verification.log').open('w') as log:
        subprocess.run([sys.executable,str(RUN.parents[1]/'apps/01-fruit-lens-classic/verify_snapshot.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    (RUN/'pipeline_status.json').write_text(json.dumps(dict(stage='complete',completed=True,elapsed_seconds=time.time()-start),indent=2))
    print('PIPELINE_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        (RUN/'pipeline_failure.json').write_text(json.dumps(dict(error=str(exc),time=time.time()),indent=2))
        raise
