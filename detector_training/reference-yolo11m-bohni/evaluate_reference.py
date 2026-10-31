"""Restricted-load and benchmark the public reference on the existing test pool."""
from pathlib import Path
import os,json,hashlib,inspect,copy,time,sys

RUN=Path(__file__).resolve().parent
BASE=RUN.parent/'yolov8n-fruit9-v1'
os.environ['YOLO_CONFIG_DIR']=str(BASE/'runtime_config')
os.environ['MPLCONFIGDIR']=str(BASE/'runtime_config/matplotlib')
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


def main():
    image_size=int(sys.argv[1]) if len(sys.argv)>1 else 416
    if image_size not in [416,640]:raise ValueError('Only the declared common-size and native-size benchmarks are supported')
    import torch
    from ultralytics import YOLO
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block_modules
    import ultralytics.nn.modules.conv as conv_modules
    import ultralytics.nn.modules.head as head_modules
    from ultralytics.nn.tasks import DetectionModel
    provenance=json.loads((RUN/'provenance.json').read_text())
    weights=RUN/'original_best.pt'
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==provenance['sha256']
    allowed=[set,DetectionModel]
    for namespace in [torch.nn,modules,block_modules,conv_modules,head_modules]:
        allowed.extend(obj for _,obj in inspect.getmembers(namespace,inspect.isclass) if issubclass(obj,torch.nn.Module))
    # No arbitrary pickle functions/classes are allowed. No remote source is executed.
    with torch.serialization.safe_globals(allowed):
        checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
    source_model=checkpoint.get('ema') or checkpoint['model']
    assert isinstance(source_model,DetectionModel)
    source_names=source_model.names
    if isinstance(source_names,list):source_names=dict(enumerate(source_names))
    assert len(source_names)==35
    # Rebuild from the installed library's architecture, then transfer tensors only.
    trusted=YOLO('yolo11m.yaml')
    cfg=copy.deepcopy(trusted.model.yaml)
    rebuilt=DetectionModel(cfg,nc=35,verbose=False)
    rebuilt.load_state_dict(source_model.float().state_dict(),strict=True)
    rebuilt.names=source_names
    trusted.model=rebuilt
    del checkpoint,source_model
    targets=['apple','banana','grape','lemon','mango','orange','pomegranate','strawberry','tomato']
    aliases={'green_orange':'orange'}
    mappings={}
    for i,name in source_names.items():
        normalized=name.lower().replace(' ','_')
        normalized=aliases.get(normalized,normalized)
        if normalized in targets:mappings[int(i)]=targets.index(normalized)
    missing=sorted(set(targets)-{targets[i] for i in mappings.values()})
    def remap_output(module,inputs,output):
        pred=output[0] if isinstance(output,tuple) else output
        assert pred.shape[1]==39
        mapped=pred.new_zeros((pred.shape[0],13,pred.shape[2]))
        mapped[:,:4]=pred[:,:4]
        for old,new in mappings.items():mapped[:,4+new]=torch.maximum(mapped[:,4+new],pred[:,4+old])
        return (mapped,*output[1:]) if isinstance(output,tuple) else mapped
    trusted.model.model[-1].register_forward_hook(remap_output)
    trusted.model.names=dict(enumerate(targets))
    (RUN/'actual_model_metadata.json').write_text(json.dumps({'source_names':source_names,'parameters':sum(p.numel() for p in rebuilt.parameters()),
        'source_to_target':{source_names[i]:targets[j] for i,j in mappings.items()},'missing_targets':missing,
        'checkpoint_restricted_load':True,'strict_state_transfer_to_trusted_architecture':True},indent=2),encoding='utf-8')
    start=time.perf_counter()
    result=trusted.val(data=str(BASE/'data.yaml'),split='test',imgsz=image_size,batch=8,device=0,workers=2,
                       half=False,conf=0.001,iou=0.7,max_det=300,plots=True,save_json=False,
                       project=str(RUN/'runs'),name=f'test{image_size}',exist_ok=True,verbose=True)
    baseline=json.loads((BASE/'reports/test_evaluation.json').read_text())
    reference=[]
    for j,class_id in enumerate(result.box.ap_class_index):
        i=int(class_id)
        support=next(r for r in baseline['per_class'] if r['class_id']==i)
        reference.append({'fruit':targets[i],'images':support['test_images'],'boxes':support['test_boxes'],
                          'AP50':float(result.box.ap50[j]),'AP50_95':float(result.box.ap[j]),
                          'precision':float(result.box.p[j]),'recall':float(result.box.r[j])})
    shared=set(targets)-set(missing)
    def mean(rows,key):return sum(r[key] for r in rows if r['fruit'] in shared)/len(shared)
    report={'reference':'Piyu12/fruit-veg-yolo11m-detector','revision':provenance['revision'],
            'source_classes_verified':35,'weights_bytes':weights.stat().st_size,'input_size':image_size,
            'test_images':baseline['images'],'test_boxes':baseline['boxes'],'missing_target_classes':missing,
            'metrics_nine_target_scope':{k:float(v) for k,v in result.results_dict.items()},'per_class':reference,
            'shared_eight_classes':sorted(shared),
            'shared_scope':{'reference_mAP50':mean(reference,'AP50'),'reference_mAP50_95':mean(reference,'AP50_95'),
                            'trained_yolov8n_mAP50':mean(baseline['per_class'],'AP50'),
                            'trained_yolov8n_mAP50_95':mean(baseline['per_class'],'AP50_95')},
            'GPU_framework_timing_ms':{k:float(v) for k,v in result.speed.items()},
            'baseline_GPU_framework_timing_ms':baseline['framework_timing_ms_per_image'],
            'elapsed_seconds':time.perf_counter()-start,
            'limitations':['Reference-training overlap with this test pool is unknown; not independently validated',
                           'Rare-class support remains tiny; do not generalize from mango/pomegranate scores',
                           'Common size at 416; native 640 run is an exploratory resolution check, not model selection',
                           'Green_Orange and Orange scores merged by maximum before class-aware NMS',
                           'Lemon is unsupported; its score channel is zero in nine-target evaluation',
                           'Desktop GPU timings, not Android latency','No freshness inference']}
    report_name='comparison.json' if image_size==416 else 'comparison_640.json'
    (RUN/report_name).write_text(json.dumps(report,indent=2),encoding='utf-8')
    rows='\n'.join(f"| {r['fruit']} | {r['images']} | {100*r['AP50']:.1f}% | {100*r['AP50_95']:.1f}% |" for r in reference)
    text=f'''# Public YOLO11m reference benchmark

Source: https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector
Revision: `{provenance['revision']}`. SHA256: `{provenance['sha256']}`.
Original file: {weights.stat().st_size:,} bytes. Actual checkpoint has 35 classes.
It was loaded with restricted globals and its state tensors transferred strictly
to an architecture constructed by the installed Ultralytics library.

## Scope and results

Evaluated on the existing frozen 483-image test pool at input size {image_size}.
The 416 run matches the first experiment's size. The 640 run checks the reference's
published input size; it is exploratory and is not used to select training settings.
The same confidence/NMS metric settings are retained. This pool is held out from our
nine-fruit training, but overlap with the reference's training data is UNKNOWN.
It is not a new independent evaluation set. Mango and pomegranate each have only
one test photo; their AP cannot establish generalization.

The reference covers eight targets and has no lemon class. Green_Orange and
Orange scores are merged before NMS. Unsupported class predictions are excluded
from this target-scope benchmark. Original 35-class names are retained in metadata.

| Fruit | Test images | Reference AP50 | Reference AP50-95 |
|---|---:|---:|---:|
{rows}

For the eight shared classes, reference mAP50 is {100*report['shared_scope']['reference_mAP50']:.2f}%
and mAP50-95 is {100*report['shared_scope']['reference_mAP50_95']:.2f}%.
Our trained YOLOv8n scores on the same eight-class scope are
{100*report['shared_scope']['trained_yolov8n_mAP50']:.2f}% and
{100*report['shared_scope']['trained_yolov8n_mAP50_95']:.2f}% respectively.
These are descriptive results, not proof of model superiority on independent phones.

Timings in comparison.json are desktop GPU validation measurements. They do not
establish Android performance. No app weights, completed run, or raw source image
was replaced. This model detects identities, not freshness.

## Bohni dataset access

The exact v13 page returned HTTP403 and the normal public API returned HTTP401.
An authorized Roboflow export or local ZIP is needed before image-level auditing.
No dataset images were downloaded in this step. Keep the dataset export separate
until provenance, counts, boxes, augmentation lineage and overlap have been checked.
'''
    (RUN/('README.md' if image_size==416 else 'NATIVE_SIZE_640.md')).write_text(text,encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
