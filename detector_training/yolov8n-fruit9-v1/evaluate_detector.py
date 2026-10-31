"""Evaluate the selected checkpoint and produce a self-contained local handoff."""
from pathlib import Path
from collections import Counter
import os, csv, json, hashlib, time, statistics

RUN=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(RUN/'runtime_config')
os.environ['MPLCONFIGDIR']=str(RUN/'runtime_config/matplotlib')
os.environ['OMP_NUM_THREADS']='4'


def main():
    import torch, cv2, numpy as np
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from PIL import Image, ImageDraw, ImageOps
    torch.set_num_threads(4);cv2.setNumThreads(0)
    if not (RUN/'reports/training_summary.json').exists():raise RuntimeError('Training has not completed')
    best=RUN/'runs/train/weights/best.pt'
    model=YOLO(str(best))
    metrics=model.val(data=str(RUN/'data.yaml'),split='test',imgsz=416,batch=8,
                      device=0,workers=2,half=False,conf=0.001,iou=0.7,max_det=300,
                      plots=True,save_json=False,project=str(RUN/'runs'),name='test',exist_ok=True,verbose=True)
    with (RUN/'reports/split_manifest.csv').open(encoding='utf-8',newline='') as f:manifest=list(csv.DictReader(f))
    test=[r for r in manifest if r['split']=='test']
    names=[model.names[i] for i in range(len(model.names))]
    support=Counter()
    for row in test:
        for line in (RUN/row['label_path']).read_text().splitlines():support[int(line.split()[0])]+=1
    perclass=[]
    for j,class_id in enumerate(metrics.box.ap_class_index):
        i=int(class_id)
        perclass.append({'class_id':i,'fruit':names[i],'test_images':sum(names[i] in r['fruits'].split('|') for r in test),
                         'test_boxes':support[i],'precision':float(metrics.box.p[j]),'recall':float(metrics.box.r[j]),
                         'f1':float(metrics.box.f1[j]),'AP50':float(metrics.box.ap50[j]),'AP50_95':float(metrics.box.ap[j])})
    evaluation={'checkpoint_sha256':hashlib.sha256(best.read_bytes()).hexdigest(),
                'split':'test','images':len(test),'boxes':sum(support.values()),
                'metrics':{k:float(v) for k,v in metrics.results_dict.items()},'per_class':perclass,
                'framework_timing_ms_per_image':{k:float(v) for k,v in metrics.speed.items()},
                'confidence_for_AP':0.001,'NMS_iou':0.7,'max_detections':300,
                'note':'Precision/recall use framework max-F1 operating point; this is not a separately calibrated app threshold.',
                'limitations':['Same-source grouped test, not independent phone evaluation',
                               'Mango and pomegranate test support too small for robust conclusions',
                               'Original session and physical-fruit IDs unavailable',
                               'Prepared-food and cut-fruit scenes occur in source data',
                               'No Android latency measured; framework timing is desktop CUDA timing']}
    (RUN/'reports/test_evaluation.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
    with (RUN/'reports/per_class_test.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(perclass[0]));writer.writeheader();writer.writerows(perclass)
    artifacts=RUN/'artifacts';artifacts.mkdir(exist_ok=True)
    exported=artifacts/'fruit-detector-yolov8n-9class-v1.pt'
    if exported.exists() and exported.read_bytes()!=best.read_bytes():raise ValueError('Refusing to replace a different final artifact')
    exported.write_bytes(best.read_bytes())
    # One deterministic image per fruit, plus ordinary held-out scenes, for qualitative review.
    selected=[]
    for fruit in names:
        options=sorted([r for r in test if fruit in r['fruits'].split('|')],key=lambda r:r['image_id'])
        if options and options[0] not in selected:selected.append(options[0])
    for row in sorted(test,key=lambda r:r['source_sha256']):
        if len(selected)>=12:break
        if row not in selected:selected.append(row)
    canvas=Image.new('RGB',(1200,((len(selected)+3)//4)*300),'white');draw=ImageDraw.Draw(canvas)
    predictions=[]
    for i,row in enumerate(selected):
        result=model.predict(str(RUN/row['image_path']),imgsz=416,conf=0.25,iou=0.7,device=0,verbose=False)[0]
        rgb=cv2.cvtColor(result.plot(),cv2.COLOR_BGR2RGB)
        tile=ImageOps.contain(Image.fromarray(rgb),(295,260))
        x=(i%4)*300;y=(i//4)*300;canvas.paste(tile,(x,y+35))
        draw.text((x+2,y+2),row['image_id'],fill='black');draw.text((x+2,y+17),'GT: '+row['fruits'],fill='black')
        predictions.append({'image_id':row['image_id'],'ground_truth_fruits':row['fruits'],
                            'predictions':[{'fruit':names[int(c)],'score':float(s),'xyxy':[float(v) for v in xy]}
                                           for xy,s,c in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]})
    canvas.save(RUN/'reports/test_prediction_samples.jpg')
    (RUN/'reports/sample_predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8')
    # Portable raw-output export; no Android integration is performed here.
    model=YOLO(str(best));model.to('cpu')
    script_path=Path(model.export(format='torchscript',imgsz=416,batch=1,half=False,optimize=False,device='cpu'))
    script_dest=artifacts/'fruit-detector-yolov8n-9class-v1.torchscript'
    script_dest.write_bytes(script_path.read_bytes())
    scripted=torch.jit.load(str(script_dest),map_location='cpu').eval()
    native=YOLO(str(best)).model.float().eval().cpu()
    errors=[];latencies=[]
    for row in selected:
        frame=cv2.imread(str(RUN/row['image_path']))
        frame=LetterBox(new_shape=(416,416),auto=False)(image=frame)
        tensor=torch.from_numpy(np.ascontiguousarray(frame[:,:,::-1].transpose(2,0,1))).float().unsqueeze(0)/255.0
        with torch.inference_mode():
            expected=native(tensor)[0];actual=scripted(tensor)
            if isinstance(actual,(tuple,list)):actual=actual[0]
            errors.append(float((expected-actual).abs().max()))
            if not torch.allclose(expected,actual,atol=0.002,rtol=0.001):raise ValueError('TorchScript parity failed')
            for _ in range(3):scripted(tensor)
            for _ in range(5):
                start=time.perf_counter();scripted(tensor);latencies.append((time.perf_counter()-start)*1000)
    parity={'images':len(selected),'passed':True,'max_absolute_raw_output_difference':max(errors),
            'input_shape':[1,3,416,416],'output_shape':list(actual.shape),'CPU_threads':4,
            'CPU_raw_forward_median_ms':statistics.median(latencies),
            'CPU_raw_forward_p95_ms':float(np.percentile(latencies,95)),
            'note':'Desktop CPU raw forward only. Excludes decoding, letterbox, NMS and rendering; not Android latency.'}
    (RUN/'reports/export_parity.json').write_text(json.dumps(parity,indent=2),encoding='utf-8')
    metadata={'model_id':'yolov8n-fruit9-v1','task':'fruit_object_detection','names':names,
              'input':{'shape':[1,3,416,416],'dtype':'float32','color':'RGB','normalization':'pixel/255',
                       'resize':'aspect-preserving letterbox','padding_value':114},
              'raw_output':{'shape':list(actual.shape),'layout':'batch, 4+num_classes, candidate_count',
                            'boxes':'center_x,center_y,width,height in letterboxed input pixels',
                            'class_values':'per-class scores; use class-aware NMS and undo letterbox'},
              'freshness_classifier':False,'android_validated':False,
              'parameters':sum(p.numel() for p in native.parameters()),
              'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [exported,script_dest]}}
    (artifacts/'model.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (artifacts/'README.md').write_text('Nine-fruit detection artifacts. Read model.json and ../MODEL_CARD.md. These do not classify freshness and have not been integrated into Android. Class IDs follow model.json, not the planned thirteen-class taxonomy.\n',encoding='utf-8')
    print('EVALUATION_COMPLETE '+json.dumps(evaluation),flush=True)
    print('EXPORT_PARITY '+json.dumps(parity),flush=True)


if __name__=='__main__':main()
