"""Run the requested nine-class YOLOv8n fine-tuning experiment."""
from pathlib import Path
import os, json, time, shutil, platform, importlib.metadata

RUN=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(RUN/'runtime_config')
os.environ['MPLCONFIGDIR']=str(RUN/'runtime_config/matplotlib')
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['WANDB_DISABLED']='true'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


def main():
    import torch
    import cv2
    from ultralytics import YOLO, settings
    settings.update({'sync':False,'hub':False,'dvc':False,'neptune':False,'raytune':False,
                     'wandb':False,'comet':False,'mlflow':False,'clearml':False,'tensorboard':False})
    torch.set_num_threads(4);cv2.setNumThreads(0)
    if not torch.cuda.is_available():raise RuntimeError('CUDA environment must pass the smoke check before training')
    if shutil.disk_usage(RUN).free<350*1024**2:raise RuntimeError('Insufficient free disk space for checkpoints')
    audit=json.loads((RUN/'reports/data_audit.json').read_text())
    config=dict(data=str(RUN/'data.yaml'),epochs=60,patience=12,imgsz=416,batch=8,
                device=0,workers=2,cache=False,pretrained=True,optimizer='AdamW',
                lr0=0.001,lrf=0.01,momentum=0.937,weight_decay=0.0005,
                warmup_epochs=3.0,seed=42,deterministic=True,amp=False,
                mosaic=0.5,close_mosaic=10,mixup=0.0,degrees=5.0,
                translate=0.1,scale=0.4,fliplr=0.5,hsv_h=0.015,hsv_s=0.5,hsv_v=0.3,
                project=str(RUN/'runs'),name='train',exist_ok=False,
                save=True,save_period=-1,plots=True,verbose=False)
    environment={'python':platform.python_version(),'torch':torch.__version__,
                 'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
                 'gpu_vram_bytes':torch.cuda.get_device_properties(0).total_memory,
                 'packages':{name:importlib.metadata.version(name) for name in ['ultralytics','torchvision','numpy','Pillow','opencv-python']},
                 'split_manifest_sha256':audit['manifest_sha256'],'D_free_bytes':shutil.disk_usage(RUN).free}
    (RUN/'reports/environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    (RUN/'reports/training_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    history=RUN/'reports/epoch_history.jsonl'
    last=RUN/'runs/train/weights/last.pt'
    model=YOLO(str(last if last.exists() else RUN/'pretrained/yolov8n.pt'))
    start=time.time()
    def report(trainer):
        row={'epoch':trainer.epoch+1,'elapsed_seconds':round(time.time()-start,2),
             'metrics':{k:float(v) for k,v in trainer.metrics.items()},'best_fitness':float(trainer.best_fitness),
             'D_free_bytes':shutil.disk_usage(RUN).free}
        with history.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        (RUN/'reports/progress.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        print('EPOCH_SUMMARY '+json.dumps(row),flush=True)
        if row['D_free_bytes']<200*1024**2:trainer.stop=True
    model.add_callback('on_fit_epoch_end',report)
    if last.exists():
        results=model.train(resume=True,device=0,workers=2)
    else:
        results=model.train(**config)
    summary={'training_completed':True,'elapsed_seconds_this_invocation':time.time()-start,
             'best_checkpoint':str(RUN/'runs/train/weights/best.pt'),
             'last_checkpoint':str(last),'metrics':{k:float(v) for k,v in results.results_dict.items()}}
    (RUN/'reports/training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('TRAINING_COMPLETE '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
