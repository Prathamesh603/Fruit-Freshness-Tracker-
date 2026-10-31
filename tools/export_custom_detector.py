"""Export an owned YOLO checkpoint into a NEW version's NCNN deployment folder.

Usage: CUDA_ENV/python tools/export_custom_detector.py checkpoint.pt NEW_FOLDER
Pinned tools: ultralytics8.3.40, torch2.5.1, pnnx20261020, ncnn1.0.20261020.
Refuses an existing destination so completed model versions remain immutable.
"""
from pathlib import Path
import sys,os,shutil,subprocess,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'detector_training/yolov8n-fruit9-v2/runtime_config')
def main():
    source=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
    if out.exists():raise FileExistsError('Use a new, empty version directory')
    out.mkdir(parents=True)
    target=out/'detector.pt';shutil.copy2(source,target)
    from ultralytics import YOLO
    import pnnx
    model=YOLO(str(target));script=model.export(format='torchscript',imgsz=416,batch=1,half=False,device='cpu')
    command=[pnnx.EXEC_PATH,str(script),'inputshape=[1,3,416,416]','fp16=1',f'ncnnparam={out}/detector.param',f'ncnnbin={out}/detector.bin',f'ncnnpy={out}/detector_ncnn.py',f'pnnxparam={out}/detector.pnnx.param',f'pnnxbin={out}/detector.pnnx.bin',f'pnnxpy={out}/detector_pnnx.py',f'pnnxonnx={out}/detector.pnnx.onnx']
    with (out/'conversion.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    meta=dict(source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input='in0 RGB/255 CHW1x3x416x416 letterbox114',output='out0 13x3549 xywh+9class scores',stored_weights='FP16',files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [out/'detector.param',out/'detector.bin']},verified_against_source=False,note='Run numerical parity and Android tests before integration; conversion alone does not prove accuracy.')
    (out/'conversion.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
