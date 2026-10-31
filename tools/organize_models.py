"""Create a versioned model catalog; preserve all existing READMEs and source artifacts."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def doc(folder,text):
    folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'README.md').exists():(folder/'README.md').write_text(text,encoding='utf-8')
def copy(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert sha(src)==sha(dest),dest
    else:shutil.copy2(src,dest)
def main():
    base=ROOT/'models';classic=ROOT/'apps/01-fruit-lens-classic/source/android/app/src/main/assets'
    doc(base,'# Model catalog\n\nImmutable model versions, grouped by Classic, custom detectors and custom classifiers. Start at ../PROJECT_GUIDE.md. catalog.json records exact files/hashes. Git LFS tracks binary weights; clone with Git LFS installed and run git lfs pull. Each model README explains its data, input/output, accuracy limits, parameters and training/code location. Never put a new version into an older version folder. New deployment formats may be added only with verification and recorded hashes. Classic source assets remain the preserved reference.\n')
    for group in ['classic','custom-detectors','custom-classifiers']:doc(base/group,f'# {group}\n\nEach subfolder is an explicitly named model version. Its README and metadata explain what it does. Existing model versions are retained for comparison.\n')
    entries=[]
    configs=[('classic/nanodet-plus-m416-v1','NanoDet-Plus-m416','detection',classic/'detector','COCO80 categories; app uses banana/apple/orange. Official pretrained export; no local training or matched accuracy benchmark.','Input416 BGR, mean103.53/116.28/123.675 and std57.375/57.12/58.395. CPU NCNN2threads; no Vulkan. Output3598 rows of80scores+32distance logits. See Classic TECHNICAL_DETAILS.md.','Original architecture parameter/layer count is not independently established for this exact asset. See graph/operator inspection for executable structure.'),
    ('classic/smartfruit-mobilenetv2-v1','SmartFruit MobileNetV2 based','classification',classic/'model','Existing JosephMoustaid/smart-fruit-models-notebooks model. Nine classes: apple/banana/orange times fresh/rotten/unripe. Local training not performed. Original training dataset/split and independently measured accuracy unavailable. Redistribution terms unresolved; retained for local baseline comparison.','96RGB, nearest resize, pixel/255. LiteRT CPU2threads. Nine probability outputs; no unsupported-fruit rejection.','Original trainable parameter/layer count cannot be recovered reliably from the optimized TFLite graph alone. Graph weight and operator counts are separately measured in inspection.json.'),
    ('custom-detectors/yolov8n-fruit9-v1','YOLOv8n nine-fruit v1','detection',ROOT/'detector_training/yolov8n-fruit9-v1/artifacts','Nine fruits.3,221prepared images.60epochs, best49, GTX1650 CUDA. GitHub fruit freshness boxes + Open Images. Test mAP50:58.68%; mAP50-95:47.72%. Mango0% with tiny support. Not camera validated.','416RGB/255, letterbox114. Class order apple,banana,grape,lemon,mango,orange,pomegranate,strawberry,tomato. See training model.json for raw tensor contract.','3,012,603 unfused parameters. Detailed architecture saved beside weights.'),
    ('custom-detectors/yolov8n-fruit9-v2','YOLOv8n nine-fruit v2','detection',ROOT/'detector_training/yolov8n-fruit9-v2/artifacts','Nine fruits.4,936images:3,761train/591val/584test. Bohni v13 eight-fruit subset plus frozen v1 data.43epochs, best31, GTX1650 CUDA. Combined test mAP50:79.01%; mAP50-95:62.04%. Original frozen test mAP50:52.36%, below v1(58.68%). Lemon remains weak; v2 is not a universal improvement.','416RGB/255, letterbox114. NCNN input in0; output out0 is13x3549:xywh then9scores. FP16stored weights, CPU2threads, no Vulkan, FP16arithmetic disabled. App NMS0.5/score0.4/max5regions are engineering settings, not calibrated accuracy thresholds.','3,012,603 unfused parameters; fused checkpoint graph3,007,403. Detailed module/layer tables saved beside weights.'),
    ('custom-classifiers/mobilenetv3small-sixclass-v1','MobileNetV3Small six-class v1','classification',ROOT/'model_exports/mobilenetv3small-sixclass-v1','Fresh/rotten apple,banana,orange;6classes.1,197images:837train/180val/180test. Mendeley8-fruit source subset.12head+4fine epochs, TensorFlow CPU. Test176/180(97.78%), not phone validated.','224RGB, bilinear resize, (pixel-127.5)/127.5. Float32TFLite; six labels in source export metadata.','942,582 parameters. Optimized graph inspection is recorded separately.'),
    ('custom-classifiers/mobilenetv3small-fruit9-v1','MobileNetV3Small nine-fruit v1','classification',ROOT/'classifier_training/mobilenetv3small-fruit9-v1/artifacts','Fresh/spoiled for9fruits,18joint classes.13,354candidate images:9,338train/2,012val/2,004test. Mendeley8-fruit originals + FRUIT-16K; no freshness invented from Bohni. Frozen ImageNet features train head(up to30epochs), then12fine epochs; best fine11. TensorFlow2.21/Keras3.15 CPU. Joint identity+condition test accuracy99.65%(1,997/2,004), macro F1.99283. Same-source grouped split; adjacent captures/source backgrounds may inflate scores. No independent camera accuracy.','160RGB, bilinear stretch, (pixel-127.5)/127.5. Output18softmax probabilities; order fresh/spoiled pairs for apple,banana,grape,lemon,mango,orange,pomegranate,strawberry,tomato. Float16weights with float32IO1.97MB; float32export3.80MB. App uses Fresh/Not fresh/Uncertain while source label IDs are preserved.','949,506 parameters; last40backbone layers selected for fine-tuning with BatchNorm frozen. Full Keras layer summary and configuration saved beside weights.')]
    for version,title,task,source,data,contract,architecture in configs:
        folder=base/version;weights=folder/'weights';weights.mkdir(parents=True,exist_ok=True)
        if version.startswith('classic/'):
            for p in source.iterdir():
                if p.is_file():copy(p,weights/p.name)
        elif version.endswith('yolov8n-fruit9-v2'):
            copy(source/'fruit-detector-yolov8n-9class-v2.pt',weights/'detector.pt')
            copy(source/'model.json',folder/'training_model.json')
        else:
            for p in source.iterdir():
                if p.is_file() and p.suffix in ['.pt','.tflite','.json','.keras']:copy(p,weights/p.name)
        files={p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in weights.rglob('*') if p.is_file() and p.suffix in ['.pt','.bin','.param','.tflite','.keras','.torchscript'] and '.pnnx.' not in p.name}
        metadata=dict(version=version,title=title,task=task,training_data_and_results=data,preprocessing_and_runtime=contract,architecture=architecture,files=files)
        (folder/'version.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        training='apps/01-fruit-lens-classic/TECHNICAL_DETAILS.md' if version.startswith('classic/') else 'ml/README.md' if 'sixclass' in version else 'detector_training/'+version.split('/')[-1]+'/MODEL_CARD.md' if task=='detection' else 'classifier_training/mobilenetv3small-fruit9-v1/MODEL_CARD.md'
        table='\n'.join(f"| {name} | {v['bytes']:,} | `{v['sha256']}` |" for name,v in files.items())
        doc(folder,f'# {title}\n\n## What and why\n\n{data}\n\n## Architecture and size\n\n{architecture}\n\n| File | Bytes | SHA256 |\n|---|---:|---|\n{table}\n\n## Input, outputs and execution\n\n{contract}\n\n## Training, evaluation and code\n\nStart at `{training}` from the project root. That experiment owns exact configurations, split manifests, logs, evaluation and source attribution. Layers/operator summaries are measurements of the deployed graph where the original architecture is unavailable, not invented original layer counts.\n\n## Version policy\n\nKeep weights immutable. A changed training checkpoint needs a new version folder. Android Classic and Custom have separate package IDs and model assets. Test scores describe the stated evaluation split; they do not guarantee camera accuracy or food safety. Full training data remains local and is not included in Git.\n')
        doc(weights,'# Weights\n\nExact model binaries for the parent version. File hashes, shapes, preprocessing and score meaning are documented one folder above. Git LFS stores binaries. Do not replace weights in an older version.')
        entries.append(metadata)
    (base/'catalog.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    # Deploy exactly the chosen versions into the separate app bundle.
    assets=ROOT/'apps/02-fruit-lens-custom/android/app/src/main/assets'
    for name in ['detector.param','detector.bin']:copy(base/'custom-detectors/yolov8n-fruit9-v2/weights/ncnn'/name,assets/'detector'/name)
    source=ROOT/'classifier_training/mobilenetv3small-fruit9-v1/artifacts/freshness_float16.tflite'
    copy(source,assets/'model/freshness.tflite')
    names=['apple','banana','grape','lemon','mango','orange','pomegranate','strawberry','tomato']
    labels=[dict(id=f'{condition}_{fruit}',fruit=fruit,condition=condition,displayName=f'{fruit.title()} · {"Fresh" if condition=="fresh" else "Not fresh"}') for fruit in names for condition in ['fresh','spoiled']]
    bundle=dict(schemaVersion=1,modelId='mobilenetv3small-fruit9-v1',modelFile='freshness.tflite',sha256=sha(source),input=dict(layout='NHWC',width=160,height=160,channels=3,colorSpace='RGB',resizeMode='stretch',interpolation='bilinear',mean=[127.5]*3,std=[127.5]*3),output=dict(scoreType='probabilities'),labels=labels,decision=dict(minScore=.8,thresholdValidated=False),note='0.8 is an uncalibrated UI abstention heuristic. Scores are not freshness percentages. Camera validation pending.')
    (assets/'model/model.json').write_text(json.dumps(bundle,indent=2),encoding='utf-8')
    deployment=dict(detector='yolov8n-fruit9-v2',classifier='mobilenetv3small-fruit9-v1',backend='CPU',threads_per_model=2,GPU=False,NPU=False,files={p.relative_to(assets).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in assets.rglob('*') if p.is_file()},source_namespace='com.fruitlens.app',application_id='com.fruitlens.custom')
    (ROOT/'apps/02-fruit-lens-custom/MODEL_BUNDLE.json').write_text(json.dumps(deployment,indent=2),encoding='utf-8')
    print('Catalog created:',len(entries),'model versions; Custom assets installed.')
if __name__=='__main__':main()
