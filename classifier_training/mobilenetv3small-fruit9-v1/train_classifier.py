"""Train a compact 18-class visible-condition model without changing earlier editions."""
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import os,csv,json,hashlib,time,random,sys,shutil
RUN=Path(__file__).resolve().parent
PROJECT=RUN.parents[1]
SOURCE=PROJECT/'datasets/fruit13'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['OMP_NUM_THREADS']='4'
os.environ['TF_NUM_INTRAOP_THREADS']='4'
os.environ['TF_NUM_INTEROP_THREADS']='2'
import numpy as np
from PIL import Image,ImageOps
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report,confusion_matrix,f1_score
NAMES=['apple','banana','grape','lemon','mango','orange','pomegranate','strawberry','tomato']
LABELS=[condition+'_'+fruit for fruit in NAMES for condition in ['fresh','spoiled']]
SIZE=160
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
    p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2),encoding='utf-8')
def stage(name):save('status.json',dict(stage=name,completed=False));print('STAGE',name,flush=True)
def csvwrite(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dataset(x,y,training=False):
    ds=tf.data.Dataset.from_tensor_slices((x,y))
    if training:ds=ds.shuffle(min(len(y),4096),seed=42,reshuffle_each_iteration=True)
    def process(image,label):
        image=tf.cast(image,tf.float32)
        if training:
            image=tf.image.random_flip_left_right(image)
            image=tf.image.random_brightness(image,10.)
            image=tf.image.random_contrast(image,.9,1.1)
        return (tf.clip_by_value(image,0.,255.)-127.5)/127.5,label
    return ds.map(process,num_parallel_calls=2).batch(32).prefetch(1)
def main():
    started=time.time();keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    for folder in ['reports','cache','checkpoints','artifacts','logs']:
        p=RUN/folder;p.mkdir(exist_ok=True)
        (p/'README.md').write_text({'reports':'Dataset split audit, validation selection, held-out metrics and integrity checks.','cache':'160 by 160 RGB uint8 images and frozen features, derived from raw images. This speeds training; original files are untouched.','checkpoints':'Best validation-selected Keras model and latest fine-tuning checkpoint.','artifacts':'Final Keras model, float32 and float16 TFLite models, exact class order and preprocessing metadata.','logs':'Training output and per-epoch CSV losses/metrics.'}[folder])
    (RUN/'README.md').write_text('MobileNetV3Small nine-fruit visible freshness experiment. Eighteen classes: fresh/spoiled for each fruit. Two Mendeley sources supply condition labels; Bohni and Open Images are detection data and are not used to invent freshness labels. Training uses existing native Windows TensorFlow CPU environment. See MODEL_CARD.md after completion. Earlier models and Classic remain preserved.')
    stage('prepare_grouped_splits')
    with (SOURCE/'classification/candidates.csv').open(encoding='utf-8',newline='') as f:rows=[r for r in csv.DictReader(f) if r['fruit'] in NAMES]
    groups=defaultdict(list)
    for r in rows:
        assert r['decode_ok']=='True' and r['group_has_label_conflict']=='False' and r['exact_duplicate']=='False'
        groups[r['group_id']].append(r)
    for label in LABELS:
        gg=[items for items in groups.values() if items[0]['label']==label]
        assert len(gg)>=3
        assert all(len({r['label'] for r in items})==1 for items in gg)
        random.Random(42+LABELS.index(label)).shuffle(gg)
        total=sum(map(len,gg));targets=dict(train=.7*total,val=.15*total,test=.15*total);counts=Counter()
        for items in gg:
            split=max(targets,key=lambda s:(targets[s]-counts[s])/targets[s])
            for r in items:r['split']=split;r['class_id']=LABELS.index(r['label'])
            counts[split]+=len(items)
    rows.sort(key=lambda r:(r['split'],r['image_id']))
    csvwrite(RUN/'reports/split_manifest.csv',rows)
    for first in ['train','val','test']:
        for second in ['train','val','test']:
            if first!=second:
                for key in ['sha256','group_id']:assert not ({r[key] for r in rows if r['split']==first}&{r[key] for r in rows if r['split']==second})
    manifest_hash=sha(RUN/'reports/split_manifest.csv')
    audit=dict(images=len(rows),classes=LABELS,fruits=NAMES,source_counts=dict(Counter(r['source'] for r in rows)),groups=len(groups),split_counts=dict(Counter(r['split'] for r in rows)),class_counts={s:dict(Counter(r['label'] for r in rows if r['split']==s)) for s in ['train','val','test']},manifest_sha256=manifest_hash,source_inventory_sha256=sha(SOURCE/'classification/candidates.csv'),excluded='Uses candidate inventory only: exact duplicates and pHash label-conflict groups withheld.',limitations=['Canonical 64-bit pHash groups, not known physical-fruit/session IDs; crops/rotations and adjacent captures may escape grouping','Condition labels inherited from publisher folders, not independently adjudicated','No independent phone evaluation; background/source shortcuts possible','FRUIT-16K fresh/spoiled terminology harmonized to canonical labels','Prior six-class experiment images may occur; initialize from ImageNet only, do not compare these new test scores directly to old test scores'])
    save('reports/data_audit.json',audit)
    save('reports/config.json',dict(backbone='MobileNetV3Small',initialization='ImageNet',input_size=SIZE,normalization='(RGB-127.5)/127.5',backend='TensorFlow CPU',seed=42,head_epochs=30,head_patience=5,fine_epochs=12,fine_patience=4,head_lr=.001,fine_lr=.0001,batch=32,trainable_backbone_tail_layers=40,freeze_batchnorm=True,selection='validation cross entropy',class_weights='inverse training frequency',tensorflow=tf.__version__,keras=keras.__version__))
    stage('decode_and_verify_images')
    arrays={};ys={};splitrows={}
    def read_one(r):
        p=SOURCE/r['path'];assert sha(p)==r['sha256']
        with Image.open(p) as im:return np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((SIZE,SIZE),Image.Resampling.BILINEAR),dtype=np.uint8)
    for split in ['train','val','test']:
        rr=[r for r in rows if r['split']==split];splitrows[split]=rr
        p=RUN/'cache'/(split+'.npy')
        x=np.lib.format.open_memmap(p,mode='w+',dtype=np.uint8,shape=(len(rr),SIZE,SIZE,3))
        with ThreadPoolExecutor(max_workers=4) as ex:
            for i,img in enumerate(ex.map(read_one,rr)):x[i]=img
        x.flush();arrays[split]=x;ys[split]=np.array([r['class_id'] for r in rr],dtype=np.int32)
    stage('frozen_feature_extraction')
    backbone=keras.applications.MobileNetV3Small(include_top=False,weights='imagenet',input_shape=(SIZE,SIZE,3),include_preprocessing=False,pooling='avg',name='backbone')
    backbone.trainable=False
    features={}
    for split in ['train','val']:
        ds=dataset(arrays[split],ys[split]).map(lambda x,y:x,num_parallel_calls=2)
        features[split]=backbone.predict(ds,verbose=0)
        np.save(RUN/'cache'/(split+'_features.npy'),features[split])
    feature_input=keras.Input((576,));z=keras.layers.Dropout(.2)(feature_input);dense=keras.layers.Dense(18,activation='softmax',name='condition');outputs=dense(z)
    head=keras.Model(feature_input,outputs)
    counts=Counter(ys['train'].tolist());weights={c:len(ys['train'])/(18*counts[c]) for c in range(18)}
    head.compile(optimizer=keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    head.fit(features['train'],ys['train'],validation_data=(features['val'],ys['val']),batch_size=64,epochs=30,class_weight=weights,verbose=2,callbacks=[keras.callbacks.CSVLogger(str(RUN/'logs/head_epochs.csv')),keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),keras.callbacks.TerminateOnNaN()])
    inputs=keras.Input((SIZE,SIZE,3),name='image');out=dense(keras.layers.Dropout(.2)(backbone(inputs,training=False)))
    model=keras.Model(inputs,out,name='fruit9_visible_condition')
    model.save(RUN/'checkpoints/head_best.keras')
    stage('fine_tune_backbone')
    backbone.trainable=True
    for layer in backbone.layers:layer.trainable=False
    for layer in backbone.layers[-40:]:
        if not isinstance(layer,keras.layers.BatchNormalization):layer.trainable=True
    model.compile(optimizer=keras.optimizers.Adam(.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    # Save head baseline as the initial candidate; fine-tuning must improve validation loss.
    baseval=float(model.evaluate(dataset(arrays['val'],ys['val']),verbose=0)[0])
    bestpath=RUN/'checkpoints/best.keras';model.save(bestpath)
    cb=keras.callbacks.ModelCheckpoint(str(bestpath),monitor='val_loss',mode='min',save_best_only=True,initial_value_threshold=baseval)
    history=model.fit(dataset(arrays['train'],ys['train'],True),validation_data=dataset(arrays['val'],ys['val']),epochs=12,class_weight=weights,verbose=2,callbacks=[cb,keras.callbacks.CSVLogger(str(RUN/'logs/fine_epochs.csv')),keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True),keras.callbacks.TerminateOnNaN()])
    model.save(RUN/'checkpoints/last.keras');model=keras.models.load_model(bestpath)
    save('reports/training_summary.json',dict(completed=True,elapsed_seconds=time.time()-started,fine_epochs=len(history.history['loss']),head_baseline_val_loss=baseval,selected_val_loss=float(model.evaluate(dataset(arrays['val'],ys['val']),verbose=0)[0]),parameters=model.count_params(),train_backend='TensorFlow CPU',GPU_used=False))
    stage('heldout_evaluation')
    probabilities=model.predict(dataset(arrays['test'],ys['test']).map(lambda x,y:x),verbose=0)
    np.save(RUN/'reports/test_probabilities.npy',probabilities)
    pred=probabilities.argmax(1);truth=ys['test'];report=classification_report(truth,pred,labels=list(range(18)),target_names=LABELS,output_dict=True,zero_division=0)
    save('reports/classification_report.json',report);save('reports/confusion_matrix.json',dict(labels=LABELS,matrix=confusion_matrix(truth,pred,labels=list(range(18))).tolist()))
    perfruit=[]
    for i,fruit in enumerate(NAMES):
        mask=truth//2==i;actual=truth[mask]%2
        # Known-fruit assessment takes only that fruit's pair of logits/probabilities.
        pair=probabilities[mask,2*i:2*i+2];guess=pair.argmax(1)
        spoiled=actual==1;fresh=actual==0
        perfruit.append(dict(fruit=fruit,test_images=int(mask.sum()),fresh_test_images=int(fresh.sum()),spoiled_test_images=int(spoiled.sum()),joint18_accuracy=float((pred[mask]==truth[mask]).mean()),known_fruit_condition_accuracy=float((guess==actual).mean()),spoiled_recall=float((guess[spoiled]==1).mean()),spoiled_predicted_fresh=int(((guess==0)&spoiled).sum())))
    csvwrite(RUN/'reports/per_fruit_test.csv',perfruit)
    evaluation=dict(test_images=len(truth),joint18_accuracy=float((pred==truth).mean()),macro_f1=float(f1_score(truth,pred,average='macro')),fruit_identity_accuracy=float((pred//2==truth//2).mean()),condition_accuracy_joint_prediction=float((pred%2==truth%2).mean()),per_fruit=perfruit,independent_phone_validation=False)
    save('reports/test_evaluation.json',evaluation)
    shutil.copy2(bestpath,RUN/'artifacts/freshness.keras')
    finalize(model,arrays,ys,probabilities,rows,audit,evaluation,perfruit,manifest_hash,splitrows,started)

def finalize(model,arrays,ys,probabilities,rows,audit,evaluation,perfruit,manifest_hash,splitrows,started):
    truth=ys['test']
    stage('export_tflite')
    artifactrecords={}
    model.export(str(RUN/'cache/saved_model'),format='tf_saved_model')
    for name,half in [('freshness_float32.tflite',False),('freshness_float16.tflite',True)]:
        converter=tf.lite.TFLiteConverter.from_saved_model(str(RUN/'cache/saved_model'))
        converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
        if half:converter.optimizations=[tf.lite.Optimize.DEFAULT];converter.target_spec.supported_types=[tf.float16]
        data=converter.convert();p=RUN/'artifacts'/name;p.write_bytes(data);artifactrecords[name]=dict(bytes=len(data),sha256=sha(p))
    stage('verify_exports_and_preservation')
    from ai_edge_litert.interpreter import Interpreter
    samples=[]
    for c in range(18):samples.extend(np.flatnonzero(truth==c)[:5].tolist())
    x=(arrays['test'][samples].astype(np.float32)-127.5)/127.5;expected=probabilities[samples]
    parity={}
    for name in artifactrecords:
        interpreter=Interpreter(model_path=str(RUN/'artifacts'/name),num_threads=4);interpreter.allocate_tensors();inp=interpreter.get_input_details()[0];out=interpreter.get_output_details()[0];got=[];times=[]
        for image in x:
            interpreter.set_tensor(inp['index'],image[None]);t=time.perf_counter();interpreter.invoke();times.append((time.perf_counter()-t)*1000);got.append(interpreter.get_tensor(out['index'])[0])
        got=np.array(got);error=float(np.abs(got-expected).max());agreement=float((got.argmax(1)==expected.argmax(1)).mean())
        assert error<(0.001 if 'float32' in name else .025),str((name,error))
        parity[name]=dict(samples=len(samples),maximum_probability_difference=error,argmax_agreement=agreement,desktop_CPU_median_ms=float(np.median(times[5:])),timing_note='Four-thread desktop CPU TFLite inference only, not Android latency')
    save('reports/export_verification.json',parity)
    for s in ['train','val','test']:
        for other in ['train','val','test']:
            if s!=other:assert not({r['group_id'] for r in splitrows[s]}&{r['group_id'] for r in splitrows[other]})
    assert sha(RUN/'reports/split_manifest.csv')==manifest_hash
    protected={'apps/01-fruit-lens-classic/source/android/app/src/main/assets/model/freshness.tflite':'e09307ca0aacdfc83666ca8ff25c02659d6323a1372409b390561619778b10a4','model_exports/mobilenetv3small-sixclass-v1/freshness.tflite':'dd519afd78d27d731fc4cef0e9a02b96086e42122a51a768874b8b6fdfc9052e','detector_training/yolov8n-fruit9-v2/artifacts/fruit-detector-yolov8n-9class-v2.pt':'440c4ef36dc825878d1bd84a58eb91885726b9714774ddcc42081b11341d9419'}
    for path,expectedsha in protected.items():assert sha(PROJECT/path)==expectedsha
    save('artifacts/model.json',dict(model_id='mobilenetv3small-fruit9-v1',task='joint_fruit_visible_condition_classification',labels=[dict(id=i,fruit=NAMES[i//2],condition='fresh' if i%2==0 else 'spoiled',display='No visible spoilage' if i%2==0 else 'Visible spoilage') for i in range(18)],input=dict(shape=[1,SIZE,SIZE,3],dtype='float32',color='RGB',resize='EXIF transpose then bilinear resize to160x160',normalization='(pixel-127.5)/127.5'),output=dict(shape=[1,18],activation='softmax',unknown_class=False),parameters=model.count_params(),training_backend='TensorFlow CPU',artifacts=artifactrecords,integration_rule='Do not force unsupported fruit into these 18 classes. Require detector-supported identity, check classifier identity agreement, and allow uncertain results. Known-fruit pair metrics are separately reported and are not the accuracy of the entire detector/classifier pipeline.',android_validated=False))
    save('reports/final_verification.json',dict(passed=True,source_hashes_verified=len(rows),manifest_unchanged=True,hash_and_group_split_disjoint=True,all18_classes_present_each_split=all(len(set(ys[s]))==18 for s in ys),export_parity_passed=True,detector_and_existing_classifiers_unchanged=True))
    table='\n'.join(f"| {r['fruit']} | {r['test_images']} | {r['joint18_accuracy']*100:.2f}% | {r['known_fruit_condition_accuracy']*100:.2f}% | {r['spoiled_recall']*100:.2f}% |" for r in perfruit)
    (RUN/'MODEL_CARD.md').write_text(f'''# MobileNetV3Small nine-fruit visible-condition classifier

ImageNet initialized MobileNetV3Small,160x160RGB,18 joint fruit-condition classes. Training: native Windows TensorFlow CPU. GPU was not used. Frozen features train a weighted head, then the last40 backbone layers are fine-tuned with batch normalization frozen. Best candidate selected by validation loss only. Exact configuration: reports/config.json.

Data: {len(rows)} candidate images from Mendeley Sultana8-fruit originals and FRUIT-16K, covering nine fruit identities. Publisher condition folders supply labels. Bohni does not supply freshness labels and is not used. Sources, attribution, licenses and original hashes: datasets/fruit13/SOURCES.md and source metadata. Split counts: {audit['split_counts']}. pHash groups and exact hashes do not cross splits. Physical fruit/session independence is unknown; adjacent captures and source backgrounds can inflate performance.

Joint18 test accuracy: {evaluation['joint18_accuracy']*100:.2f}%. Macro F1: {evaluation['macro_f1']:.4f}. Fruit identity accuracy: {evaluation['fruit_identity_accuracy']*100:.2f}%. Test images: {len(truth)}.

| Fruit | Test images | Joint18 accuracy | Condition accuracy given correct fruit | Spoiled recall given correct fruit |
|---|---:|---:|---:|---:|
{table}

Known-fruit columns select only the two outputs for the actual fruit; they do not include detector mistakes. This model has no unknown/reject class and does not estimate shelf life, internal quality, or food safety. No real-phone or independent field validation has occurred. Fresh means source-labeled visually fresh, not a safety determination. Allow uncertain output and fruit-identity mismatch rejection in a future app; thresholds require phone validation.

Deployment artifacts: float32 and float16 TFLite, with exact labels/preprocessing/hashes in artifacts/model.json. Logs: logs/train.log; epoch losses: logs/head_epochs.csv and logs/fine_epochs.csv. Checkpoints: checkpoints/best.keras and last.keras. Exports verified against Keras on {len(samples)} held-out images. Float16 compresses weights, does not guarantee faster device inference. Existing detector, Classic and six-class freshness model preserved.
''',encoding='utf-8')
    save('status.json',dict(stage='complete',completed=True,elapsed_seconds=time.time()-started));print('COMPLETE',json.dumps(evaluation),flush=True)

def resume_finish():
    model=keras.models.load_model(RUN/'checkpoints/best.keras')
    with (RUN/'reports/split_manifest.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    arrays={s:np.load(RUN/'cache'/(s+'.npy'),mmap_mode='r') for s in ['train','val','test']}
    splitrows={s:[r for r in rows if r['split']==s] for s in arrays}
    ys={s:np.array([int(r['class_id']) for r in splitrows[s]]) for s in arrays}
    probabilities=np.load(RUN/'reports/test_probabilities.npy')
    audit=json.loads((RUN/'reports/data_audit.json').read_text());evaluation=json.loads((RUN/'reports/test_evaluation.json').read_text())
    finalize(model,arrays,ys,probabilities,rows,audit,evaluation,evaluation['per_fruit'],audit['manifest_sha256'],splitrows,time.time()-json.loads((RUN/'reports/training_summary.json').read_text())['elapsed_seconds'])
if __name__=='__main__':
    try:
        if '--finish' in sys.argv:resume_finish()
        else:main()
    except Exception as exc:save('failure.json',dict(error=str(exc)));raise
