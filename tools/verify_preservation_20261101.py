"""Verify saved model/APK hashes and the preserved Classic snapshot."""
from pathlib import Path
import hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/storage-cleanup-2026-11-01'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
items={x['path']:x['sha256'] for x in json.loads((ROOT/'datasets/storage-cleanup-2026-10-31/preserved-artifacts.json').read_text()) if '.gradle' not in Path(x['path']).parts}
base=ROOT/'apps/01-fruit-lens-classic'
items.update({(base/rel).relative_to(ROOT).as_posix():h for rel,h in json.loads((base/'SNAPSHOT_MANIFEST.json').read_text())['files'].items()})
new=ROOT/'detector_training/yolo26s-fruit9-v3/pretrained/provenance.json'
items['detector_training/yolo26s-fruit9-v3/pretrained/yolo26s.pt']=json.loads(new.read_text())['sha256']
errors=[]
for rel,h in items.items():
    p=ROOT/rel
    if not p.is_file() or sha(p)!=h:errors.append(rel)
result={'checked':len(items),'unchanged':len(items)-len(errors),'errors':errors}
name='after.json' if '--after' in sys.argv else 'before.json'
(OUT/name).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
if errors:raise SystemExit(1)
