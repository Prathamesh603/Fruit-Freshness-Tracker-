"""Fetch the official pinned wheel in ranges and verify its official SHA256."""
from pathlib import Path
import urllib.request, re, html, hashlib, concurrent.futures, json, time, threading

NAME='torch-2.5.1+cu121-cp310-cp310-win_amd64.whl'
URL='https://download-r2.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-win_amd64.whl'
TOTAL=2449372784


def main():
    temp=Path('C:/Users/prath/AppData/Local/Temp/FruitLens-yolov8n-setup')
    temp.mkdir(exist_ok=True)
    dest=temp/NAME
    page=urllib.request.urlopen('https://download.pytorch.org/whl/cu121/torch/',timeout=45).read().decode()
    links=[html.unescape(x) for x in re.findall(r'href="([^"]+)"',page)]
    match=next(x for x in links if 'torch-2.5.1%2Bcu121-cp310-cp310-win_amd64.whl#sha256=' in x)
    expected=match.split('#sha256=')[1]
    state_path=temp/'download_state.json'
    candidates=list(Path('C:/Users/prath/AppData/Local/Temp').glob('pip-unpack-*/'+NAME))
    if dest.exists() and state_path.exists():
        state=json.loads(state_path.read_text());prefix=state['prefix']
        positions=state['positions'];progress=state['progress']
    elif dest.exists():
        # Recover a prefix from the first interrupted, preallocated download.
        prefix=0
        with dest.open('rb') as f:
            while block:=f.read(8*1024*1024):
                if not any(block):break
                prefix+=len(block)
        prefix=max(0,prefix-8*1024*1024)
        positions=[prefix+(TOTAL-prefix)*i//8 for i in range(9)]
        progress=positions[:-1].copy()
    elif candidates:
        previous=max(candidates,key=lambda p:p.stat().st_mtime)
        prefix=previous.stat().st_size
        if not 0<prefix<TOTAL:raise ValueError('Unexpected partial size')
        previous.replace(dest)
        positions=[prefix+(TOTAL-prefix)*i//8 for i in range(9)]
        progress=positions[:-1].copy()
    else:
        dest.touch();prefix=0
        positions=[TOTAL*i//8 for i in range(9)];progress=positions[:-1].copy()
    with dest.open('r+b') as f:f.truncate(TOTAL)
    lock=threading.Lock()
    def save_state():
        state_path.write_text(json.dumps({'prefix':prefix,'positions':positions,'progress':progress,'sha256':expected}),encoding='utf-8')
    save_state()
    print('Resuming verified-source download; completed MB',round((prefix+sum(progress[i]-positions[i] for i in range(8)))/1e6,1),flush=True)
    def part(i):
        stop=positions[i+1]
        for attempt in range(20):
            start=progress[i]
            if start==stop:return
            try:
                request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{stop-1}'})
                with urllib.request.urlopen(request,timeout=90) as response,dest.open('r+b',buffering=0) as f:
                    if response.status!=206 or response.headers['Content-Range']!=f'bytes {start}-{stop-1}/{TOTAL}':raise ValueError('Range mismatch')
                    f.seek(start)
                    while progress[i]<stop:
                        chunk=response.read(min(1024*1024,stop-progress[i]))
                        if not chunk:raise ValueError('Incomplete range')
                        f.write(chunk)
                        with lock:
                            progress[i]+=len(chunk);save_state()
                with lock:print('CUDA part',i+1,'of 8 complete',flush=True)
                return
            except Exception:
                if attempt==19:raise
                time.sleep(min(attempt+2,15))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:list(executor.map(part,range(8)))
    h=hashlib.sha256()
    with dest.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=expected:raise ValueError('Official wheel SHA256 mismatch')
    record={'url':URL,'bytes':TOTAL,'sha256':expected,'official_index':'https://download.pytorch.org/whl/cu121/torch/','checksum_verified':True}
    (Path(__file__).parent/'reports/cuda_wheel.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('CUDA_WHEEL_VERIFIED',str(dest),flush=True)


if __name__=='__main__':main()
