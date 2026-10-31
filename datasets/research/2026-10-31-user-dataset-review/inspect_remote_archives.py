"""Inspect ZIP class maps and entry counts using bounded HTTP byte ranges only."""
from pathlib import Path
import io,requests,json,zipfile,collections,hashlib
ROOT=Path(__file__).resolve().parent
class RemoteFile(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.transferred=0;self.session=requests.Session()
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        if self.pos<0:raise ValueError('Negative offset')
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        if n>16*1024**2 or self.transferred+n>32*1024**2:raise ValueError('Metadata byte budget exceeded; refusing image archive download')
        start=self.pos;end=start+n-1
        with self.session.get(self.url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,timeout=45) as response:
            if response.status_code!=206:raise RuntimeError(f'Range not supported: HTTP{response.status_code}; full response not downloaded')
            chunks=[];total=0
            for chunk in response.iter_content(65536):
                total+=len(chunk)
                if total>n:raise ValueError('Server exceeded requested range')
                chunks.append(chunk)
            data=b''.join(chunks)
            if len(data)!=n:raise ValueError(f'Short range {len(data)} != {n}')
        self.pos+=len(data);self.transferred+=len(data);return data
def main():
    for name in ['zenodo20216','zenodo28139','deepnir']:
        meta=json.loads((ROOT/(name+'.json')).read_text());entry=next(f for f in meta['files'] if f['key'].endswith('.zip'))
        reader=RemoteFile(entry['links']['self'],entry['size'])
        try:
            with zipfile.ZipFile(reader) as z:
                items=z.infolist();names=[i.filename for i in items]
                configs=[i for i in items if Path(i.filename).name.lower() in ['data.yaml','data.yml','dataset.yaml','classes.txt','obj.names','readme.roboflow.txt'] and i.file_size<256000]
                out=ROOT/name;out.mkdir(exist_ok=True)
                files=[]
                for i,info in enumerate(configs):
                    data=z.read(info);dest=out/f'{i:02}_{Path(info.filename).name}';dest.write_bytes(data)
                    files.append(dict(member=info.filename,path=dest.relative_to(ROOT).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
                counts=collections.Counter()
                for info in items:
                    lower=info.filename.lower();kind='images' if Path(lower).suffix in ['.jpg','.jpeg','.png','.webp','.bmp'] else 'labels' if lower.endswith('.txt') and '/labels/' in lower else 'other'
                    split=next((s for s in ['train','valid','val','test'] if f'/{s}/' in '/'+lower),'other')
                    counts[(split,kind)]+=1
                report=dict(archive=entry['key'],archive_bytes=entry['size'],entries=len(items),uncompressed_bytes=sum(i.file_size for i in items),counts={f'{s}/{k}':v for (s,k),v in counts.items()},metadata_files=files,range_bytes_transferred=reader.transferred,whole_archive_crc_verified=False,image_content_verified=False)
                (ROOT/(name+'_archive_audit.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
                (out/'archive_members.json').write_text(json.dumps([dict(path=i.filename,bytes=i.file_size,compressed=i.compress_size) for i in items]),encoding='utf-8')
                print(name,json.dumps(report),flush=True)
        except Exception as exc:
            (ROOT/(name+'_archive_error.json')).write_text(json.dumps(dict(error=str(exc),bytes_transferred=reader.transferred)),encoding='utf-8');print(name,str(exc),flush=True)
if __name__=='__main__':main()
