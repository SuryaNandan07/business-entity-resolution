"""Resumable disk-backed ranking postings; bounded construction and loading."""
from array import array
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3
import time
import numpy as np
from src.candidates import Record

class ChunkPostings:
    def __init__(self,key_ids):self.key_ids=key_ids;self.values={}
    def get(self,key):
        kid=self.key_ids.get(key)
        if kid is None:return None
        if kid not in self.values:self.values[kid]=array('I')
        return self.values[kid]

class DiskPostings:
    def __init__(self,path,offsets):self.stream=Path(path).open('rb');self.offsets=offsets
    def get(self,key):
        position=self.offsets.get(key)
        if position is None:return None
        offset,count=position
        self.stream.seek(offset)
        result=np.fromfile(self.stream,dtype=np.uint32,count=count)
        if len(result)!=count:raise ValueError('Truncated ranking postings')
        return result
    def __del__(self):
        stream=getattr(self,'stream',None)
        if stream is not None:stream.close()

def atomic_json(path,value):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def cached_ranking(cls,catalog,queries,freq,provenance,guard,chunksize=10000):
    from src.compact_ranking import keys_for
    keys=keys_for(queries)
    signature=[catalog.meta['signature'],provenance,keys,chunksize,hashlib.sha256(Path(__file__).read_bytes()).hexdigest()]
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    folder=catalog.cache/f'ranking-{fingerprint}';folder.mkdir(exist_ok=True)
    ready=folder/'ready.json';start=time.perf_counter();n=catalog.meta['records']
    if ready.exists():
        meta=json.loads(ready.read_text())
        for name,size in meta['files'].items():
            if (folder/name).stat().st_size!=size:raise ValueError('Truncated ranking cache')
        with (folder/'offsets.pickle').open('rb') as f:offsets=pickle.load(f)
        index=cls.__new__(cls)
        index.name_total=np.fromfile(folder/'name.bin',dtype=np.float64)
        index.address_total=np.fromfile(folder/'address.bin',dtype=np.float64)
        index.gram_count=np.fromfile(folder/'grams.bin',dtype=np.uint32)
        if any(len(a)!=n for a in (index.name_total,index.address_total,index.gram_count)):raise ValueError('Ranking totals size mismatch')
        index.postings=DiskPostings(folder/'postings.bin',offsets)
        guard()
        return index,dict(cache_hit=True,load_seconds=time.perf_counter()-start,bytes=sum(meta['files'].values()),original_build_seconds=meta['build_seconds'])
    progress_path=folder/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else dict(next_id=0,seconds=0)
    key_ids={key:i for i,key in enumerate(keys)}
    builder=cls.__new__(cls)
    maps=[]
    for attribute,name,dtype in [('name_total','name.bin',np.float64),('address_total','address.bin',np.float64),('gram_count','grams.bin',np.uint32)]:
        file=folder/name
        mapped=np.memmap(file,dtype=dtype,mode='r+' if file.exists() else 'w+',shape=(n,))
        setattr(builder,attribute,mapped);maps.append(mapped)
    db=sqlite3.connect(folder/'segments.sqlite')
    db.execute('PRAGMA cache_size=-16384');db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE IF NOT EXISTS segments (key_id INTEGER,batch INTEGER,data BLOB,PRIMARY KEY(key_id,batch)) WITHOUT ROWID')
    cursor=catalog.db.execute('SELECT * FROM records WHERE id>=? ORDER BY id',(progress['next_id'],))
    try:
        while True:
            guard();tick=time.perf_counter();rows=cursor.fetchmany(chunksize)
            if not rows:break
            builder.postings=ChunkPostings(key_ids)
            for row in rows:builder.add(row[0],Record(*row[1:]),freq)
            batch=rows[0][0]//chunksize
            db.executemany('INSERT OR REPLACE INTO segments VALUES (?,?,?)',
                ((kid,batch,values.tobytes()) for kid,values in sorted(builder.postings.values.items())))
            for mapped in maps:mapped.flush()
            db.commit()
            progress['next_id']=rows[-1][0]+1
            del rows,builder.postings;gc.collect()
            progress['seconds']+=time.perf_counter()-tick
            atomic_json(progress_path,progress)
            if progress['next_id']%100000==0:print(f'Disk ranking index {progress["next_id"]:,}, {progress["seconds"]:.1f}s',flush=True)
        if progress['next_id']!=n:raise ValueError('Incomplete ranking build')
        tick=time.perf_counter();offsets={};position=0;current=None;begin=0;count=0
        partial=folder/'postings.partial'
        with partial.open('wb') as output:
            for kid,data in db.execute('SELECT key_id,data FROM segments ORDER BY key_id,batch'):
                if kid!=current:
                    if current is not None:offsets[keys[current]]=(begin,count)
                    current=kid;begin=position;count=0
                output.write(data);position+=len(data);count+=len(data)//4
                if position//2**20%32==0:guard()
            if current is not None:offsets[keys[current]]=(begin,count)
        partial.replace(folder/'postings.bin')
        with (folder/'offsets.pickle').open('wb') as f:pickle.dump(offsets,f,protocol=5)
        files={name:(folder/name).stat().st_size for name in ('name.bin','address.bin','grams.bin','postings.bin','offsets.pickle')}
        elapsed=progress['seconds']+time.perf_counter()-tick
        atomic_json(ready,dict(build_seconds=elapsed,files=files,records=n,keys=len(offsets),postings=position//4,
                              fingerprint=fingerprint,provenance=provenance,catalog_signature=catalog.meta['signature']))
    finally:
        db.close()
        for mapped in maps:mapped._mmap.close()
    del builder,maps,key_ids,offsets;gc.collect()
    (folder/'segments.sqlite').unlink()
    index,info=cached_ranking(cls,catalog,queries,freq,provenance,guard,chunksize)
    info.update(cache_hit=False,build_seconds=elapsed)
    return index,info
