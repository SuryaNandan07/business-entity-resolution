"""Read-only integrity audit before resuming a committed ranking build."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='4'
from contextlib import closing
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3
import time
import math
import numpy as np
from src.catalog_cache import Catalog, requested_keys
from src.compact_ranking import keys_for
from src.scalable_holdout2 import PATHS, CACHE, OUT, queries, feature_provenance
from src.build_features import load_frequencies
from src.ranking import prepare
from src.resource_guard import check

def main():
    started=time.perf_counter();check();qs=queries();catalog=Catalog(CACHE,PATHS)
    try:
        result={'catalog_records':catalog.meta['records'],'source_and_version_metadata_valid':True}
        result['catalog_quick_check']=catalog.db.execute('PRAGMA quick_check').fetchall()
        assert result['catalog_quick_check']==[('ok',)],result
        bounds=catalog.db.execute('SELECT min(id),max(id),count(*) FROM records').fetchone()
        n=catalog.meta['records'];assert bounds==(0,n-1,n),bounds
        signature=[catalog.meta['signature'],feature_provenance(),keys_for(qs),10000,
                   hashlib.sha256(Path(__file__).with_name('disk_ranking.py').read_bytes()).hexdigest()]
        digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        folder=CACHE/f'ranking-{digest}'
        progress=json.loads((folder/'progress.json').read_text())
        next_id=progress['next_id'];assert 0<next_id<=n and (next_id%10000==0 or next_id==n)
        result.update(checkpoint_folder=str(folder),next_id=next_id,completed_chunks=(next_id+9999)//10000,
                      stored_build_seconds=progress['seconds'],checkpoint_fingerprint_valid=True)
        maps=[]
        for name,dtype in [('name.bin',np.float64),('address.bin',np.float64),('grams.bin',np.uint32)]:
            assert (folder/name).stat().st_size==n*np.dtype(dtype).itemsize,(folder,name)
            maps.append(np.memmap(folder/name,dtype=dtype,mode='r',shape=(n,)))
        freq=load_frequencies(Path('output/pair_features/training_frequencies.jsonl.gz'))
        checked=sorted({0,next_id-1,*range(9999,next_id,max(10000,(next_id//200000)*10000))})
        for i in checked:
            r=catalog.get(i);p=prepare(r)
            expected=(math.fsum(freq.idf(r.country,t,'name') for t in p.name_tokens),
                      math.fsum(freq.idf(r.country,t,'address') for t in p.address_tokens),len(p.name_bigrams))
            assert tuple(m[i] for m in maps)==expected,(i,expected)
        for m in maps:m._mmap.close()
        result['totals_boundary_records_verified']=len(checked)
        with closing(sqlite3.connect(f'file:{(folder/"segments.sqlite").as_posix()}?mode=ro',uri=True)) as db:
            integrity=db.execute('PRAGMA quick_check').fetchall();assert integrity==[('ok',)],integrity
            low,high,batches=db.execute('SELECT min(batch),max(batch),count(distinct batch) FROM segments').fetchone()
            assert low==0 and high in ((next_id-1)//10000,next_id//10000)
            assert batches==high+1,(low,high,batches)
            assert not db.execute('SELECT 1 FROM segments WHERE length(data)%4!=0 LIMIT 1').fetchone()
            result.update(segments_quick_check=integrity,segment_batches=batches,last_segment_batch=high)
        fingerprint=hashlib.sha256(json.dumps([catalog.meta['signature'],requested_keys(qs)],sort_keys=True).encode()).hexdigest()
        with (CACHE/f'postings-{fingerprint}.pickle').open('rb') as f:idx=pickle.load(f)
        postings=0
        for values in idx.postings.values():
            if values is None or not values:continue
            a=np.frombuffer(values,dtype=np.uint32)
            assert int(a[-1])<n and not np.any(a[:-1]>=a[1:])
            postings+=len(a)
        result.update(retrieval_postings_verified=postings,seconds=time.perf_counter()-started,memory=check(),status='valid; resume at next_id')
        (OUT/'checkpoint_recovery_audit.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:catalog.close()

if __name__=='__main__':main()
