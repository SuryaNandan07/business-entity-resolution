"""Exact query-driven full-catalog retrieval with bounded rescue postings.

Normalized records live on D: in SQLite; each S1 batch gets a reusable compact
posting cache. Every target is visited, including for requested-key frequencies.
Unrequested keys cannot contribute to these queries. Over-limit rescue keys are
discarded only after the reference's exact frequency limit has been exceeded.
"""
from array import array
from collections import Counter
from contextlib import closing
import gc
import hashlib
from itertools import islice
import json
from pathlib import Path
import pickle
import sqlite3
import time
from datetime import datetime, timezone

from src.candidates import Record, read_rows, blocking_keys, CONFIGURATIONS
from src.candidate_improvements import char_ngrams

SCHEMA = 1
RULES = CONFIGURATIONS['C']
LIMITS = {'rare_name':200, 'rare_address':300, 'grams':200}

def signature(paths):
    files = [dict(path=str(Path(p).resolve()), size=Path(p).stat().st_size,
                  mtime_ns=Path(p).stat().st_mtime_ns) for p in paths]
    code = hashlib.sha256()
    for name in ('normalize.py','candidates.py','candidate_improvements.py','catalog_cache.py'):
        code.update(Path(__file__).with_name(name).read_bytes())
    return dict(schema=SCHEMA, sources=files, code_sha256=code.hexdigest(),
                config={'rules':RULES,'limits':LIMITS,'integer':'uint32'})

def valid(cache, expected):
    try:
        meta=json.loads((Path(cache)/'metadata.json').read_text())
        return meta['signature']==json.loads(json.dumps(expected)) and (Path(cache)/'records.sqlite').stat().st_size==meta['database_bytes']
    except (OSError,ValueError,KeyError):
        return False

def connect(path):
    db=sqlite3.connect(path)
    db.execute('PRAGMA cache_size=-32768')
    db.execute('PRAGMA mmap_size=0')
    db.execute('PRAGMA threads=1')
    db.execute('PRAGMA temp_store=MEMORY')
    return db

def build_catalog(paths, cache, chunksize=10000, guard=lambda:None):
    """Sequential four-column streaming; completed cache published atomically."""
    if chunksize<1: raise ValueError('chunksize must be positive')
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    expected=signature(paths)
    if valid(cache,expected): return json.loads((cache/'metadata.json').read_text())
    start=time.perf_counter(); count=0; timings=Counter()
    partial=cache/'records.partial.sqlite'
    if partial.exists(): partial.unlink()
    with closing(connect(partial)) as db:
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('CREATE TABLE records (id INTEGER PRIMARY KEY, entity_id TEXT UNIQUE, business_name TEXT, business_address TEXT, country TEXT, name TEXT, address TEXT)')
        for path in paths:
            iterator=read_rows(path)
            while True:
                guard(); t=time.perf_counter()
                raw=[{k:r.get(k,'') for k in ('entity_id','business_name','business_address','country')} for r in islice(iterator,chunksize)]
                timings['read_seconds']+=time.perf_counter()-t
                if not raw: break
                t=time.perf_counter(); records=[Record.from_row(r) for r in raw]
                timings['normalization_seconds']+=time.perf_counter()-t
                del raw
                t=time.perf_counter()
                db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?)',
                    ((count+i,r.entity_id,r.business_name,r.business_address,r.country,r.name,r.address) for i,r in enumerate(records)))
                db.commit(); timings['write_seconds']+=time.perf_counter()-t
                count+=len(records); del records; gc.collect()
                if count%100000==0: print(f'Catalog {count:,}, {time.perf_counter()-start:.1f}s',flush=True)
        if signature(paths)!=expected: raise RuntimeError('Sources changed during build')
    partial.replace(cache/'records.sqlite')
    meta=dict(signature=expected, records=count, created_utc=datetime.now(timezone.utc).isoformat(),
              seconds=time.perf_counter()-start, timings=dict(timings),database_bytes=(cache/'records.sqlite').stat().st_size)
    temp=cache/'metadata.partial.json'; temp.write_text(json.dumps(meta,indent=2)); temp.replace(cache/'metadata.json')
    return meta

def requested_keys(queries):
    keys=set()
    for q in queries:
        keys.update(blocking_keys(q,RULES))
        if not q.country: continue
        keys.update(('rare_name',q.country,t) for t in set(q.name.split()))
        keys.update(('rare_address',q.country,t) for t in set(q.address.split()) if not t.isdecimal())
        keys.update(('grams',q.country,t) for t in char_ngrams(q.name))
    return sorted(keys)

class CompactIndex:
    def __init__(self, keys):
        self.postings={k:array('I') for k in keys}
        self.counts=Counter()
        self.timings=Counter()

    def add(self, integer_id, r):
        if not 0<=integer_id<2**32: raise ValueError('uint32 target ID overflow')
        if not r.country: return
        keys=list(blocking_keys(r,RULES))
        keys.extend(('rare_name',r.country,t) for t in set(r.name.split()))
        keys.extend(('rare_address',r.country,t) for t in set(r.address.split()) if not t.isdecimal())
        keys.extend(('grams',r.country,t) for t in char_ngrams(r.name))
        for key in keys:
            if key not in self.postings: continue
            self.counts[key]+=1
            postings=self.postings[key]
            if postings is None: continue
            limit=LIMITS.get(key[0])
            if limit is not None and self.counts[key]>limit:
                self.postings[key]=None
            else: postings.append(integer_id)

    def candidates(self,q):
        ids=set()
        for key in blocking_keys(q,RULES): ids.update(self.postings.get(key) or ())
        if not q.country: return ids
        for t in set(q.name.split()): ids.update(self.postings.get(('rare_name',q.country,t)) or ())
        votes=Counter()
        for t in set(q.address.split()):
            if not t.isdecimal(): votes.update(self.postings.get(('rare_address',q.country,t)) or ())
        ids.update(i for i,n in votes.items() if n>=2)
        grams=[g for g in char_ngrams(q.name) if self.postings.get(('grams',q.country,g))]
        grams.sort(key=lambda g:(len(self.postings['grams',q.country,g]),g))
        votes=Counter()
        for g in grams[:4]: votes.update(self.postings['grams',q.country,g])
        ids.update(i for i,n in votes.items() if n>=2)
        return ids

    def statistics(self):
        result={}
        for kind in (*RULES,*LIMITS):
            sizes=[len(v) for k,v in self.postings.items() if k[0]==kind and v is not None]
            counts=[v for k,v in self.counts.items() if k[0]==kind]
            result[kind]=dict(materialized_keys=len(sizes),postings=sum(sizes),
                posting_bytes=4*sum(sizes),maximum=max(sizes,default=0),
                average=sum(sizes)/max(1,len(sizes)),full_frequency_max=max(counts,default=0))
        return result

class Catalog:
    def __init__(self,cache,paths):
        start=time.perf_counter(); self.cache=Path(cache)
        if not valid(cache,signature(paths)): raise ValueError('Missing or stale catalog cache')
        self.meta=json.loads((self.cache/'metadata.json').read_text())
        self.db=connect(self.cache/'records.sqlite')
        self.load_seconds=time.perf_counter()-start

    def close(self): self.db.close()

    def records(self,chunksize=10000):
        cursor=self.db.execute('SELECT * FROM records ORDER BY id')
        while True:
            rows=cursor.fetchmany(chunksize)
            if not rows: break
            for row in rows: yield row[0],Record(*row[1:])

    def get(self,i):
        row=self.db.execute('SELECT * FROM records WHERE id=?',(int(i),)).fetchone()
        if row is None: raise KeyError(i)
        return Record(*row[1:])

    def index(self,queries,guard=lambda:None):
        keys=requested_keys(queries)
        fingerprint=hashlib.sha256(json.dumps([self.meta['signature'],keys],sort_keys=True).encode()).hexdigest()
        file=self.cache/f'postings-{fingerprint}.pickle'
        start=time.perf_counter()
        if file.exists():
            with file.open('rb') as stream: idx=pickle.load(stream)
            return idx,dict(cache_hit=True,load_seconds=time.perf_counter()-start)
        idx=CompactIndex(keys)
        for i,r in self.records():
            idx.add(i,r)
            if i%10000==0: guard()
            if i and i%100000==0: print(f'Postings {i:,}, {time.perf_counter()-start:.1f}s',flush=True)
        elapsed=time.perf_counter()-start
        partial=file.with_suffix('.partial')
        with partial.open('wb') as stream: pickle.dump(idx,stream,protocol=5)
        partial.replace(file)
        return idx,dict(cache_hit=False,build_seconds=elapsed,statistics=idx.statistics())
