import csv
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from src.catalog_cache import Catalog, CompactIndex, build_catalog, signature, valid, requested_keys
from src.candidates import Record, build_index, generate_candidates, CONFIGURATIONS
from src.candidate_improvements import AdditionalIndex, Frequencies, prepare_weighted, improved_signals, improved_score

def record(i,name,address='',country='X'):
    return Record.from_row(dict(entity_id=i,business_name=name,business_address=address,country=country))

class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.path=self.root/'source.tsv'; self.cache=self.root/'cache'
        self.pool=[record('S2-1','Alpha Shop','1 Long Road'),record('S3-2','Alphx Shop','2 Long Road'),record('S2-3','東京店','３ 道', 'JP'),record('S3-4','Empty','','')]
        with self.path.open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f,delimiter='\t'); w.writerow(['entity_id','business_name','business_address','country'])
            w.writerows((r.entity_id,r.business_name,r.business_address,r.country) for r in self.pool)
    def tearDown(self): self.temp.cleanup()
    def test_chunk_boundaries_mapping_and_reuse(self):
        for chunk in (1,2,3,10):
            cache=self.root/str(chunk); build_catalog([self.path],cache,chunk)
            c=Catalog(cache,[self.path])
            try:
                self.assertEqual(list(c.records()),list(enumerate(self.pool)))
                for i,r in enumerate(self.pool): self.assertEqual(c.get(i),r)
                self.assertEqual(build_catalog([self.path],cache,chunk),c.meta)
            finally:c.close()
    def test_stale_cache_rejected(self):
        build_catalog([self.path],self.cache)
        self.assertTrue(valid(self.cache,signature([self.path])))
        with patch('src.catalog_cache.SCHEMA',999):
            self.assertFalse(valid(self.cache,signature([self.path])))
        with self.path.open('a') as f:f.write('\n')
        self.assertFalse(valid(self.cache,signature([self.path])))
        with self.assertRaises(ValueError):Catalog(self.cache,[self.path])
    def test_missing_incomplete_and_truncated(self):
        self.assertFalse(valid(self.cache,signature([self.path])))
        build_catalog([self.path],self.cache)
        with (self.cache/'records.sqlite').open('ab') as f:f.write(b'x')
        self.assertFalse(valid(self.cache,signature([self.path])))
    def test_reference_equivalence_and_determinism(self):
        qs=self.pool+[record('S1-q','Shop Alpha','Long Road')]
        build_catalog([self.path],self.cache,2); c=Catalog(self.cache,[self.path])
        try:
            compact,info=c.index(qs); loaded,info=c.index(list(reversed(qs)))
            self.assertTrue(info['cache_hit']); self.assertEqual(compact.postings,loaded.postings)
            freq=Frequencies.fit(self.pool); base=build_index(self.pool,CONFIGURATIONS['C']); extra=AdditionalIndex(self.pool,freq)
            lookup={r.entity_id:r for r in self.pool}
            for q in qs:
                ref=generate_candidates(q,base,CONFIGURATIONS['C']); a,b=extra.candidates(q); ref.update(a);ref.update(b)
                opt={c.get(i).entity_id for i in compact.candidates(q)}
                self.assertEqual(ref,opt)
                qw=prepare_weighted(q,freq)
                rank=lambda ids:sorted(ids,key=lambda i:(-improved_score(improved_signals(qw,prepare_weighted(lookup[i],freq),freq)),i))[:50]
                self.assertEqual(rank(ref),rank(opt))
        finally:c.close()
    def test_rescue_frequency_limits_across_chunks(self):
        qs=[record('S1-q','Rare Shared','Long Road'),record('S1-name','Rare Shared'),record('S1-address','','Long Road')]
        pool=[record(f'S2-{i}','Changed Shared','Long Road') for i in range(301)]
        for n in (199,200,201,299,300,301):
            compact=CompactIndex(requested_keys(qs)); selected=pool[:n]
            for i,r in enumerate(selected):compact.add(i,r)
            freq=Frequencies.fit(selected); extra=AdditionalIndex(selected,freq); base=build_index(selected,CONFIGURATIONS['C'])
            for q in qs:
                ref=generate_candidates(q,base,CONFIGURATIONS['C']);a,b=extra.candidates(q);ref.update(a);ref.update(b)
                self.assertEqual(ref,{selected[i].entity_id for i in compact.candidates(q)})
    def test_duplicate_ids_do_not_publish_cache(self):
        with self.path.open('a',encoding='utf-8') as f:f.write('S2-1\tDuplicate\t\tX\n')
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):build_catalog([self.path],self.cache,1)
        self.assertFalse(valid(self.cache,signature([self.path])))
    def test_uint32_guard(self):
        c=CompactIndex([])
        with self.assertRaises(ValueError):c.add(2**32,self.pool[0])

if __name__=='__main__':unittest.main()
