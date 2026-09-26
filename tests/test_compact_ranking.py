import heapq
import random
import unittest
import tempfile
import csv
from pathlib import Path
from src.catalog_cache import build_catalog, Catalog
from src.disk_ranking import cached_ranking
from src.candidates import Record
from src.candidate_improvements import Frequencies, prepare_weighted, improved_signals, improved_score
from src.compact_ranking import RankingIndex, VectorRanker, keys_for

def r(i,name,address=''):
    return Record.from_row(dict(entity_id=f'S2-{i:05}',business_name=name,business_address=address,country='X'))

class CompactRankingTests(unittest.TestCase):
    def compare(self,pool,qs):
        freq=Frequencies.fit(pool);idx=RankingIndex(keys_for(qs),len(pool))
        for i,record in enumerate(pool):idx.add(i,record,freq)
        ranker=VectorRanker(idx,freq)
        for q in qs:
            qw=prepare_weighted(q,freq)
            ref=heapq.nsmallest(50,((i,record.entity_id,improved_score(improved_signals(qw,prepare_weighted(record,freq),freq))) for i,record in enumerate(pool)),key=lambda x:(-x[2],x[1]))
            actual,_=ranker.rank(q,set(range(len(pool))),pool.__getitem__)
            self.assertEqual(ref,actual)
    def test_random_weighted_and_unicode_equivalence(self):
        rng=random.Random(19);names=['alpha','beta','東京','café','store','company','a','b'];addresses=['1','００','2','road','hill','street','north','south']
        pool=[r(i,' '.join(rng.choices(names,k=rng.randrange(0,6))),' '.join(rng.choices(addresses,k=rng.randrange(0,8)))) for i in range(1200)]
        self.compare(pool,pool[:30]+[r(9999,'βeta 東京','００ street')])
    def test_ties_missing_and_repeated_queries(self):
        pool=[r(i,'Same Shop','') for i in range(200)]+[r(i,'Other','1 Long Road') for i in range(200,400)]
        self.compare(pool,[r(900,'Same Shop'),r(901,'','1 Long Road'),r(902,''),r(903,'Same Shop')])
    def test_disk_cache_resume_and_equivalence(self):
        pool=[r(i,'Alpha Shop' if i%2 else 'Beta Shop',f'{i%7} Long Road') for i in range(80)]
        qs=pool[:3];freq=Frequencies.fit(pool)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.tsv'
            with source.open('w',encoding='utf-8',newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['entity_id','business_name','business_address','country'])
                w.writerows((p.entity_id,p.business_name,p.business_address,p.country) for p in pool)
            build_catalog([source],root/'cache',7);catalog=Catalog(root/'cache',[source])
            calls=0
            def stop():
                nonlocal calls
                calls+=1
                if calls==2:raise MemoryError('Simulated interruption')
            with self.assertRaises(MemoryError):cached_ranking(RankingIndex,catalog,qs,freq,{},stop,7)
            index,info=cached_ranking(RankingIndex,catalog,qs,freq,{},lambda:None,7)
            try:
                ranker=VectorRanker(index,freq)
                for q in qs:
                    qw=prepare_weighted(q,freq)
                    ref=heapq.nsmallest(50,((i,p.entity_id,improved_score(improved_signals(qw,prepare_weighted(p,freq),freq))) for i,p in enumerate(pool)),key=lambda p:(-p[2],p[1]))
                    self.assertEqual(ref,ranker.rank(q,set(range(80)),pool.__getitem__)[0])
                again,hit=cached_ranking(RankingIndex,catalog,qs,freq,{},lambda:None,7)
                self.assertTrue(hit['cache_hit']);again.postings.stream.close()
            finally:index.postings.stream.close();catalog.close()

if __name__=='__main__':unittest.main()
