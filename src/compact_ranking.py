"""Vector score screening followed by the unchanged exact Python scorer.

Positive IDF sums incur at most O(n * machine epsilon) relative summation error.
A deliberately conservative band (>=1e-9, plus the 12-decimal quantization
error) retains every possible Top-K contender. All retained contenders are
rescored with improved_signals/improved_score and entity-ID tie breaking.
No approximate score is emitted as a ranking score or model feature.
"""
from array import array
from collections import defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
import pickle
import time
import numpy as np
from src.ranking import prepare
from src.candidate_improvements import prepare_weighted, improved_signals, improved_score

def keys_for(queries):
    keys=set()
    for q in queries:
        if not q.country:continue
        p=prepare(q)
        for kind,tokens in [('name',p.name_tokens),('address',p.address_tokens),('bigram',p.name_bigrams)]:
            keys.update((kind,q.country,t) for t in tokens)
        if q.name:keys.add(('exact',q.country,q.name))
    return sorted(keys)

class RankingIndex:
    def __init__(self,keys,count):
        self.postings={k:array('I') for k in keys}
        self.name_total=np.zeros(count,dtype=np.float64)
        self.address_total=np.zeros(count,dtype=np.float64)
        self.gram_count=np.zeros(count,dtype=np.uint32)

    def add(self,i,r,freq):
        p=prepare(r)
        self.name_total[i]=math.fsum(freq.idf(r.country,t,'name') for t in p.name_tokens)
        self.address_total[i]=math.fsum(freq.idf(r.country,t,'address') for t in p.address_tokens)
        self.gram_count[i]=len(p.name_bigrams)
        if not r.country:return
        for kind,tokens in [('name',p.name_tokens),('address',p.address_tokens),('bigram',p.name_bigrams)]:
            for token in tokens:
                postings=self.postings.get((kind,r.country,token))
                if postings is not None:postings.append(i)
        postings=self.postings.get(('exact',r.country,r.name))
        if postings is not None:postings.append(i)

    @classmethod
    def cached(cls,catalog,queries,freq,provenance,guard):
        from src.disk_ranking import cached_ranking
        return cached_ranking(cls,catalog,queries,freq,provenance,guard)

class VectorRanker:
    def __init__(self,index,freq):
        self.index=index;self.freq=freq;n=len(index.name_total)
        self.nmass=np.zeros(n);self.amass=np.zeros(n);self.numeric=np.zeros(n)
        self.grams=np.zeros(n,dtype=np.uint32);self.words=np.zeros(n,dtype=np.uint32)
        self.distinct=np.zeros(n,dtype=np.bool_);self.exact=np.zeros(n,dtype=np.bool_)

    def approximate(self,q,ids,qw):
        for values in (self.nmass,self.amass,self.numeric,self.grams,self.words,self.distinct,self.exact):values[ids]=0
        def postings(kind,token):
            data=self.index.postings.get((kind,q.country,token))
            return np.frombuffer(data,dtype=np.uint32) if data is not None and len(data) else np.empty(0,dtype=np.uint32)
        for t,w in qw.name_weights.items():self.nmass[postings('name',t)]+=w
        maximum_idf=1+math.log(self.freq.documents[q.country]+1)
        for t,w in qw.address_weights.items():
            p=postings('address',t);self.amass[p]+=w
            if t.isdecimal():self.numeric[p]=np.maximum(self.numeric[p],w/maximum_idf)
            else:
                self.words[p]+=1
                if not self.freq.common(q.country,t,'address'):self.distinct[p]=True
        for gram in qw.basic.name_bigrams:self.grams[postings('bigram',gram)]+=1
        if q.name:self.exact[postings('exact',q.name)]=True
        def divide(numerator,denominator):
            return np.divide(numerator,denominator,out=np.zeros(len(ids)),where=denominator>0)
        name_dice=divide(2*self.nmass[ids],qw.name_total+self.index.name_total[ids])
        char_dice=divide(2*self.grams[ids],len(qw.basic.name_bigrams)+self.index.gram_count[ids])
        name=np.maximum(self.exact[ids],.65*name_dice+.35*char_dice)
        addr_dice=divide(2*self.amass[ids],qw.address_total+self.index.address_total[ids])
        containment=divide(self.amass[ids],np.minimum(qw.address_total,self.index.address_total[ids]))
        address=np.where((self.words[ids]>=2)&self.distinct[ids],.65*addr_dice+.35*containment,addr_dice)
        numeric=self.numeric[ids]
        return np.maximum(.65*name+.30*address+.05*numeric,.85*address+.10*name+.05*numeric)

    def rank(self,q,candidates,lookup,k=50):
        qw=prepare_weighted(q,self.freq);ids=np.fromiter(candidates,dtype=np.uint32,count=len(candidates))
        if len(ids)>k:
            scores=self.approximate(q,ids,qw)
            if not np.isfinite(scores).all():raise ValueError('Non-finite ranking score')
            n=len(qw.name_weights)+len(qw.address_weights)+10
            band=max(1e-9,128*np.finfo(np.float64).eps*n+1e-12)
            cutoff=np.partition(scores,len(scores)-k)[len(scores)-k]
            ids=ids[scores>=cutoff-band]
        def exact():
            for i in ids:
                r=lookup(int(i));rw=prepare_weighted(r,self.freq)
                yield int(i),r.entity_id,improved_score(improved_signals(qw,rw,self.freq))
        ranked=heapq.nsmallest(k,exact(),key=lambda p:(-p[2],p[1]))
        return ranked,len(ids)
