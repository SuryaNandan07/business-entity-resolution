"""Bounded, measured reference-stage profile; never opens competition files."""
import gc
import json
import time
from collections import defaultdict
from itertools import islice
from pathlib import Path
from src.resource_guard import check, watchdog
from src.candidates import Record, read_rows, build_index, generate_candidates, CONFIGURATIONS
from src.candidate_improvements import Frequencies, AdditionalIndex, char_ngrams, prepare_weighted, improved_score, improved_signals
from src.features import pair_features


def main():
    Path('output/scalability').mkdir(parents=True,exist_ok=True)
    watchdog('output/scalability/resource_stop.json')
    results = {}
    def measure(name, function):
        gc.collect()
        before = check()['rss']
        start = time.perf_counter()
        value = function()
        rss = check()['rss']
        results[name] = dict(seconds=time.perf_counter()-start, rss_mib=rss/2**20,
                             rss_delta_mib=(rss-before)/2**20)
        print(name, results[name], flush=True)
        Path('output/scalability/reference_profile.json').write_text(json.dumps(results,indent=2))
        return value
    raw = measure('reading', lambda: list(islice(read_rows('dataset/train/train_source2.tsv'), 50000)) + list(islice(read_rows('dataset/train/train_source3.tsv'), 50000)))
    pool = measure('normalization', lambda: [Record.from_row(r) for r in raw])
    del raw
    freq = measure('document_frequencies', lambda: Frequencies.fit(pool))
    indexes = {}
    for rule in CONFIGURATIONS['C']:
        indexes[rule] = measure(rule, lambda: build_index(pool, (rule,)))
    extra = measure('rescue_index', lambda: AdditionalIndex(pool, freq))
    # Isolated replays of the reference loops separate the three rescue costs.
    # They are diagnostic repeats, not additional production stages.
    def isolated_rescue(field,limit):
        result=defaultdict(list)
        for r in pool:
            if not r.country:continue
            tokens=char_ngrams(r.name) if field=='grams' else set(getattr(r,field).split())
            for token in tokens:
                if field=='address' and token.isdecimal():continue
                if getattr(freq,field)[r.country,token]<=limit:result[r.country,token].append(r.entity_id)
        return result
    for field,limit in [('name',200),('address',300),('grams',200)]:
        replay=measure('isolated_rescue_'+field,lambda:isolated_rescue(field,limit))
        del replay;gc.collect()
    for field in ('name','address','grams'):
        counts=getattr(freq,field)
        results['frequency_'+field+'_structure']=dict(keys=len(counts),document_occurrences=sum(counts.values()),maximum=max(counts.values(),default=0))
    for name, index in list(indexes.items()) + [('rare_name', extra.name), ('rescue_address', extra.address), ('rescue_trigrams', extra.grams)]:
        lengths = [len(v) for v in index.values()]
        results[name + '_structure'] = dict(keys=len(lengths), postings=sum(lengths), average=sum(lengths)/max(1,len(lengths)), maximum=max(lengths,default=0))
    qs = [Record.from_row(r) for r in islice(read_rows('dataset/train/train_source1.tsv'), 100)]
    index = {k:v for part in indexes.values() for k,v in part.items()}
    def retrieve():
        out=[]
        for q in qs:
            ids=generate_candidates(q,index,CONFIGURATIONS['C']); a,b=extra.candidates(q)
            ids.update(a); ids.update(b); out.append(ids)
        return out
    candidates=measure('candidate_union', retrieve)
    lookup={r.entity_id:r for r in pool}
    def rank():
        out=[]
        for q,ids in zip(qs,candidates):
            qw=prepare_weighted(q,freq)
            out.append(sorted(((i,improved_score(improved_signals(qw,prepare_weighted(lookup[i],freq),freq))) for i in ids),key=lambda x:(-x[1],x[0]))[:50])
        return out
    ranked=measure('ranking',rank)
    measure('features',lambda: [pair_features(q,lookup[i],freq,s,n) for q,rs in zip(qs,ranked) for n,(i,s) in enumerate(rs,1)])
    results['scope']={'targets':len(pool),'queries':len(qs),'note':'Bounded reference profile, not a full-catalog measurement'}
    out=Path('output/scalability'); out.mkdir(parents=True,exist_ok=True)
    (out/'reference_profile.json').write_text(json.dumps(results,indent=2))

if __name__ == '__main__':
    main()
