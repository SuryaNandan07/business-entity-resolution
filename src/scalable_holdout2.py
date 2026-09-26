"""Phase 4A.1: exact retrieval and frozen comparison, training files only."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='4'
import argparse
import csv
from collections import Counter
from functools import lru_cache
import gc
import hashlib
import heapq
from itertools import islice
import json
from pathlib import Path
import sqlite3
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from src.catalog_cache import Catalog, CompactIndex, requested_keys, build_catalog, signature
from src.resource_guard import check, watchdog, memory
from src.candidates import Record, read_rows, load_truth, load_candidate_pool, build_index, generate_candidates, CONFIGURATIONS
from src.candidate_improvements import Frequencies, AdditionalIndex, prepare_weighted, improved_signals, improved_score
from src.build_features import load_frequencies
from src.features import FEATURE_NAMES, pair_features
from src.evaluate import macro_f05, entity_f05
from src.holdout2 import queries, prior
from src.compact_ranking import RankingIndex, VectorRanker, keys_for

TRAIN=Path('dataset/train')
PATHS=[TRAIN/f'train_source{s}.tsv' for s in (2,3)]
CACHE=Path('output/cache/train_catalog')
OUT=Path('output/scalability')

def feature_provenance():
    """Invalidate resumed features after a feature/ranking/frequency change."""
    paths=[Path(__file__).with_name(n) for n in ('features.py','ranking.py','candidate_improvements.py','compact_ranking.py','disk_ranking.py')]
    paths.append(Path('output/pair_features/training_frequencies.jsonl.gz'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def save(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/name; tmp=path.with_suffix('.partial'); tmp.write_text(json.dumps(data,indent=2),encoding='utf-8'); tmp.replace(path)

def verify():
    start=time.perf_counter()
    qs=[Record.from_row(r) for r in islice(read_rows(TRAIN/'train_source1.tsv'),200)]
    truth=load_truth(TRAIN/'train_ground_truth.tsv',{q.entity_id for q in qs})
    required=set().union(*truth.values()); pool=[]
    for source,path in zip((2,3),PATHS):
        records,_=load_candidate_pool(path,{i for i in required if i.startswith(f'S{source}-')},20000,42+source)
        pool.extend(records); del records; check()
    freq=Frequencies.fit(pool); base=build_index(pool,CONFIGURATIONS['C']); extra=AdditionalIndex(pool,freq)
    opt=CompactIndex(requested_keys(qs))
    for i,r in enumerate(pool): opt.add(i,r)
    frozen=load_frequencies(Path('output/pair_features/training_frequencies.jsonl.gz'))
    # Exercise the real disk-backed path on exactly the reference target pool.
    sample_path=OUT/'verification_targets.tsv'
    with sample_path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['entity_id','business_name','business_address','country'])
        writer.writerows((r.entity_id,r.business_name,r.business_address,r.country) for r in pool)
    build_catalog([sample_path],OUT/'verification_cache',guard=check)
    verification_catalog=Catalog(OUT/'verification_cache',[sample_path])
    ranking_index,_=RankingIndex.cached(verification_catalog,qs,frozen,feature_provenance(),check)
    ranker=VectorRanker(ranking_index,frozen)
    lookup={r.entity_id:r for r in pool}
    result=dict(queries=len(qs),targets=len(pool),broad_changed_entities=0,top50_changed_entities=0,
                broad_added=0,broad_removed=0,top50_added=0,top50_removed=0,true_links_gained=0,true_links_lost=0,
                reference_true_links=0,optimized_true_links=0,true_links=sum(map(len,truth.values())),
                fast_top50_changed_entities=0,reference_ranking_seconds=0,fast_ranking_seconds=0)
    for q in qs:
        check(); ref=generate_candidates(q,base,CONFIGURATIONS['C']);a,b=extra.candidates(q);ref.update(a);ref.update(b)
        actual={pool[i].entity_id for i in opt.candidates(q)}
        result['broad_changed_entities']+=ref!=actual;result['broad_added']+=len(actual-ref);result['broad_removed']+=len(ref-actual)
        qw=prepare_weighted(q,frozen)
        def rank(ids):
            return heapq.nsmallest(50,((i,improved_score(improved_signals(qw,prepare_weighted(lookup[i],frozen),frozen))) for i in ids),key=lambda p:(-p[1],p[0]))
        t=time.perf_counter();left=rank(ref);result['reference_ranking_seconds']+=time.perf_counter()-t
        right=rank(actual); l={i for i,_ in left};r={i for i,_ in right}
        t=time.perf_counter();fast,_=ranker.rank(q,opt.candidates(q),verification_catalog.get)
        result['fast_ranking_seconds']+=time.perf_counter()-t
        result['fast_top50_changed_entities']+=left!=[(eid,score) for _,eid,score in fast]
        result['top50_changed_entities']+=left!=right;result['top50_added']+=len(r-l);result['top50_removed']+=len(l-r)
        result['true_links_gained']+=len((r-l)&truth[q.entity_id]);result['true_links_lost']+=len((l-r)&truth[q.entity_id])
        result['reference_true_links']+=len(l&truth[q.entity_id]);result['optimized_true_links']+=len(r&truth[q.entity_id])
    result['reference_recall']=result['reference_true_links']/result['true_links']
    result['optimized_recall']=result['optimized_true_links']/result['true_links']
    result['recall_difference']=result['optimized_recall']-result['reference_recall']
    result['seconds']=time.perf_counter()-start;result['peak_memory']=memory()
    result['signature']=signature(PATHS)
    result['feature_provenance']=feature_provenance()
    save('equivalence.json',result);print(json.dumps(result),flush=True)
    ranking_index.postings.stream.close();verification_catalog.close()
    if result['broad_changed_entities'] or result['top50_changed_entities'] or result['fast_top50_changed_entities']: raise ValueError('Equivalence failed')

def summarize(truth,pred,cand):
    total=sum(map(len,truth.values()));found=sum(len(cand[s]&truth[s]) for s in truth)
    tp=sum(len(pred[s]&truth[s]) for s in truth);count=sum(len(pred[s]) for s in truth);n=len(truth)
    return dict(s1_count=n,macro_f05=macro_f05(truth,pred),pair_precision=tp/count if count else 0,
        pair_recall=tp/found if found else 0,full_ground_truth_link_recall=tp/total if total else None,
        candidate_recall=found/total if total else None,predicted_links=count,average_predictions_per_s1=count/n,
        predicted_singleton_percentage=100*sum(len(pred[s])==1 for s in truth)/n,
        correct_singletons=sum(len(pred[s])==1 and pred[s]==truth[s] for s in truth),
        incorrect_singletons=sum(len(pred[s])==1 and pred[s]!=truth[s] for s in truth),
        zero_prediction_s1_count=sum(not pred[s] for s in truth),true_links=total,candidate_true_links=found)

def run(skip_inference=False):
    started=time.perf_counter();check()
    equivalence=json.loads((OUT/'equivalence.json').read_text())
    if equivalence['signature']!=json.loads(json.dumps(signature(PATHS))) or equivalence['broad_changed_entities'] or equivalence['top50_changed_entities']:
        raise ValueError('Current reference equivalence required')
    if equivalence.get('fast_top50_changed_entities')!=0 or equivalence.get('feature_provenance')!=feature_provenance():
        raise ValueError('Current accelerated-ranking equivalence required')
    qs=queries(); ids={q.entity_id for q in qs};old=prior()
    if len(qs)!=20000 or len(ids)!=20000 or ids&old:raise ValueError('Invalid or overlapping Holdout 2')
    meta=build_catalog(PATHS,CACHE,guard=check);gc.collect()
    if meta['records']!=5034616+5285603:raise ValueError('Incomplete catalog')
    catalog=Catalog(CACHE,PATHS);idx,index_info=catalog.index(qs,guard=check)
    # Prove both cache loading paths and measure them without a duplicate index.
    del idx;gc.collect();idx,load_info=catalog.index(qs,guard=check)
    freq=load_frequencies(Path('output/pair_features/training_frequencies.jsonl.gz'))
    provenance=feature_provenance()
    ranking_index,ranking_info=RankingIndex.cached(catalog,qs,freq,provenance,check)
    del ranking_index;gc.collect()
    ranking_index,ranking_load_info=RankingIndex.cached(catalog,qs,freq,provenance,check)
    ranker=VectorRanker(ranking_index,freq)
    model_paths={'hgb':Path('output/baseline_models/hist_gradient_boosting.joblib'),'lightgbm':Path('output/phase4a/lightgbm.joblib')}
    stamp=hashlib.sha256(json.dumps([meta['signature'],[(q.entity_id,q.name,q.address,q.country) for q in qs],list(FEATURE_NAMES),provenance],sort_keys=True).encode()).hexdigest()
    db=sqlite3.connect(OUT/f'holdout2-{stamp}.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS completed (s1 TEXT PRIMARY KEY, timings TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS pairs (s1 TEXT, candidate TEXT, features BLOB, PRIMARY KEY(s1,candidate))')
    done={r[0] for r in db.execute('SELECT s1 FROM completed')}
    @lru_cache(maxsize=2000)
    def prepared(i):
        r=catalog.get(i);return r,prepare_weighted(r,freq)
    holdout_started=time.perf_counter()
    for number,q in enumerate(qs,1):
        if q.entity_id in done:continue
        check();times={};t=time.perf_counter();candidates=idx.candidates(q);times['candidate_generation']=time.perf_counter()-t
        qw=prepare_weighted(q,freq);t=time.perf_counter()
        ranked,contenders=ranker.rank(q,candidates,catalog.get);times['ranking']=time.perf_counter()-t
        times['exact_rescored_candidates']=contenders
        t=time.perf_counter();rows=[]
        for rank,(i,eid,score) in enumerate(ranked,1):
            r,rw=prepared(i);f=pair_features(q,r,freq,score,rank,qw,rw)
            rows.append((q.entity_id,eid,np.asarray([f[n] for n in FEATURE_NAMES],dtype=np.float32).tobytes()))
        times['features']=time.perf_counter()-t;times['broad_candidates']=len(candidates)
        db.executemany('INSERT INTO pairs VALUES (?,?,?)',rows);db.execute('INSERT INTO completed VALUES (?,?)',(q.entity_id,json.dumps(times)));db.commit()
        if number%25==0:
            print(f'Holdout {number:,}/20,000; last broad={len(candidates):,}, ranking={times["ranking"]:.2f}s; RSS={memory()["rss"]/2**20:.0f} MiB',flush=True)
            gc.collect()
    prepared.cache_clear();del idx,ranker,ranking_index;gc.collect()
    # Labels are first read after candidate ranking has completed.
    truth=load_truth(TRAIN/'train_ground_truth.tsv',ids);cand={s:set() for s in ids}
    for s,i in db.execute('SELECT s1,candidate FROM pairs'):cand[s].add(i)
    timings=Counter()
    for row in db.execute('SELECT timings FROM completed'):timings.update(json.loads(row[0]))
    generation_wall=time.perf_counter()-holdout_started
    previous_path=OUT/'holdout2_candidates.json'
    if len(done)==len(qs) and previous_path.exists():
        previous=json.loads(previous_path.read_text())
        if previous.get('checkpoint_stamp')==stamp:generation_wall=previous['holdout_processing_wall_seconds']
    def candidate_metrics(subset,source=None):
        true_count=found=pair_count=0
        for s in subset:
            t={i for i in truth[s] if source is None or i.startswith(source)}
            c={i for i in cand[s] if source is None or i.startswith(source)}
            true_count+=len(t);found+=len(t&c);pair_count+=len(c)
        return dict(s1_count=len(subset),true_links=true_count,candidate_true_links=found,
                    candidate_recall=found/true_count if true_count else None,candidate_pairs=pair_count)
    candidate_report=dict(checkpoint_stamp=stamp,checkpoint_path=str(OUT/f'holdout2-{stamp}.sqlite'),
        k=50,overlap_count=len(ids&old),prior_ids=len(old),catalog=meta,index=index_info,ranking_index=ranking_info,
        feature_provenance=provenance,metrics=candidate_metrics(ids),zero_candidate_s1_count=sum(not c for c in cand.values()),
        maximum_candidates_per_s1=max(map(len,cand.values())),timings=dict(timings),
        holdout_processing_wall_seconds=generation_wall,cache_load_seconds=catalog.load_seconds+load_info['load_seconds']+ranking_load_info['load_seconds'],
        peak_memory=memory(),cache_bytes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file()),
        by_source={s:candidate_metrics(ids,s+'-') for s in ('S2','S3')},
        by_country={c:candidate_metrics({q.entity_id for q in qs if q.country==c}) for c in sorted({q.country for q in qs})},
        model_sha256={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in model_paths.items()},
        inference_status='Skipped at user request pending an approved Python environment; Windows Application Control blocked sklearn.utils.arrayfuncs')
    save('holdout2_candidates.json',candidate_report)
    if skip_inference:
        db.close();catalog.close();print(json.dumps(candidate_report),flush=True);return
    inference_started=time.perf_counter()
    bundles={n:joblib.load(p) for n,p in model_paths.items()}
    for b in bundles.values():
        if any(n not in FEATURE_NAMES for n in b['feature_names']):raise ValueError('Frozen feature schema mismatch')
    predictions={n:{s:set() for s in ids} for n in bundles};inference=Counter();thresholds={'hgb':.670,'lightgbm':.585}
    cursor=db.execute('SELECT s1,candidate,features FROM pairs ORDER BY s1,candidate')
    while True:
        check();batch=cursor.fetchmany(10000)
        if not batch:break
        x=np.asarray([np.frombuffer(r[2],dtype=np.float32) for r in batch])
        for s,i,_ in batch:cand[s].add(i)
        for name,b in bundles.items():
            cols=[FEATURE_NAMES.index(n) for n in b['feature_names']];t=time.perf_counter()
            with threadpool_limits(limits=4):p=b['model'].predict_proba(x[:,cols])[:,1]
            inference[name]+=time.perf_counter()-t
            for (s,i,_),prob in zip(batch,p):
                if prob>=thresholds[name]:predictions[name][s].add(i)
    models={}
    for name,pred in predictions.items():
        m=summarize(truth,pred,cand);m['threshold']=thresholds[name];m['by_source']={};m['by_country']={}
        for source in ('S2-','S3-'):
            filter_source=lambda data:{s:{i for i in v if i.startswith(source)} for s,v in data.items()}
            m['by_source'][source[:2]]=summarize(filter_source(truth),filter_source(pred),filter_source(cand))
        for country in sorted({q.country for q in qs}):
            subset={q.entity_id for q in qs if q.country==country}
            m['by_country'][country]=summarize({s:truth[s] for s in subset},{s:pred[s] for s in subset},{s:cand[s] for s in subset})
        models[name]=m
    db.close();catalog.close()
    cached_seconds=sum(timings[k] for k in ('candidate_generation','ranking','features'))+sum(inference.values())
    differences=np.asarray([entity_f05(truth[s],predictions['lightgbm'][s])-entity_f05(truth[s],predictions['hgb'][s]) for s in sorted(ids)])
    delta=float(differences.mean());error=1.96*float(differences.std(ddof=1))/len(ids)**.5
    if feature_provenance()!=provenance:raise RuntimeError('Feature implementation or frozen frequencies changed during evaluation')
    report=dict(models=models,k=50,s1_count=len(qs),overlap_count=len(ids&old),prior_ids=len(old),catalog=meta,index=index_info,ranking_index=ranking_info,feature_provenance=provenance,
        cache_load_seconds=catalog.load_seconds+load_info['load_seconds']+ranking_load_info['load_seconds'],timings=dict(timings),inference_seconds=dict(inference),
        total_this_invocation_seconds=time.perf_counter()-started,completed_processing_seconds=cached_seconds,
        holdout_processing_wall_seconds=generation_wall+time.perf_counter()-inference_started,
        peak_memory=memory(),cache_bytes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file()),
        competition_1_7m_estimate_seconds=cached_seconds/len(qs)*1700000,
        estimate_note='Measured Holdout 2 throughput extrapolation; excludes new-batch posting construction and is not guaranteed',
        model_sha256={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in model_paths.items()},
        comparison={'macro_f05_delta_lightgbm_minus_hgb':delta,'paired_normal_95pct_interval':[delta-error,delta+error],
                    'pair_precision_delta_lightgbm_minus_hgb':models['lightgbm']['pair_precision']-models['hgb']['pair_precision']},
        decision='Keep HGB pending review; no automatic adoption')
    save('holdout2_report.json',report);print(json.dumps(report),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['verify','run','candidates']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);watchdog(OUT/'resource_stop.json')
    with threadpool_limits(limits=4):
        if args.stage=='verify':verify()
        else:run(skip_inference=args.stage=='candidates')

if __name__=='__main__':main()
