"""Holdout 2 entry point; reference helpers retained for bounded comparisons."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[_name]='4'
import csv, json, time
from itertools import islice
from pathlib import Path
from collections import defaultdict
import numpy as np, joblib
from src.candidates import Record, read_rows, load_truth, build_index, generate_candidates, CONFIGURATIONS
from src.candidate_improvements import Frequencies, AdditionalIndex, prepare_weighted, improved_signals, improved_score
from src.build_features import load_frequencies
from src.features import FEATURE_NAMES, pair_features, label_pair
from src.evaluate import macro_f05
from src.failure_analysis import input_manifest
from src.ranking import peak_memory_mib

TRAIN=Path('dataset/train'); OUT=Path('output/holdout2'); K=50; SEED=42
def queries():
 rows=list(islice(read_rows(TRAIN/'train_source1.tsv'),80000)); return [Record.from_row(r) for r in rows[60000:80000]]
def prior():
 ids=set()
 with Path('output/pair_features/s1_split.csv').open(encoding='utf-8') as f: ids.update(r['source1_entity_id'] for r in csv.DictReader(f))
 for r in islice(read_rows(TRAIN/'train_source1.tsv'),60000): ids.add(r['entity_id'])
 return ids
def full_catalog(source):
 return [Record.from_row(r) for r in read_rows(TRAIN/f'train_source{source}.tsv')]
def pairs(qs,pool,truth,freq):
 cf=Frequencies.fit(pool); idx=build_index(pool,CONFIGURATIONS['C']); extra=AdditionalIndex(pool,cf); lookup={r.entity_id:r for r in pool}; out=[]
 for num,q in enumerate(qs,1):
  ids=generate_candidates(q,idx,CONFIGURATIONS['C']); a,b=extra.candidates(q); ids.update(a);ids.update(b); qw=prepare_weighted(q,freq)
  # Prepare only the posting-set candidates for this query; the full catalog
  # remains available for blocking, but millions of unused records are never
  # expanded into weighted feature objects.
  scored=sorted(((i,improved_score(improved_signals(qw,prepare_weighted(lookup[i],freq),freq))) for i in ids),key=lambda z:(-z[1],z[0]))[:K]
  for rank,(i,score) in enumerate(scored,1):
   cp=prepare_weighted(lookup[i],freq)
   out.append((q.entity_id,i,label_pair(q.entity_id,i,truth),pair_features(q,lookup[i],freq,score,rank,qw,cp)))
  if num%1000==0: print(f'Generated {num:,}/{len(qs):,}',flush=True)
 return out
def metrics(rows,qs,truth,bundle,threshold):
 names=bundle['feature_names']; base=np.asarray([[f[n] for n in names] for _,_,_,f in rows],dtype=np.float32); y=np.asarray([r[2] for r in rows],dtype=np.uint8); p=bundle['model'].predict_proba(base)[:,1]
 pred={q.entity_id:set() for q in qs}; cand={q.entity_id:set() for q in qs}
 for (s1,c,label,_),prob in zip(rows,p): cand[s1].add(c); pred[s1].add(c) if prob>=threshold else None
 truthlinks=sum(map(len,truth.values())); tp=sum(len(pred[s]&truth[s]) for s in pred); found=int(y.sum()); n=len(qs); count=sum(map(len,pred.values()))
 m={'candidate_recall':found/truthlinks,'macro_f05':macro_f05(truth,pred),'pair_precision':tp/count if count else 0,'pair_recall_available_candidates':tp/found if found else 0,'full_ground_truth_link_recall':tp/truthlinks,'predicted_links':count,'average_predictions_per_s1':count/n,'predicted_singleton_percentage':100*sum(len(v)==1 for v in pred.values())/n,'correct_singleton_count':sum(len(pred[s])==1 and pred[s]==truth[s] for s in pred),'incorrect_singleton_count':sum(len(pred[s])==1 and pred[s]!=truth[s] for s in pred),'zero_prediction_count':sum(not v for v in pred.values()),'true_links':truthlinks,'candidate_true_links':found}
 by={}
 for src in ('S2','S3'):
  t={s:{i for i in truth[s] if i.startswith(src+'-')} for s in truth}; pr={s:{i for i in pred[s] if i.startswith(src+'-')} for s in truth}; total=sum(map(len,t.values())); by[src]={'true_links':total,'macro_f05':macro_f05(t,pr),'full_ground_truth_link_recall':sum(len(pr[s]&t[s]) for s in t)/max(1,total),'candidate_recall':sum(1 for r in rows if r[2] and r[1].startswith(src+'-'))/max(1,total)}
 m['by_source']=by; return m
def main():
 from src.scalable_holdout2 import run, OUT as scalable_output
 from src.resource_guard import watchdog
 from threadpoolctl import threadpool_limits
 scalable_output.mkdir(parents=True,exist_ok=True)
 watchdog(scalable_output/'resource_stop.json')
 with threadpool_limits(limits=4): run()
if __name__=='__main__': main()
