"""Phase 4A validation-only model comparison on fixed Phase 3A features."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
from src.modeling import load_pairs, tune_threshold, training_columns
from src.failure_analysis import input_manifest

F=Path('output/pair_features'); O=Path('output/phase4a'); O.mkdir(parents=True,exist_ok=True)
def main():
 t0=time.perf_counter(); schema=json.loads((F/'schema.json').read_text()); prior=json.loads((F/'report.json').read_text()); names=schema['feature_columns']
 split=pd.read_csv(F/'s1_split.csv'); tr=set(split.loc[split.split=='train','source1_entity_id']); va=set(split.loc[split.split=='validation','source1_entity_id'])
 x,y,_=load_pairs(F/'train_features.csv.gz',names,prior['balances']['train']['candidate_pairs'],tr); v,vy,meta=load_pairs(F/'validation_features.csv.gz',names,prior['balances']['validation']['candidate_pairs'],va,True)
 selected=training_columns(x,names); names=[names[i] for i in selected]; x=x[:,selected]; v=v[:,selected]
 order=sorted(va); lookup={s:i for i,s in enumerate(order)}; ei=np.array([lookup[s] for s,_ in meta]); from src.candidates import load_truth
 truth=load_truth(Path('dataset/train/train_ground_truth.tsv'),va); tc=np.array([len(truth[s]) for s in order])
 models={
  'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=50,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=42),
  'logistic_unweighted':make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,random_state=42)),
  'lightgbm':LGBMClassifier(n_estimators=150,num_leaves=31,max_depth=-1,learning_rate=.05,min_child_samples=50,reg_lambda=1.,verbosity=-1,n_jobs=4,random_state=42),
  'catboost':CatBoostClassifier(iterations=150,depth=7,learning_rate=.08,l2_leaf_reg=3.,loss_function='Logloss',verbose=False,thread_count=4,random_seed=42),
 }
 report={'train_s1':len(tr),'validation_s1':len(va),'train_pairs':len(y),'validation_pairs':len(vy),'feature_names':names,'licenses':{'scikit-learn':'BSD-3-Clause (permissive)','LightGBM':'MIT','CatBoost':'Apache-2.0'},'models':{}}
 best=None
 for n,m in models.items():
  st=time.perf_counter(); m.fit(x,y); p=m.predict_proba(v)[:,1]; b,curve=tune_threshold(p,vy,ei,tc)
  report['models'][n]={'best':b,'roc_auc':float(roc_auc_score(vy,p)),'average_precision':float(average_precision_score(vy,p)),'fit_seconds':time.perf_counter()-st}
  pd.DataFrame(curve).to_csv(O/(n+'_thresholds.csv'),index=False); joblib.dump({'model':m,'feature_names':names,'threshold':b['threshold']},O/(n+'.joblib'),compress=3)
  if best is None or b['macro_f05']>best[1]: best=(n,b['macro_f05'])
  print(n,json.dumps(b),flush=True)
 report['selected_challenger']=best[0] if best[0]!='hist_gradient_boosting' else None; report['runtime_seconds']=time.perf_counter()-t0
 from src.ranking import peak_memory_mib
 report['peak_memory_mib']=peak_memory_mib(); (O/'report.json').write_text(json.dumps(report,indent=2)); (O/'summary.md').write_text('# Phase 4A validation model comparison\n\n'+json.dumps(report,indent=2))
if __name__=='__main__':main()
