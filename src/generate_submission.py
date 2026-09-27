"""ML Challenge 2026 — Submission Generator & Official Validator Runner.

Executes exact test candidate generation (K=50), feature extraction,
LightGBM model scoring (threshold 0.585), TSV formatting, and official validation.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'

from array import array
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import joblib
import numpy as np

from src.catalog_cache import Catalog, build_catalog, RULES
from src.resource_guard import check, memory, reset_guard_state
from src.candidates import Record, read_rows, blocking_keys
from src.candidate_improvements import prepare_weighted
from src.build_features import load_frequencies
from src.features import pair_features, FEATURE_NAMES
from src.compact_ranking import RankingIndex, VectorRanker

DATASET_TEST = Path('dataset/test')
TEST_PATHS = [DATASET_TEST / f'test_source{s}.tsv' for s in (2, 3)]
CACHE = Path('output/cache/test_catalog')
OUT = Path('output')

def main():
    start_time = time.perf_counter()
    reset_guard_state()
    OUT.mkdir(parents=True, exist_ok=True)
    
    print("=== STEP 1: LOAD TEST QUERIES & FREQUENCIES ===", flush=True)
    test_qs = [Record.from_row(r) for r in read_rows(DATASET_TEST / 'test_source1.tsv')]
    print(f"Loaded {len(test_qs):,} test S1 queries from {DATASET_TEST / 'test_source1.tsv'}", flush=True)
    
    freq = load_frequencies(Path("output/pair_features/training_frequencies.jsonl.gz"))
    print("Loaded training frequency tables.", flush=True)

    print("\n=== STEP 2: BUILD / LOAD TEST TARGET CATALOG ===", flush=True)
    meta = build_catalog(TEST_PATHS, CACHE, guard=check)
    catalog = Catalog(CACHE, TEST_PATHS)
    print(f"Test catalog loaded with {catalog.meta['records']:,} target records.", flush=True)

    print("\n=== STEP 3: CANDIDATE GENERATION & RANKING (K=50) ===", flush=True)
    idx, index_info = catalog.index(test_qs, guard=check)
    print(f"Candidate index ready (cache_hit={index_info.get('cache_hit')})", flush=True)
    
    feature_prov = {'provenance': 'test_inference'}
    ranking_index, _ = RankingIndex.cached(catalog, test_qs, freq, feature_prov, check)
    ranker = VectorRanker(ranking_index, freq)
    
    # Load trained LightGBM model
    model_path = Path('output/phase4a/lightgbm.joblib')
    if not model_path.exists():
        model_path = Path('output/baseline_models/hist_gradient_boosting.joblib')
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    model_feature_names = model_data['feature_names']
    threshold = model_data.get('threshold', 0.585)
    print(f"Loaded model from {model_path} (Threshold: {threshold}, Features: {len(model_feature_names)})", flush=True)
    
    # Feature column indices
    feat_indices = [FEATURE_NAMES.index(fn) for fn in model_feature_names]

    matching_path = OUT / 'matching_results.tsv'
    candidate_path = OUT / 'candidate_pairs.tsv'
    
    matching_lines = ["source1_entity_id\tmatched_entity_ids\n"]
    candidate_lines = ["source1_entity_id\tcandidate_entity_ids\n"]
    
    total_matches = 0
    total_candidates = 0
    
    print("\n=== STEP 4: FEATURE GENERATION & MODEL SCORING ===", flush=True)
    for q_idx, q in enumerate(test_qs, 1):
        check()
        cands = idx.candidates(q)
        if not cands:
            candidate_lines.append(f"{q.entity_id}\t\n")
            matching_lines.append(f"{q.entity_id}\t\n")
            continue
            
        ranked, _ = ranker.rank(q, cands, catalog.get, k=50)
        
        # Prepare candidates and features
        cand_records = [catalog.get(cid) for cid, _, _ in ranked]
        cand_eids = [r.entity_id for r in cand_records]
        total_candidates += len(cand_eids)
        
        candidate_lines.append(f"{q.entity_id}\t{','.join(cand_eids)}\n")
        
        # Build features for candidates
        qw = prepare_weighted(q, freq)
        X_cand = []
        for rank_pos, (r, (cid, eid, score)) in enumerate(zip(cand_records, ranked), 1):
            rw = prepare_weighted(r, freq)
            feat_dict = pair_features(q, r, freq, ranking_score=score, candidate_rank=rank_pos, prepared_left=qw, prepared_right=rw)
            feat_vec = [feat_dict[fn] for fn in FEATURE_NAMES]
            X_cand.append([feat_vec[i] for i in feat_indices])
            
        X_cand = np.array(X_cand, dtype=np.float32)
        probs = model.predict_proba(X_cand)[:, 1]
        
        matched_eids = [eid for eid, prob in zip(cand_eids, probs) if prob >= threshold]
        total_matches += len(matched_eids)
        matching_lines.append(f"{q.entity_id}\t{','.join(matched_eids)}\n")
        
        if q_idx % 10000 == 0 or q_idx == len(test_qs):
            print(f"Processed {q_idx:,}/{len(test_qs):,} queries... matches={total_matches:,}", flush=True)
            
    print("\n=== STEP 5: WRITE OUTPUT FILES ===", flush=True)
    matching_path.write_text("".join(matching_lines), encoding='utf-8')
    candidate_path.write_text("".join(candidate_lines), encoding='utf-8')
    print(f"Wrote {matching_path} ({matching_path.stat().st_size:,} bytes)", flush=True)
    print(f"Wrote {candidate_path} ({candidate_path.stat().st_size:,} bytes)", flush=True)
    
    print("\n=== STEP 6: RUN OFFICIAL SUBMISSION VALIDATOR ===", flush=True)
    cmd = [
        sys.executable, "utils/validate_submission.py",
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(DATASET_TEST),
        "--check-ids"
    ]
    print("Running:", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
        
    if res.returncode == 0:
        print("\n✅ SUBMISSION VALIDATION PASSED PERFECTLY!", flush=True)
    else:
        print("\n❌ SUBMISSION VALIDATION FAILED!", flush=True)
        sys.exit(1)
        
    print(f"Total end-to-end execution time: {time.perf_counter() - start_time:.2f}s", flush=True)

if __name__ == '__main__':
    main()
