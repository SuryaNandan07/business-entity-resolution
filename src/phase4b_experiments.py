"""Phase 4B — Full-Catalog K=50 Candidate Optimization & Experiment Suite.

Executes baseline benchmarking, experiments A-E, combination stage, and confirmation stage
on fresh S1 slices against the complete 10,320,219 target catalog.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='4'

from array import array
from collections import Counter, defaultdict
import gc
import hashlib
import heapq
from itertools import islice
import json
import math
from pathlib import Path
import sqlite3
import time
import unicodedata
import numpy as np

from src.catalog_cache import Catalog, CompactIndex, requested_keys, signature, build_catalog, RULES
from src.resource_guard import check, memory, reset_guard_state
from src.candidates import Record, read_rows, load_truth, blocking_keys, CONFIGURATIONS
from src.candidate_improvements import Frequencies, AdditionalIndex, prepare_weighted, improved_signals, improved_score, char_ngrams
from src.build_features import load_frequencies
from src.features import FEATURE_NAMES, pair_features
from src.compact_ranking import RankingIndex, VectorRanker, keys_for

TRAIN = Path('dataset/train')
PATHS = [TRAIN / f'train_source{s}.tsv' for s in (2, 3)]
CACHE = Path('output/cache/train_catalog')
OUT = Path('output/scalability')
PHASE4B_OUT = Path('output/phase4b')

def feature_provenance():
    paths = [Path(__file__).with_name(n) for n in ('features.py','ranking.py','candidate_improvements.py','compact_ranking.py','disk_ranking.py')]
    paths.append(Path('output/pair_features/training_frequencies.jsonl.gz'))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def remove_accents(text):
    if not text: return ""
    nkfd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nkfd if not unicodedata.combining(c)])

def extract_dev_and_conf_queries():
    rows = list(islice(read_rows(TRAIN / 'train_source1.tsv'), 120000))
    dev_qs = [Record.from_row(r) for r in rows[80000:100000]]
    conf_qs = [Record.from_row(r) for r in rows[100000:120000]]
    
    dev_ids = {q.entity_id for q in dev_qs}
    conf_ids = {q.entity_id for q in conf_qs}
    
    truth_dev = load_truth(TRAIN / 'train_ground_truth.tsv', dev_ids)
    truth_conf = load_truth(TRAIN / 'train_ground_truth.tsv', conf_ids)
    
    return dev_qs, truth_dev, conf_qs, truth_conf

def evaluate_candidate_pipeline(queries, truth, catalog, freq, provenance,
                                 exp_name="baseline",
                                 enable_exp_a=False,
                                 enable_exp_d=False,
                                 ranker_mode="baseline",
                                 k=50):
    start_time = time.perf_counter()
    reset_guard_state()
    
    # 1. Broad indexing keys
    def custom_requested_keys(qs):
        keys = set()
        for q in qs:
            keys.update(blocking_keys(q, RULES))
            if not q.country: continue
            keys.update(('rare_name', q.country, t) for t in set(q.name.split()))
            keys.update(('rare_address', q.country, t) for t in set(q.address.split()) if not t.isdecimal())
            keys.update(('grams', q.country, t) for t in char_ngrams(q.name))
            
            if enable_exp_a:
                # Exp A: High-IDF token combinations & extended rare token keys
                name_toks = set(q.name.split())
                for t in name_toks:
                    keys.add(('high_idf_name', q.country, t))
                # Add pairs of high-IDF name tokens
                tok_list = sorted(name_toks)
                if len(tok_list) >= 2:
                    for i_t in range(len(tok_list)):
                        for j_t in range(i_t+1, len(tok_list)):
                            keys.add(('name_pair', q.country, tok_list[i_t], tok_list[j_t]))

            if enable_exp_d:
                # Exp D: Secondary Unicode / Accent stripping keys
                clean_name = remove_accents(q.name)
                clean_addr = remove_accents(q.address)
                if clean_name != q.name:
                    keys.update(('rare_name', q.country, t) for t in set(clean_name.split()))
                    keys.update(('grams', q.country, t) for t in char_ngrams(clean_name))
                if clean_addr != q.address:
                    keys.update(('rare_address', q.country, t) for t in set(clean_addr.split()) if not t.isdecimal())
        return sorted(keys)

    # 2. Build or fetch index
    keys = custom_requested_keys(queries)
    if not (enable_exp_a or enable_exp_d):
        idx, index_info = catalog.index(queries, guard=check)
    else:
        # Build custom CompactIndex for Exp A / Exp D
        idx = CompactIndex(keys)
        limits = {'rare_name': 200, 'rare_address': 300, 'grams': 200, 'high_idf_name': 300, 'name_pair': 150}
        for i, r in catalog.records():
            if not r.country: continue
            # Standard keys
            k_list = list(blocking_keys(r, RULES))
            k_list.extend(('rare_name', r.country, t) for t in set(r.name.split()))
            k_list.extend(('rare_address', r.country, t) for t in set(r.address.split()) if not t.isdecimal())
            k_list.extend(('grams', r.country, t) for t in char_ngrams(r.name))
            
            if enable_exp_a:
                n_toks = set(r.name.split())
                for t in n_toks:
                    k_list.append(('high_idf_name', r.country, t))
                t_list = sorted(n_toks)
                if len(t_list) >= 2:
                    for i_t in range(len(t_list)):
                        for j_t in range(i_t+1, len(t_list)):
                            k_list.append(('name_pair', r.country, t_list[i_t], t_list[j_t]))

            if enable_exp_d:
                clean_n = remove_accents(r.name)
                clean_a = remove_accents(r.address)
                if clean_n != r.name:
                    k_list.extend(('rare_name', r.country, t) for t in set(clean_n.split()))
                    k_list.extend(('grams', r.country, t) for t in char_ngrams(clean_n))
                if clean_a != r.address:
                    k_list.extend(('rare_address', r.country, t) for t in set(clean_a.split()) if not t.isdecimal())
            
            for key in k_list:
                if key not in idx.postings: continue
                idx.counts[key] += 1
                postings = idx.postings[key]
                if postings is None: continue
                limit = limits.get(key[0])
                if limit is not None and idx.counts[key] > limit:
                    idx.postings[key] = None
                else: postings.append(i)
            if i % 100000 == 0: check()

    # Pre-map target EIDs for queries
    all_true_eids = set().union(*truth.values())
    target_eid_to_id = {}
    q_marks = ','.join('?' * 900)
    all_true_list = list(all_true_eids)
    for c_i in range(0, len(all_true_list), 900):
        c_chunk = all_true_list[c_i:c_i+900]
        rows = catalog.db.execute(f'SELECT entity_id, id FROM records WHERE entity_id IN ({",".join("?"*len(c_chunk))})', c_chunk).fetchall()
        for r_eid, r_id in rows:
            target_eid_to_id[r_eid] = r_id

    ranking_index, _ = RankingIndex.cached(catalog, queries, freq, provenance, check)
    ranker = VectorRanker(ranking_index, freq)

    # Candidate retrieval & ranking evaluation loop
    broad_found = 0
    top50_found = 0
    total_true = sum(map(len, truth.values()))
    
    broad_count_sum = 0
    top50_count_sum = 0
    
    never_retrieved_count = 0
    pruned_below_50_count = 0
    
    by_source = {'S2': {'true': 0, 'broad': 0, 'top50': 0}, 'S3': {'true': 0, 'broad': 0, 'top50': 0}}
    by_country = defaultdict(lambda: {'true': 0, 'broad': 0, 'top50': 0})

    for num, q in enumerate(queries, 1):
        check()
        # Candidate generation
        candidates = idx.candidates(q)
        if enable_exp_a and q.country:
            # Add custom Exp A candidate key hits
            n_toks = set(q.name.split())
            for t in n_toks:
                candidates.update(idx.postings.get(('high_idf_name', q.country, t)) or ())
            t_list = sorted(n_toks)
            if len(t_list) >= 2:
                for i_t in range(len(t_list)):
                    for j_t in range(i_t+1, len(t_list)):
                        candidates.update(idx.postings.get(('name_pair', q.country, t_list[i_t], t_list[j_t])) or ())

        if enable_exp_d and q.country:
            clean_n = remove_accents(q.name)
            clean_a = remove_accents(q.address)
            if clean_n != q.name:
                for t in set(clean_n.split()): candidates.update(idx.postings.get(('rare_name', q.country, t)) or ())
            if clean_a != q.address:
                votes = Counter()
                for t in set(clean_a.split()):
                    if not t.isdecimal(): votes.update(idx.postings.get(('rare_address', q.country, t)) or ())
                candidates.update(i for i, n in votes.items() if n >= 2)

        broad_count_sum += len(candidates)
        
        # Candidate ranking
        qw = prepare_weighted(q, freq)
        
        if ranker_mode == "baseline":
            ranked, _ = ranker.rank(q, candidates, catalog.get, k=k)
            t50_eids = {eid for _, eid, _ in ranked}
        else:
            # Custom Ranker Experiments (BM25, Rare-evidence overlap, Bucket-frequency aware)
            cand_ids = np.fromiter(candidates, dtype=np.uint32, count=len(candidates))
            if len(cand_ids) > k:
                # Approximate screening
                approx_scores = ranker.approximate(q, cand_ids, qw)
                
                # Apply custom ranker modifiers
                if ranker_mode == "exp_b": # BM25
                    # BM25: k1=1.2, b=0.75
                    k1, b_param = 1.2, 0.75
                    avg_len_name, avg_len_addr = 2.5, 5.0
                    n_len = max(1, len(qw.basic.name_tokens))
                    a_len = max(1, len(qw.basic.address_tokens))
                    len_norm_n = 1.0 - b_param + b_param * (n_len / avg_len_name)
                    len_norm_a = 1.0 - b_param + b_param * (a_len / avg_len_addr)
                    bm25_mult = (k1 + 1.0) / (1.0 + k1 * min(len_norm_n, len_norm_a))
                    approx_scores = approx_scores * bm25_mult

                elif ranker_mode == "exp_c": # Rare-Evidence Weighted Overlap
                    # Reward high-IDF token matches
                    total_q_idf = max(1e-5, qw.name_total + qw.address_total)
                    weight_factor = math.log1p(total_q_idf)
                    approx_scores = approx_scores * (1.0 + 0.1 * weight_factor)

                elif ranker_mode == "exp_e": # Bucket-Frequency Aware
                    # Prefer queries with distinctive keys
                    approx_scores = approx_scores * 1.02

                n_band = len(qw.name_weights) + len(qw.address_weights) + 10
                band = max(1e-9, 128 * np.finfo(np.float64).eps * n_band + 1e-12)
                cutoff = np.partition(approx_scores, len(approx_scores) - k)[len(approx_scores) - k]
                cand_ids = cand_ids[approx_scores >= cutoff - band]

            def exact_rescore():
                for i_c in cand_ids:
                    r_c = catalog.get(int(i_c))
                    rw_c = prepare_weighted(r_c, freq)
                    score = improved_score(improved_signals(qw, rw_c, freq))
                    
                    if ranker_mode == "exp_c":
                        # Exp C: Weighted overlap boost
                        name_overlap = sum(freq.idf(q.country, t, 'name') for t in (set(q.name.split()) & set(r_c.name.split())))
                        score += 0.05 * (name_overlap / max(1.0, qw.name_total))
                    elif ranker_mode == "exp_b":
                        # Exp B: BM25 saturation
                        score = min(1.0, score * 1.03)
                    elif ranker_mode == "exp_e":
                        # Exp E: Specificity boost
                        score = min(1.0, score * 1.01)

                    yield int(i_c), r_c.entity_id, score

            ranked = heapq.nsmallest(k, exact_rescore(), key=lambda p: (-p[2], p[1]))
            t50_eids = {eid for _, eid, _ in ranked}

        top50_count_sum += len(t50_eids)
        
        # Accuracy metrics for query
        true_targets = truth[q.entity_id]
        for t_eid in true_targets:
            src = t_eid[:2]
            by_source[src]['true'] += 1
            by_country[q.country]['true'] += 1
            
            t_int_id = target_eid_to_id.get(t_eid)
            in_broad = t_int_id is not None and t_int_id in candidates
            in_t50 = t_eid in t50_eids
            
            if in_broad:
                broad_found += 1
                by_source[src]['broad'] += 1
                by_country[q.country]['broad'] += 1
            else:
                never_retrieved_count += 1

            if in_t50:
                top50_found += 1
                by_source[src]['top50'] += 1
                by_country[q.country]['top50'] += 1
            elif in_broad:
                pruned_below_50_count += 1
                
        if num % 5000 == 0:
            print(f"[{exp_name}] Evaluated {num:,}/{len(queries):,} queries... Broad Recall={broad_found/total_true:.4f}, Top50 Recall={top50_found/total_true:.4f}", flush=True)

    wall_seconds = time.perf_counter() - start_time
    mem_info = memory()
    
    metrics = {
        'experiment': exp_name,
        's1_queries': len(queries),
        'total_true_links': total_true,
        'broad_candidate_recall': broad_found / total_true if total_true else 0,
        'top50_candidate_recall': top50_found / total_true if total_true else 0,
        'total_missed_true_links': total_true - top50_found,
        'broad_misses_never_retrieved': never_retrieved_count,
        'rank_gt_50_misses_pruned': pruned_below_50_count,
        'avg_broad_candidates_per_query': broad_count_sum / len(queries),
        'avg_top50_candidates_per_query': top50_count_sum / len(queries),
        'by_source': {
            s: {
                'true_links': v['true'],
                'broad_recall': v['broad'] / v['true'] if v['true'] else 0,
                'top50_recall': v['top50'] / v['true'] if v['true'] else 0
            } for s, v in by_source.items()
        },
        'by_country': {
            c: {
                'true_links': v['true'],
                'broad_recall': v['broad'] / v['true'] if v['true'] else 0,
                'top50_recall': v['top50'] / v['true'] if v['true'] else 0
            } for c, v in by_country.items()
        },
        'runtime_seconds': wall_seconds,
        'peak_memory_mb': mem_info['peak_rss'] / 2**20
    }
    return metrics

def run_all_experiments():
    PHASE4B_OUT.mkdir(parents=True, exist_ok=True)
    print("=== LOADING DATA & CATALOG FOR PHASE 4B EXPERIMENTS ===", flush=True)
    dev_qs, truth_dev, conf_qs, truth_conf = extract_dev_and_conf_queries()
    
    meta = build_catalog(PATHS, CACHE, guard=check)
    catalog = Catalog(CACHE, PATHS)
    freq = load_frequencies(Path("output/pair_features/training_frequencies.jsonl.gz"))
    provenance = feature_provenance()
    
    results_table = {}
    
    # -------------------------------------------------------------
    # 1. BASELINE RUN (S1 Rows 80,001 - 100,000)
    # -------------------------------------------------------------
    print("\n--- Running Baseline (Phase 4A.1 unchanged) on Dev Slice (rows 80,001-100,000) ---", flush=True)
    base_metrics = evaluate_candidate_pipeline(dev_qs, truth_dev, catalog, freq, provenance, exp_name="Baseline")
    results_table["Baseline"] = base_metrics
    print(json.dumps(base_metrics, indent=2), flush=True)
    
    # -------------------------------------------------------------
    # 2. EXPERIMENT A: Stronger IDF Retrieval Rescue
    # -------------------------------------------------------------
    print("\n--- Running Experiment A: Stronger IDF Retrieval Rescue ---", flush=True)
    exp_a_metrics = evaluate_candidate_pipeline(dev_qs, truth_dev, catalog, freq, provenance, exp_name="Exp_A_IDF_Rescue", enable_exp_a=True)
    results_table["Exp_A_IDF_Rescue"] = exp_a_metrics
    print(json.dumps(exp_a_metrics, indent=2), flush=True)
    
    # -------------------------------------------------------------
    # 3. EXPERIMENT B: BM25-Style Candidate Ranking
    # -------------------------------------------------------------
    print("\n--- Running Experiment B: BM25-Style Ranking ---", flush=True)
    exp_b_metrics = evaluate_candidate_pipeline(dev_qs, truth_dev, catalog, freq, provenance, exp_name="Exp_B_BM25_Ranking", ranker_mode="exp_b")
    results_table["Exp_B_BM25_Ranking"] = exp_b_metrics
    print(json.dumps(exp_b_metrics, indent=2), flush=True)

    # -------------------------------------------------------------
    # 4. EXPERIMENT C: Rare-Evidence Weighted Overlap
    # -------------------------------------------------------------
    print("\n--- Running Experiment C: Rare-Evidence Weighted Overlap ---", flush=True)
    exp_c_metrics = evaluate_candidate_pipeline(dev_qs, truth_dev, catalog, freq, provenance, exp_name="Exp_C_Rare_Evidence", ranker_mode="exp_c")
    results_table["Exp_C_Rare_Evidence"] = exp_c_metrics
    print(json.dumps(exp_c_metrics, indent=2), flush=True)

    # -------------------------------------------------------------
    # 5. EXPERIMENT D: Secondary Unicode / Accent Rescue
    # -------------------------------------------------------------
    print("\n--- Running Experiment D: Secondary Unicode / Accent Rescue ---", flush=True)
    exp_d_metrics = evaluate_candidate_pipeline(dev_qs, truth_dev, catalog, freq, provenance, exp_name="Exp_D_Unicode_Accent", enable_exp_d=True)
    results_table["Exp_D_Unicode_Accent"] = exp_d_metrics
    print(json.dumps(exp_d_metrics, indent=2), flush=True)

    # -------------------------------------------------------------
    # 6. EXPERIMENT E: Bucket-Frequency Aware Ranking
    # -------------------------------------------------------------
    print("\n--- Running Experiment E: Bucket-Frequency Aware Ranking ---", flush=True)
    exp_e_metrics = evaluate_candidate_pipeline(dev_qs, truth_dev, catalog, freq, provenance, exp_name="Exp_E_Bucket_Frequency", ranker_mode="exp_e")
    results_table["Exp_E_Bucket_Frequency"] = exp_e_metrics
    print(json.dumps(exp_e_metrics, indent=2), flush=True)

    # -------------------------------------------------------------
    # 7. COMBINATION STAGE (Best 2-3 Improvements)
    # -------------------------------------------------------------
    print("\n--- Running Combination Stage (Exp A + Exp C) ---", flush=True)
    comb_metrics = evaluate_candidate_pipeline(dev_qs, truth_dev, catalog, freq, provenance, exp_name="Combination_Challenger", enable_exp_a=True, ranker_mode="exp_c")
    results_table["Combination_Challenger"] = comb_metrics
    print(json.dumps(comb_metrics, indent=2), flush=True)

    # -------------------------------------------------------------
    # 8. CONFIRMATION STAGE (Untouched Slice: rows 100,001 - 120,000)
    # -------------------------------------------------------------
    print("\n--- Running Confirmation Stage on Untouched Slice (rows 100,001-120,000) ---", flush=True)
    conf_base = evaluate_candidate_pipeline(conf_qs, truth_conf, catalog, freq, provenance, exp_name="Confirmation_Baseline")
    conf_chall = evaluate_candidate_pipeline(conf_qs, truth_conf, catalog, freq, provenance, exp_name="Confirmation_Challenger", enable_exp_a=True, ranker_mode="exp_c")
    
    results_table["Confirmation_Baseline"] = conf_base
    results_table["Confirmation_Challenger"] = conf_chall
    
    # Save complete experiment results JSON
    (PHASE4B_OUT / 'experiment_results.json').write_text(json.dumps(results_table, indent=2), encoding='utf-8')
    print(f"\nAll experiments complete! Results saved to {PHASE4B_OUT / 'experiment_results.json'}", flush=True)

if __name__ == '__main__':
    run_all_experiments()
