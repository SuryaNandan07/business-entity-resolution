"""Frozen Phase 3C evaluation. Run with ``python -m src.holdout``.

The holdout is read only from training files. No model fitting, threshold
search, or candidate/ranking changes occur here.
"""
import csv, gzip, json, time
from collections import defaultdict
from itertools import islice
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score

from src.candidates import (Record, read_rows, load_truth, load_candidate_pool,
                            build_index, generate_candidates, CONFIGURATIONS)
from src.candidate_improvements import Frequencies, AdditionalIndex, prepare_weighted, improved_signals, improved_score
from src.build_features import load_frequencies
from src.features import FEATURE_NAMES, pair_features, label_pair
from src.evaluate import macro_f05
from src.failure_analysis import input_manifest
from src.ranking import peak_memory_mib

TRAIN = Path("dataset/train")
OUT = Path("output/frozen_holdout")
K, THRESHOLD, SEED = 50, .670, 42


def fresh_queries():
    rows = list(islice(read_rows(TRAIN / "train_source1.tsv"), 60000))
    if len(rows) < 60000:
        raise ValueError("Training Source 1 has fewer than 60,000 rows")
    return [Record.from_row(row) for row in rows[40000:60000]]


def prior_ids():
    ids = set()
    split = Path("output/pair_features/s1_split.csv")
    with split.open(encoding="utf-8") as stream:
        ids.update(row["source1_entity_id"] for row in csv.DictReader(stream))
    # Phase 2D used the first 20,000 Source 1 rows; retain the manifest/cache
    # when present and verify by reading those IDs directly below.
    for row in islice(read_rows(TRAIN / "train_source1.tsv"), 20000):
        ids.add(row["entity_id"])
    return ids


def build_pairs(queries, pool, truth, frequencies):
    catalog_freq = Frequencies.fit(pool)
    index = build_index(pool, CONFIGURATIONS["C"])
    extra = AdditionalIndex(pool, catalog_freq)
    prepared = {r.entity_id: prepare_weighted(r, frequencies) for r in pool}
    lookup = {r.entity_id: r for r in pool}
    rows = []
    for number, query in enumerate(queries, 1):
        ids = generate_candidates(query, index, CONFIGURATIONS["C"])
        tokens, grams = extra.candidates(query)
        ids.update(tokens); ids.update(grams)
        q = prepare_weighted(query, frequencies)
        scored = [(i, improved_score(improved_signals(q, prepared[i], frequencies))) for i in ids]
        scored.sort(key=lambda item: (-item[1], item[0]))
        for rank, (entity_id, score) in enumerate(scored[:K], 1):
            candidate = lookup[entity_id]
            feats = pair_features(query, candidate, frequencies, score, rank, q, prepared[entity_id])
            rows.append((query.entity_id, entity_id, entity_id.split("-", 1)[0],
                         label_pair(query.entity_id, entity_id, truth), feats))
        if number % 1000 == 0:
            print(f"Generated frozen candidates for {number:,}/{len(queries):,} S1 entities", flush=True)
    return rows


def evaluate(rows, queries, truth, model_bundle):
    names = model_bundle["feature_names"]
    if names != [n for n in FEATURE_NAMES if n in names]:
        raise ValueError("Saved model feature schema is invalid")
    x = np.asarray([[features[n] for n in names] for _, _, _, _, features in rows], dtype=np.float32)
    labels = np.asarray([label for _, _, _, label, _ in rows], dtype=np.uint8)
    probabilities = model_bundle["model"].predict_proba(x)[:, 1]
    predictions = {q.entity_id: set() for q in queries}
    candidate_truth = {q.entity_id: set() for q in queries}
    for (s1, candidate, source, label, _), probability in zip(rows, probabilities):
        candidate_truth[s1].add(candidate)
        if probability >= THRESHOLD:
            predictions[s1].add(candidate)
    truth_counts = sum(len(v) for v in truth.values())
    found = int(labels.sum())
    pred_count = sum(len(v) for v in predictions.values())
    tp = sum(len(predictions[s1] & truth[s1]) for s1 in predictions)
    available = {s1: candidate_truth[s1] & truth[s1] for s1 in truth}
    candidate_recall = found / truth_counts if truth_counts else 0
    metrics = {
        "s1_entities": len(queries), "candidate_pairs": len(rows), "candidate_true_links": found,
        "true_links": truth_counts, "candidate_recall": candidate_recall,
        "macro_f05": macro_f05(truth, predictions), "true_positive_links": tp,
        "predicted_links": pred_count, "average_predictions_per_s1": pred_count / len(queries),
        "pair_precision": tp / pred_count if pred_count else 0,
        "pair_recall_available_candidates": tp / found if found else 0,
        "full_ground_truth_link_recall": tp / truth_counts if truth_counts else 0,
        "predicted_singleton_percentage": 100 * sum(len(v) == 1 for v in predictions.values()) / len(queries),
        "correct_singleton_count": sum(len(predictions[s1]) == 1 and predictions[s1] == truth[s1] for s1 in predictions),
        "incorrect_singleton_count": sum(len(predictions[s1]) == 1 and predictions[s1] != truth[s1] for s1 in predictions),
        "zero_prediction_count": sum(not v for v in predictions.values()), "threshold": THRESHOLD, "k": K,
    }
    by_source = {}
    for source in ("S2", "S3"):
        true = {s1: {x for x in ids if x.startswith(source + "-")} for s1, ids in truth.items()}
        pred = {s1: {x for x in ids if x.startswith(source + "-")} for s1, ids in predictions.items()}
        by_source[source] = {"true_links": sum(map(len, true.values())),
            "candidate_true_links": sum(len(available[s1] & true[s1]) for s1 in true),
            "true_positive_links": sum(len(pred[s1] & true[s1]) for s1 in true),
            "macro_f05": macro_f05(true, pred),
            "candidate_recall": sum(len(available[s1] & true[s1]) for s1 in true) / max(1, sum(map(len, true.values()))),
            "full_ground_truth_link_recall": sum(len(pred[s1] & true[s1]) for s1 in true) / max(1, sum(map(len, true.values())))}
    metrics["by_source"] = by_source
    return metrics, predictions, probabilities


def main():
    started = time.perf_counter(); manifest = input_manifest(TRAIN)
    queries = fresh_queries()
    old = prior_ids(); query_ids = {q.entity_id for q in queries}
    if old & query_ids:
        raise ValueError(f"Holdout overlaps previous S1 IDs: {len(old & query_ids)}")
    truth = load_truth(TRAIN / "train_ground_truth.tsv", query_ids)
    required = set().union(*truth.values())
    pool, pool_stats = [], {}
    for source in (2, 3):
        selected, pool_stats[f"S{source}"] = load_candidate_pool(TRAIN / f"train_source{source}.tsv",
            {x for x in required if x.startswith(f"S{source}-")}, 20000, SEED + source)
        pool.extend(selected)
    frequencies = load_frequencies(Path("output/pair_features/training_frequencies.jsonl.gz"))
    rows = build_pairs(queries, pool, truth, frequencies)
    bundle = joblib.load(OUT.parent / "baseline_models" / "hist_gradient_boosting.joblib")
    if bundle["threshold"] != THRESHOLD or bundle["feature_names"] != json.loads(Path("output/pair_features/schema.json").read_text())["feature_columns"][:0] + bundle["feature_names"]:
        raise ValueError("Frozen model metadata does not match required threshold/schema")
    metrics, predictions, probabilities = evaluate(rows, queries, truth, bundle)
    # Compact examples: strongest false positives/true positives and missed links.
    indexed = list(zip(rows, probabilities)); examples = []
    for category, subset, reverse in [("false_positive", [(r,p) for r,p in indexed if r[3] == 0 and p >= THRESHOLD], True),
                                       ("strong_true", [(r,p) for r,p in indexed if r[3] == 1], True),
                                       ("difficult_true", [(r,p) for r,p in indexed if r[3] == 1], False)]:
        for row, p in sorted(subset, key=lambda z: z[1], reverse=reverse)[:5]:
            examples.append({"category": category, "source1_entity_id": row[0], "candidate_entity_id": row[1], "probability": float(p), "label": row[3]})
    for s1, ids in truth.items():
        for candidate in sorted(ids - {r[1] for r,p in indexed if r[0] == s1}):
            examples.append({"category": "missed_candidate", "source1_entity_id": s1, "candidate_entity_id": candidate, "probability": None, "label": 1})
            if sum(e["category"] == "missed_candidate" for e in examples) >= 5: break
        if sum(e["category"] == "missed_candidate" for e in examples) >= 5: break
    for category, predicate in [("correct_singleton", lambda s1: len(predictions[s1]) == 1 and predictions[s1] == truth[s1]),
                                ("incorrect_singleton", lambda s1: len(predictions[s1]) == 1 and predictions[s1] != truth[s1])]:
        for s1 in sorted(truth):
            if predicate(s1):
                candidate = next(iter(predictions[s1]))
                probability = next((float(p) for (r, p) in indexed if r[0] == s1 and r[1] == candidate), None)
                examples.append({"category": category, "source1_entity_id": s1, "candidate_entity_id": candidate,
                                 "probability": probability, "label": 1 if candidate in truth[s1] else 0})
                if sum(e["category"] == category for e in examples) >= 5: break
    # Country breakdown uses the query's original country and the full truth set.
    countries = defaultdict(list)
    for q in queries: countries[q.country].append(q.entity_id)
    country_metrics = {}
    for country, ids in sorted(countries.items()):
        t, p = {x: truth[x] for x in ids}, {x: predictions[x] for x in ids}
        country_metrics[country] = {"s1_entities": len(ids), "macro_f05": macro_f05(t, p),
            "true_links": sum(map(len, t.values())), "predicted_links": sum(map(len, p.values())),
            "full_ground_truth_link_recall": sum(len(p[x] & t[x]) for x in ids) / max(1, sum(map(len, t.values())))}
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"fresh_holdout_rows": "Source 1 rows 40,001–60,000 (one-based)", "s1_count": len(queries),
      "previous_manifest_ids": len(old), "overlap_count": len(old & query_ids), "seed": SEED,
      "pool_stats": pool_stats, "metrics": metrics, "country_breakdown": country_metrics,
      "examples": examples, "runtime_seconds": time.perf_counter() - started, "peak_memory_mib": peak_memory_mib(),
      "threshold_frozen": THRESHOLD, "k_frozen": K, "feature_names": bundle["feature_names"],
      "original_tsv_metadata_unchanged": input_manifest(TRAIN) == manifest,
      "comparison": {"validation_macro_f05": .9820252551563291, "holdout_macro_f05": metrics["macro_f05"],
                      "difference": metrics["macro_f05"] - .9820252551563291}}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "summary.md").write_text("# Phase 3C frozen holdout\n\n" + json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "runtime_seconds": report["runtime_seconds"], "peak_memory_mib": report["peak_memory_mib"]}), flush=True)


if __name__ == "__main__": main()
