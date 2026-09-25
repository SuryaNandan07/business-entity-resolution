"""Phase 3A sample-only feature generation: python -m src.build_features.

No classifier, threshold optimization, test data, or prediction files.
"""

import argparse
from array import array
from collections import Counter
from contextlib import ExitStack
import csv
import gzip
from itertools import islice
import json
import math
from pathlib import Path
import time

import numpy as np

from src.candidates import (Record, CONFIGURATIONS, read_rows, load_truth,
                            load_candidate_pool, build_index, generate_candidates)
from src.candidate_improvements import (Frequencies, AdditionalIndex, prepare_weighted,
                                        improved_signals, improved_score)
from src.features import FEATURE_NAMES, pair_features, label_pair, split_s1_entities
from src.failure_analysis import input_manifest
from src.ranking import peak_memory_mib


class FeatureStatistics:
    """Compact numeric buffers, without IDs, text, or dataframe copies.

    At one million pairs and 25 features these use about 191 MiB of float64
    data. Exact quantiles are computed one column at a time using zero-copy
    NumPy views. The original row order is irrelevant for these summaries.
    """

    def __init__(self):
        self.values = {split: {label: {f: array("d") for f in FEATURE_NAMES}
                               for label in (0, 1)} for split in ("train", "validation")}

    def add(self, split, label, features):
        for feature in FEATURE_NAMES:
            value = features[feature]
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite feature: {feature}")
            self.values[split][label][feature].append(value)

    def summarize(self):
        result = {}
        for split, classes in self.values.items():
            result[split] = {}
            for label, features in classes.items():
                result[split][str(label)] = {}
                for feature, buffer in features.items():
                    values = np.frombuffer(buffer, dtype=np.float64)
                    if not len(values):
                        stats = {"count": 0, "mean": None, "std": None, "median": None}
                    else:
                        mean, std = float(values.mean()), float(values.std())
                        percentiles = np.quantile(values, [.1, .25, .5, .75, .9, .95, .99], overwrite_input=True)
                        stats = {"count": len(values), "mean": mean, "std": std,
                                 "min": float(values.min()), "max": float(values.max()),
                                 **{key: float(value) for key, value in zip(
                                     ("p10", "p25", "median", "p75", "p90", "p95", "p99"), percentiles)}}
                    result[split][str(label)][feature] = stats
        return result


def build_pair_files(queries, pool, truth, assignments, frequencies, output, k=50):
    """Generate, rank, and stream pairs; each S1 is processed exactly once."""
    if set(assignments) != {r.entity_id for r in queries}:
        raise ValueError("Split manifest must cover exactly the query entities")
    if set(assignments.values()) != {"train", "validation"}:
        raise ValueError("Both train and validation splits are required")
    if k < 1:
        raise ValueError("K must be positive")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Posting frequencies describe the search catalog, not validation S1 rows.
    catalog_frequencies = Frequencies.fit(pool)
    index = build_index(pool, CONFIGURATIONS["C"])
    extra = AdditionalIndex(pool, catalog_frequencies)
    prepared = {r.entity_id: prepare_weighted(r, frequencies) for r in pool}
    lookup = {r.entity_id: r for r in pool}
    stats = FeatureStatistics()
    counts = {s: Counter() for s in ("train", "validation")}
    true_totals = {s: sum(len(truth[e]) for e in assignments if assignments[e] == s) for s in counts}
    headers = ["source1_entity_id", "candidate_entity_id", "candidate_source", "label", *FEATURE_NAMES]
    with ExitStack() as stack:
        writers = {}
        for split in counts:
            stream = stack.enter_context(gzip.open(output / f"{split}_features.csv.gz.part", "wt", encoding="utf-8", newline="", compresslevel=1))
            writers[split] = csv.writer(stream)
            writers[split].writerow(headers)
        for number, query in enumerate(queries, 1):
            split = assignments[query.entity_id]
            ids = generate_candidates(query, index, CONFIGURATIONS["C"])
            tokens, grams = extra.candidates(query)
            ids.update(tokens)
            ids.update(grams)
            q = prepare_weighted(query, frequencies)
            scored = [(entity_id, improved_score(improved_signals(q, prepared[entity_id], frequencies))) for entity_id in ids]
            scored.sort(key=lambda item: (-item[1], item[0]))
            counts[split]["s1_entities"] += 1
            counts[split]["zero_candidate_entities"] += not scored
            for rank, (entity_id, score) in enumerate(scored[:k], 1):
                candidate = lookup[entity_id]
                features = pair_features(query, candidate, frequencies, score, rank, q, prepared[entity_id])
                label = label_pair(query.entity_id, entity_id, truth)
                writers[split].writerow([query.entity_id, entity_id, entity_id.split("-", 1)[0], label,
                                        *(features[f] for f in FEATURE_NAMES)])
                stats.add(split, label, features)
                counts[split]["positive_pairs" if label else "negative_pairs"] += 1
            if number % 1000 == 0:
                print(f"Generated features for {number:,}/{len(queries):,} S1 entities", flush=True)
    for split in counts:
        (output / f"{split}_features.csv.gz.part").replace(output / f"{split}_features.csv.gz")
    generation_seconds = time.perf_counter() - started
    stat_start = time.perf_counter()
    summaries = stats.summarize()
    statistics_seconds = time.perf_counter() - stat_start
    balances = {}
    for split, c in counts.items():
        pos, neg = c["positive_pairs"], c["negative_pairs"]
        total = pos + neg
        balances[split] = {**dict(c), "candidate_pairs": total,
                           "positive_percentage": 100 * pos / total if total else 0,
                           "negative_percentage": 100 * neg / total if total else 0,
                           "negative_positive_ratio": neg / pos if pos else None,
                           "known_true_links": true_totals[split],
                           "candidate_recall": pos / true_totals[split] if true_totals[split] else None,
                           "mean_candidates_per_s1": total / c["s1_entities"]}
    return balances, summaries, {"generation_seconds": generation_seconds, "statistics_seconds": statistics_seconds}


def save_frequencies(frequencies, path):
    """Persist frozen feature/scoring IDFs for identical future extraction."""
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as stream:
        stream.write(json.dumps({"documents": dict(frequencies.documents)}) + "\n")
        for field in ("name", "address", "grams"):
            for (country, token), count in sorted(getattr(frequencies, field).items()):
                stream.write(json.dumps([field, country, token, count], ensure_ascii=False) + "\n")


def load_frequencies(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        frequencies = Frequencies(Counter(json.loads(next(stream))["documents"]), Counter(), Counter(), Counter())
        for line in stream:
            field, country, token, count = json.loads(line)
            getattr(frequencies, field)[country, token] = count
    return frequencies


def separation_report(summaries):
    """Descriptive standardized mean gaps; rank using TRAIN statistics only."""
    rows = []
    for feature in FEATURE_NAMES:
        positive = summaries["train"]["1"][feature]
        negative = summaries["train"]["0"][feature]
        if not positive["count"] or not negative["count"]:
            continue
        gap = positive["mean"] - negative["mean"]
        pooled = math.sqrt((positive["std"]**2 + negative["std"]**2) / 2)
        rows.append({"feature": feature, "positive_mean": positive["mean"],
                     "negative_mean": negative["mean"], "standardized_mean_gap": abs(gap) / max(pooled, 1e-12),
                     "positive_direction": "higher" if gap > 0 else "lower" if gap < 0 else "none"})
    return sorted(rows, key=lambda row: (-row["standardized_mean_gap"], row["feature"]))


def write_statistics(summaries, output):
    (output / "feature_statistics.json").write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")
    columns = ["split", "label", "feature", "count", "mean", "std", "min", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"]
    with (output / "feature_statistics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for split, classes in summaries.items():
            for label, features in classes.items():
                for feature, stats in features.items():
                    writer.writerow({"split": split, "label": label, "feature": feature, **stats})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/pair_features"))
    args = parser.parse_args()
    start = time.perf_counter()
    manifest = input_manifest(args.train_dir)
    # Fresh rows avoid recycling the 20,000 entities used to tune Phase 2D.
    previous_ids, queries = set(), []
    for number, row in enumerate(islice(read_rows(args.train_dir / "train_source1.tsv"), 40000)):
        if number < 20000:
            previous_ids.add(row["entity_id"])
        else:
            queries.append(Record.from_row(row))
    if len(queries) != 20000 or previous_ids & {r.entity_id for r in queries}:
        raise ValueError("Expected 20,000 fresh S1 rows disjoint from Phase 2D")
    assignments = split_s1_entities([r.entity_id for r in queries], seed=42)
    truth = load_truth(args.train_dir / "train_ground_truth.tsv", set(assignments))
    train_true = set().union(*(truth[e] for e in assignments if assignments[e] == "train"))
    val_true = set().union(*(truth[e] for e in assignments if assignments[e] == "validation"))
    required = train_true | val_true
    pool, pool_stats = [], {}
    for source in (2, 3):
        selected, pool_stats[f"S{source}"] = load_candidate_pool(
            args.train_dir / f"train_source{source}.tsv", {e for e in required if e.startswith(f"S{source}-")}, 20000, 42 + source)
        pool.extend(selected)
        del selected
        print(f"Loaded S{source}: {pool_stats[f'S{source}']}", flush=True)
    loading_seconds = time.perf_counter() - start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "s1_split.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source1_entity_id", "split"])
        writer.writerows(sorted(assignments.items()))
    # Fit similarity/ranking rarity ONLY on training-linked records + distractors.
    # Validation-only known matches remain searchable, but do not fit these IDFs.
    validation_only_ids = val_true - train_true
    fit_ids = {r.entity_id for r in pool if r.entity_id not in validation_only_ids}
    frequencies = Frequencies.fit(r for r in pool if r.entity_id in fit_ids)
    save_frequencies(frequencies, args.output_dir / "training_frequencies.jsonl.gz")
    with gzip.open(args.output_dir / "frequency_fit_entity_ids.txt.gz", "wt", encoding="utf-8") as stream:
        stream.write("\n".join(sorted(fit_ids)) + "\n")
    balances, summaries, timing = build_pair_files(queries, pool, truth, assignments, frequencies, args.output_dir, k=50)
    write_statistics(summaries, args.output_dir)
    separation = separation_report(summaries)
    constant = [f for f in FEATURE_NAMES if summaries["train"]["0"][f].get("min") == summaries["train"]["0"][f].get("max") == summaries["train"]["1"][f].get("min") == summaries["train"]["1"][f].get("max")]
    if input_manifest(args.train_dir) != manifest:
        raise ValueError("Original training files changed during execution")
    schema = {"metadata_columns": ["source1_entity_id", "candidate_entity_id", "candidate_source"],
              "target_column": "label", "feature_columns": list(FEATURE_NAMES),
              "missing_rule": "Empty text gives zero similarity; separate flags identify missingness",
              "numeric_rule": "Fully decimal tokens; no country-specific rules or number rewriting",
              "name_prefix_rule": "Common leading characters among the first four, divided by four",
              "ngram_rule": "Unique character bigram/trigram sets with whitespace removed; Dice similarity",
              "idf_rule": "Frozen training-side IDFs; same-country weighted Dice as Phase 2D",
              "length_rule": "min/max normalized character count; either empty gives zero",
              "optional_rapidfuzz_available": "name_rapidfuzz_ratio" in FEATURE_NAMES,
              "ranking": "Frozen Phase 2D address-rescue formula; K=50 across both sources; ID tie-break",
              "precision": "Numeric features rounded to 8 decimal places; statistics use float64 exact linear quantiles"}
    (args.output_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    report = {"sample": "Fresh S1 rows 20,001–40,000; zero overlap with Phase 2D S1 entities",
              "seed": 42, "validation_fraction": .2, "k": 50, "split_by": "source1_entity_id",
              "balances": balances, "pool_stats": pool_stats, "feature_count": len(FEATURE_NAMES),
              "feature_separation_training_only": separation, "constant_training_features": constant,
              "training_frequency_fit_records": len(fit_ids), "validation_only_target_records_excluded_from_idf": len(validation_only_ids),
              "true_target_ids_shared_across_s1_splits": len(train_true & val_true),
              "idf_policy": "Similarity/ranking IDFs fit training-linked target records plus sampled distractors, never validation-only targets or validation S1 rows. Blocking posting frequencies describe the full shared target search catalog and are label-free.",
              "label_policy": "Ground truth used only for sampled target inclusion, labels, and evaluation; no forced-positive pairs after pruning",
              "limitations": ["Target catalog is truth-enriched and sampled; these are not full-catalog recall estimates",
                              "The split is S1-disjoint, not target-catalog-disjoint; targets may appear as candidates in both files",
                              "Validation labels are used for requested descriptive statistics, not fitting models or choosing features",
                              "Do not feed metadata IDs, raw source strings, or label into a future feature matrix"],
              "timing": {"loading_seconds": loading_seconds, **timing, "total_seconds": time.perf_counter() - start},
              "peak_memory_mib": peak_memory_mib(), "input_manifest": manifest}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Phase 3A — validation split and pair features", "", report["sample"], "",
             "Sorted S1 IDs shuffled with seed 42; 80% train, 20% validation. K=50; no pair-level splitting.", "",
             "| Split | S1 entities | Pairs | Positive | Negative | Positive % | Negative/positive | Candidate recall |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split, b in balances.items():
        lines.append(f"| {split} | {b['s1_entities']:,} | {b['candidate_pairs']:,} | {b['positive_pairs']:,} | {b['negative_pairs']:,} | {b['positive_percentage']:.3f} | {b['negative_positive_ratio']:.3f} | {b['candidate_recall']:.4%} |")
    lines += ["", "## Feature columns", "", ", ".join(FEATURE_NAMES), "",
              "RapidFuzz is optional; unavailable columns are omitted rather than zero-filled. See schema.json for feature definitions and separate metadata/label columns.", "",
              "## Largest training-only standardized mean gaps", "",
              "This is descriptive feature separation, not model training or feature selection. Magnitude = absolute mean gap / pooled within-class standard deviation.", "",
              "| Feature | Positive mean | Negative mean | Standardized gap | Positive direction |",
              "|---|---:|---:|---:|---|"]
    for row in separation[:10]:
        lines.append(f"| {row['feature']} | {row['positive_mean']:.5f} | {row['negative_mean']:.5f} | {row['standardized_mean_gap']:.3f} | {row['positive_direction']} |")
    lines += ["", "Every feature's positive/negative mean, median, standard deviation, min/max, p10/p25/p75/p90/p95/p99 is in feature_statistics.csv and feature_statistics.json, separately for both splits.", "",
              f"Constant training features: {', '.join(constant)}. Source indicators are complementary; no columns were silently dropped.", "",
              report["idf_policy"], "", report["label_policy"], "",
              *[f"- {item}" for item in report["limitations"]], "",
              f"Timing: {report['timing']}. Peak process memory: {report['peak_memory_mib']:.2f} MiB.", "",
              "Outputs: train_features.csv.gz, validation_features.csv.gz, s1_split.csv, schema.json, frozen training_frequencies.jsonl.gz and its fit-ID manifest, feature_statistics.csv/json, report.json, summary.md.", "",
              "No classifier, thresholds, test data, predictions, submission, or merge. Original TSV metadata unchanged."]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"balances": balances, "timing": report["timing"], "peak_memory_mib": report["peak_memory_mib"], "best_separating_features": separation[:8]}), flush=True)


if __name__ == "__main__":
    main()
