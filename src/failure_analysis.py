"""Phase 2D analysis and sample-only experiments.

Run python -m src.failure_analysis --stage analyze, then --stage experiment.
The small gzip sample cache avoids rescanning training files for each stage.
Original TSVs and previous experiment outputs are only read.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time

from src.candidates import (Record, CONFIGURATIONS, build_index, generate_candidates,
                            load_s1_sample, load_truth, load_candidate_pool)
from src.ranking import (prepare, pair_signals, ranking_score, count_summary,
                         peak_memory_mib, rank_summary)
from src.candidate_improvements import (Frequencies, AdditionalIndex, prepare_weighted,
                                        improved_signals, improved_score,
                                        failure_categories, script_profile)


KS = (50, 100, 150, 200)


def input_manifest(train):
    return {name: {"path": str((train / name).resolve()), "size": (train / name).stat().st_size,
                   "mtime_ns": (train / name).stat().st_mtime_ns}
            for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv")}


def load_sample(train, output, old_report):
    """Keep only the exact deterministic Phase 2B development sample."""
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "sample.jsonl.gz"
    manifest_path = output / "sample_manifest.json"
    manifest = input_manifest(train)
    expected = json.loads(old_report.read_text(encoding="utf-8"))
    if expected["sample_size"] != 20000 or expected["seed"] != 42:
        raise ValueError("Expected original deterministic 20,000-S1 sample")
    if cache.exists() and manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["inputs"] != manifest or saved["pool_stats"] != expected["pool_stats"]:
            raise ValueError("Sample cache differs from inputs; use a fresh output directory")
        queries, pool, truth = [], [], {}
        with gzip.open(cache, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["kind"] == "truth":
                    truth[row["entity_id"]] = set(row["matches"])
                else:
                    (queries if row["kind"] == "query" else pool).append(Record(**row["record"]))
        if len(queries) != 20000 or len(pool) != sum(s["retained_rows"] for s in saved["pool_stats"].values()):
            raise ValueError("Incomplete sample cache")
        return queries, pool, truth, saved["pool_stats"]
    queries = load_s1_sample(train / "train_source1.tsv", 20000)
    if len(queries) != 20000:
        raise ValueError("Expected 20,000 S1 records")
    truth = load_truth(train / "train_ground_truth.tsv", {r.entity_id for r in queries})
    required = set().union(*truth.values())
    pool, stats = [], {}
    for source in (2, 3):
        selected, stats[f"S{source}"] = load_candidate_pool(
            train / f"train_source{source}.tsv", {e for e in required if e.startswith(f"S{source}-")},
            20000, 42 + source)
        pool.extend(selected)
        del selected
        print(f"Loaded S{source}: {stats[f'S{source}']}", flush=True)
    if stats != expected["pool_stats"]:
        raise ValueError("Candidate pools differ from Phase 2B")
    with gzip.open(cache, "wt", encoding="utf-8", compresslevel=1) as stream:
        for kind, records in (("query", queries), ("pool", pool)):
            for r in records:
                stream.write(json.dumps({"kind": kind, "record": asdict(r)}, ensure_ascii=False) + "\n")
        for entity_id, matches in truth.items():
            stream.write(json.dumps({"kind": "truth", "entity_id": entity_id, "matches": sorted(matches)}) + "\n")
    manifest_path.write_text(json.dumps({"inputs": manifest, "pool_stats": stats}, indent=2), encoding="utf-8")
    return queries, pool, truth, stats


def analyze(queries, pool, truth, frequencies, output):
    start = time.perf_counter()
    index = build_index(pool, CONFIGURATIONS["C"])
    lookup = {r.entity_id: r for r in pool}
    prepared = {r.entity_id: prepare(r) for r in pool}
    counts = {"blocking": Counter(), "ranking": Counter()}
    totals = Counter()
    rivals = Counter()
    examples = {"blocking": {}, "ranking": {}}
    broad_pair_count = 0
    with gzip.open(output / "failures.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as stream:
        for number, query in enumerate(queries, 1):
            ids = generate_candidates(query, index, CONFIGURATIONS["C"])
            broad_pair_count += len(ids)
            q = prepare(query)
            ranked = sorted(((entity_id, ranking_score(pair_signals(q, prepared[entity_id]), "address_supported"))
                             for entity_id in ids), key=lambda item: (-item[1], item[0]))
            positions = {e: (rank, score) for rank, (e, score) in enumerate(ranked, 1)}
            true_ids = truth[query.entity_id]
            for entity_id in sorted(true_ids):
                kind = "blocking" if entity_id not in ids else "ranking" if positions[entity_id][0] > 150 else None
                if kind is None:
                    continue
                candidate = lookup[entity_id]
                flags = failure_categories(query, candidate, frequencies)
                counts[kind].update(flags)
                totals[kind] += 1
                row = {"kind": kind, "s1": asdict(query), "true_candidate": asdict(candidate),
                       "s1_script": script_profile(query.name), "candidate_script": script_profile(candidate.name),
                       "categories": flags, "rank": positions.get(entity_id, (None, None))[0],
                       "score": positions.get(entity_id, (None, None))[1],
                       "true_signals": pair_signals(q, prepared[entity_id])._asdict()}
                if kind == "ranking":
                    rival_id, rival_score = next((item for item in ranked[:150] if item[0] not in true_ids), (None, None))
                    if rival_id:
                        rival = prepared[rival_id]
                        signals = pair_signals(q, rival)
                        shared_names = q.name_tokens & rival.name_tokens
                        shared_numbers = q.numbers & rival.numbers
                        checks = {
                            "highest_false_rival_shares_only_common_name_tokens": bool(shared_names) and all(frequencies.common(query.country, t) for t in shared_names),
                            "highest_false_rival_shares_a_common_number": any(frequencies.common(query.country, t, "address") for t in shared_numbers),
                            "highest_false_rival_has_less_address_overlap_than_true": signals.address_jaccard < row["true_signals"]["address_jaccard"],
                            "highest_false_rival_has_more_name_bigram_overlap": signals.name_bigram_dice > row["true_signals"]["name_bigram_dice"],
                        }
                        rivals.update(key for key, value in checks.items() if value)
                        row["highest_false_rival"] = {"record": asdict(lookup[rival_id]), "score": rival_score,
                                                       "signals": signals._asdict(), "diagnostics": checks,
                                                       "shared_name_tokens": sorted(shared_names),
                                                       "shared_numbers": sorted(shared_numbers)}
                    cutoff_id, cutoff_score = ranked[149]
                    row["cutoff"] = {"entity_id": cutoff_id, "score": cutoff_score,
                                     "is_true": cutoff_id in true_ids}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                for flag in flags:
                    examples[kind].setdefault(flag, [])
                    if len(examples[kind][flag]) < 2:
                        examples[kind][flag].append(row)
            if number % 1000 == 0:
                print(f"Analyzed {number:,}/20,000 S1 records", flush=True)
    if totals != Counter(blocking=3114, ranking=739) or broad_pair_count != 14172906:
        raise ValueError(f"Previous baseline was not reproduced: {totals}, {broad_pair_count}")
    report = {"totals": dict(totals), "categories": {k: dict(v) for k, v in counts.items()},
              "ranking_rival_diagnostics": dict(rivals), "examples": examples,
              "category_note": "Categories overlap; counts must not be summed. Spelling and abbreviation labels are proxies, not confirmed causes.",
              "script_note": "Dominant Unicode letter-name prefix; ignores digits/marks/punctuation. Approximate script analysis, not translation or full Unicode Script properties.",
              "frequency_note": "Unsupervised document frequencies within each country of the retained S2/S3 training pool; common means >=1% of records.",
              "analysis_seconds": time.perf_counter()-start, "peak_memory_mib": peak_memory_mib()}
    (output / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}), flush=True)
    return report


def metrics(counts, found, totals):
    return {**count_summary(counts), "found_links": dict(found),
            "overall_recall": sum(found.values()) / sum(totals.values()),
            **{f"{s}_recall": found[s] / totals[s] if totals[s] else None for s in totals}}


def experiment(queries, pool, truth, frequencies, output):
    start = time.perf_counter()
    index = build_index(pool, CONFIGURATIONS["C"])
    extra_index = AdditionalIndex(pool, frequencies)
    prepared = {r.entity_id: prepare_weighted(r, frequencies) for r in pool}
    preparation_seconds = time.perf_counter() - start
    totals = Counter(e.split("-", 1)[0] for links in truth.values() for e in links)
    broad_counts = {v: [] for v in ("C", "C_tokens", "C_tokens_ngrams")}
    broad_found = {v: Counter() for v in broad_counts}
    variants = ("C_weighted", "C_address_rescue", "expanded_address_rescue")
    found = {v: {k: Counter() for k in KS} for v in variants}
    ranks = {v: [] for v in variants}
    failures = {"blocking": set(), "ranking": set()}
    with gzip.open(output / "failures.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            failures[row["kind"]].add((row["s1"]["entity_id"], row["true_candidate"]["entity_id"]))
    recovered = {k: Counter() for k in KS}
    newly_lost = Counter()
    start = time.perf_counter()
    for number, query in enumerate(queries, 1):
        old = generate_candidates(query, index, CONFIGURATIONS["C"])
        tokens, grams = extra_index.candidates(query)
        expanded_tokens = old | tokens
        expanded = expanded_tokens | grams
        true_ids = truth[query.entity_id]
        for variant, ids in (("C", old), ("C_tokens", expanded_tokens), ("C_tokens_ngrams", expanded)):
            broad_counts[variant].append(len(ids))
            broad_found[variant].update(e.split("-", 1)[0] for e in ids & true_ids)
        q = prepare_weighted(query, frequencies)
        weighted, rescue = [], []
        for entity_id in expanded:
            signals = improved_signals(q, prepared[entity_id], frequencies)
            if entity_id in old:
                weighted.append((entity_id, improved_score(signals, "weighted")))
            rescue.append((entity_id, improved_score(signals, "address_rescue")))
        weighted.sort(key=lambda item: (-item[1], item[0]))
        rescue.sort(key=lambda item: (-item[1], item[0]))
        ordered = {"C_weighted": weighted, "C_address_rescue": (item for item in rescue if item[0] in old),
                   "expanded_address_rescue": rescue}
        for variant, ranked in ordered.items():
            kept150 = set()
            for rank, (entity_id, _) in enumerate(ranked, 1):
                if entity_id not in true_ids:
                    continue
                ranks[variant].append(rank)
                for k in KS:
                    if rank <= k:
                        found[variant][k][entity_id.split("-", 1)[0]] += 1
                        if variant == "expanded_address_rescue":
                            for kind in failures:
                                if (query.entity_id, entity_id) in failures[kind]:
                                    recovered[k][kind] += 1
                if rank <= 150:
                    kept150.add(entity_id)
            if variant == "expanded_address_rescue":
                for entity_id in true_ids - kept150:
                    pair = (query.entity_id, entity_id)
                    if pair not in failures["blocking"] and pair not in failures["ranking"]:
                        newly_lost["previously_kept_true_links_lost_at_150"] += 1
        if number % 1000 == 0:
            print(f"Improved experiment {number:,}/20,000 S1 records", flush=True)
    sweep_seconds = time.perf_counter() - start
    if sum(broad_counts["C"]) != 14172906 or sum(broad_found["C"].values()) != 66364:
        raise ValueError("Broad baseline differs")
    peak = peak_memory_mib()
    results = {}
    for variant in variants:
        source_counts = broad_counts["C_tokens_ngrams" if variant.startswith("expanded") else "C"]
        results[variant] = {"top_k": {
            k: {**metrics([min(k, n) for n in source_counts], found[variant][k], totals),
                "shared_experiment_sweep_seconds": sweep_seconds, "shared_process_peak_mib": peak}
            for k in KS}, "true_link_ranks": rank_summary(ranks[variant])}
    report = {"broad": {v: metrics(broad_counts[v], broad_found[v], totals) for v in broad_counts},
              "ranked": results, "recovered_old_failures": {k: dict(c) for k, c in recovered.items()},
              "newly_lost": dict(newly_lost), "true_links": dict(totals),
              "preparation_seconds": preparation_seconds, "shared_sweep_seconds": sweep_seconds,
              "peak_memory_mib": peak, "timing_note": "All variants and K values share the sweep and process peak; these are not independent per-K benchmarks.",
              "frequency_scope": "Country-specific unsupervised document frequency in retained training S2/S3 pool only",
              "limitations": ["Same first-row S1 sample and truth-enriched target pools; no held-out or full-scale claim",
                              "Blocking posting cutoffs and ranking weights are fixed heuristics, not a trained model",
                              "Original baseline C union is retained; frequent extra keys are omitted, not truncated"]}
    (output / "experiment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


def write_summary(output):
    """Render compact human-readable results after both stages succeed."""
    a = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    r = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    lines = ["# Phase 2D — failure analysis and candidate improvement", "",
             "Same deterministic 20,000 S1 rows; all known matches plus 20,000 distractors per target source. Development results only; the sample is not a held-out or full-scale test.", "",
             "## Failure categories", "", a["category_note"], "", a["script_note"], "", a["frequency_note"], "",
             "| Observable category | Blocking misses (3,114) | Ranking misses (739) |",
             "|---|---:|---:|"]
    categories = sorted(set(a["categories"]["blocking"]) | set(a["categories"]["ranking"]))
    for category in categories:
        lines.append(f"| {category.replace('_', ' ')} | {a['categories']['blocking'].get(category, 0)} | {a['categories']['ranking'].get(category, 0)} |")
    lines += ["", "Spelling-change proxy: different normalized names with the same dominant script and character-bigram Dice >=0.5. Abbreviation proxy: a name token of at least two characters is a strict prefix of a token on the other side. Short name: at most two tokens. Common name: every token appears in at least 1% of same-country target-pool records. Nonempty addresses differing is a textual observation, not proof of relocation.", "",
              "## False candidates above ranking failures", ""]
    for key, count in a["ranking_rival_diagnostics"].items():
        lines.append(f"- {key.replace('_', ' ')}: {count} / 739 links.")
    lines += ["", "Rival statistics describe the highest-ranked false candidate in the old top 150 for each failed true link. Multiple failures for the same S1 can share that rival.", "",
              "## Improvements", "",
              "1. Smoothed country-specific IDF = 1 + log((N+1)/(document_frequency+1)); frequencies use the retained training target pool, without labels or a manual stopword list.",
              "2. Preserve all baseline C candidates. Union additional same-country matches on any name token with <=200 postings, or at least two nonnumeric address tokens each with <=300 postings.",
              "3. Union candidates sharing at least two of the query's four rarest available character trigrams (each <=200 postings). These operate on normalized names with whitespace removed. Omitted common keys are not truncated postings.",
              "4. Weighted name evidence N = max(nonempty exact name, 0.65*IDF token Dice + 0.35*character bigram Dice).",
              "5. Address evidence A = 0.65*IDF address-token Dice + 0.35*IDF containment when at least two nonnumeric words are shared and one occurs in <1% of same-country records. Otherwise use address Dice alone. Containment divides shared token weight by the smaller address weight.",
              "6. Numeric evidence Q = maximum IDF of a shared fully-decimal address token, divided by 1+log(N_country+1). A common number gives less evidence than a rare number.",
              "7. Weighted score = 0.65*N + 0.30*A + 0.05*Q. Address-rescue score = max(weighted score, 0.85*A + 0.10*N + 0.05*Q). Scores round to 12 decimal places; ties use ascending entity ID. Missing evidence contributes zero. K is combined across S2 and S3.", "",
              "## Broad blocking comparison", "",
              "| Version | Recall | S2 recall | S3 recall | Average candidates | Median | Maximum |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for version, m in r["broad"].items():
        lines.append(f"| {version} | {m['overall_recall']:.4%} | {m['S2_recall']:.4%} | {m['S3_recall']:.4%} | {m['average_candidates']:.2f} | {m['median_candidates']:.0f} | {m['maximum_candidates']:,} |")
    lines += ["", "## Ranking ablations", "", "Old Phase 2C K=150: recall 94.4544%, average 106.48 candidates.", "",
              "| Version | K | Recall | S2 recall | S3 recall | Average | Median | Maximum | Shared runtime (s) | Shared peak MiB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for version, results in r["ranked"].items():
        for k, m in results["top_k"].items():
            lines.append(f"| {version} | {k} | {m['overall_recall']:.4%} | {m['S2_recall']:.4%} | {m['S3_recall']:.4%} | {m['average_candidates']:.2f} | {m['median_candidates']:.0f} | {m['maximum_candidates']} | {m['shared_experiment_sweep_seconds']:.2f} | {m['shared_process_peak_mib']:.2f} |")
    lines += ["", r["timing_note"], "", "## Recovery accounting", "",
              "| K | Old blocking misses recovered | Old pruning misses recovered |",
              "|---|---:|---:|"]
    for k, counts in r["recovered_old_failures"].items():
        lines.append(f"| {k} | {counts.get('blocking', 0)} | {counts.get('ranking', 0)} |")
    lines += ["", f"Previously kept true links newly lost at K=150: {r['newly_lost'].get('previously_kept_true_links_lost_at_150', 0)}.", "",
              f"Analysis stage: {a['total_stage_seconds']:.2f}s, peak {a['peak_memory_mib']:.2f} MiB. Improvement stage: {r['total_stage_seconds']:.2f}s, peak {r['peak_memory_mib']:.2f} MiB. Improvement timing includes cache load, frequency counting, index/preparation, and joint evaluation.", "",
              "No models, GPU, external data/APIs, test files, predictions, submissions, or branch merges. Original TSVs and prior source modules were not modified.", "",
              "Full failed-pair details: failures.jsonl.gz. Representative pairs: examples.md. Exact metrics: analysis.json and experiment.json. The approximately 12 MiB sample cache avoids repeated large training-file scans."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# Representative old failures", "",
             "First two observed pairs per category, deduplicated for display. Country values below are normalized. Names and addresses include original and normalized text. These are examples, not prevalence estimates.", ""]
    for kind in ("blocking", "ranking"):
        lines += [f"## {kind.title()} failures", ""]
        seen = set()
        for category, examples in a["examples"][kind].items():
            for row in examples:
                key = (row["s1"]["entity_id"], row["true_candidate"]["entity_id"])
                if key in seen:
                    continue
                seen.add(key)
                lines += [f"### {key[0]} → {key[1]}", "", f"Categories: {', '.join(row['categories'])}", ""]
                for label, record in (("S1", row["s1"]), ("True candidate", row["true_candidate"])):
                    lines += [f"**{label}:** {record['entity_id']}; country: {record['country']}", "",
                              f"- Name: {record['business_name']}", f"- Normalized name: {record['name']}",
                              f"- Address: {record['business_address'] or '(missing)'}",
                              f"- Normalized address: {record['address'] or '(missing)'}", ""]
                lines += [f"Script profiles: {row['s1_script']} → {row['candidate_script']}", ""]
                if kind == "ranking":
                    lines += [f"Old rank: {row['rank']}; true score: {row['score']:.6f}; cutoff score: {row['cutoff']['score']:.6f}.", ""]
                    rival = row.get("highest_false_rival")
                    if rival:
                        lines += [f"Highest false rival: {rival['record']['entity_id']} — {rival['record']['business_name']}; address: {rival['record']['business_address']}; score: {rival['score']:.6f}.", "",
                                  f"Shared name tokens: {rival['shared_name_tokens']}; shared numbers: {rival['shared_numbers']}.", ""]
    (output / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("analyze", "experiment"), required=True)
    parser.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/candidate_improvement"))
    parser.add_argument("--baseline-report", type=Path, default=Path("output/candidate_baseline/report.json"))
    args = parser.parse_args()
    start = time.perf_counter()
    queries, pool, truth, stats = load_sample(args.train_dir, args.output_dir, args.baseline_report)
    loading = time.perf_counter() - start
    freq_start = time.perf_counter()
    frequencies = Frequencies.fit(pool)
    frequency_seconds = time.perf_counter() - freq_start
    if args.stage == "analyze":
        report = analyze(queries, pool, truth, frequencies, args.output_dir)
    else:
        report = experiment(queries, pool, truth, frequencies, args.output_dir)
    report.update(sample_size=len(queries), pool_stats=stats, loading_seconds=loading,
                  frequency_seconds=frequency_seconds, total_stage_seconds=time.perf_counter()-start)
    path = args.output_dir / ("analysis.json" if args.stage == "analyze" else "experiment.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.stage == "experiment":
        write_summary(args.output_dir)


if __name__ == "__main__":
    main()
