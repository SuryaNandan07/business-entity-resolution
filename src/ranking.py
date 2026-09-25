"""Memory-bounded Phase 2C experiment: python -m src.ranking.

Rebuild the Phase 2B sample deterministically, rank configuration C candidates,
and evaluate several top-K cutoffs. No models or pair files are produced.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import NamedTuple

from src.candidates import (CONFIGURATIONS, build_index, generate_candidates,
                            load_candidate_pool, load_s1_sample, load_truth)


K_VALUES = (20, 30, 50, 75, 100, 150)
# Columns: exact name, name-token Jaccard, character-bigram Dice,
# four-character prefix agreement, address-token Jaccard, shared number.
FORMULAS = {
    "name_first": (0.15, 0.30, 0.30, 0.05, 0.17, 0.03),
    "balanced": (0.10, 0.25, 0.25, 0.05, 0.30, 0.05),
    "address_supported": (0.10, 0.20, 0.25, 0.05, 0.35, 0.05),
}


@dataclass(slots=True)
class Prepared:
    name: str
    name_tokens: frozenset
    name_bigrams: frozenset
    address_tokens: frozenset
    numbers: frozenset
    country: str


def prepare(record):
    """Cache token sets per retained record, not per pair."""
    compact_name = "".join(record.name.split())
    bigrams = frozenset(compact_name[i:i + 2] for i in range(len(compact_name) - 1))
    address = frozenset(record.address.split())
    return Prepared(record.name, frozenset(record.name.split()), bigrams,
                    address, frozenset(t for t in address if t.isdecimal()), record.country)


def jaccard(left, right):
    """Empty sets provide no positive evidence, even when both are empty."""
    intersection = len(left & right)
    union_size = len(left) + len(right) - intersection
    return intersection / union_size if union_size else 0.0


class Signals(NamedTuple):
    name_exact: float
    name_jaccard: float
    name_bigram_dice: float
    name_prefix: float
    address_jaccard: float
    shared_number: float
    country_equal: bool
    missing_address: bool


SIGNAL_NAMES = Signals._fields[:6]


def pair_signals(left, right):
    """Cheap, Unicode-preserving evidence; no labels or country-specific rules."""
    prefix_length = 0
    for a, b in zip(left.name[:4], right.name[:4]):
        if a != b:
            break
        prefix_length += 1
    gram_count = len(left.name_bigrams) + len(right.name_bigrams)
    dice = 2 * len(left.name_bigrams & right.name_bigrams) / gram_count if gram_count else 0.0
    return Signals(
        float(bool(left.name) and left.name == right.name),
        jaccard(left.name_tokens, right.name_tokens), dice, prefix_length / 4,
        jaccard(left.address_tokens, right.address_tokens),
        float(bool(left.numbers & right.numbers)),
        bool(left.country) and left.country == right.country,
        not left.address_tokens or not right.address_tokens,
    )


def ranking_score(signals, formula="balanced"):
    """Weighted evidence in [0, 1]; country mismatch is ineligible.

    Country is a gate, not a ranking bonus constant within every block.
    Missing-address is diagnostic: address terms are already zero, so there
    is no extra penalty and weights are not redistributed.
    """
    if not signals.country_equal:
        return float("-inf")
    return sum(weight * value for weight, value in zip(FORMULAS[formula], signals[:6]))


def rank_candidates(query, candidate_ids, prepared_pool, formula="balanced"):
    """Deduplicate and rank one candidate set; break ties by entity ID."""
    scored = []
    for entity_id in set(candidate_ids):
        score = ranking_score(pair_signals(query, prepared_pool[entity_id]), formula)
        if math.isfinite(score):
            scored.append((entity_id, score))
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def top_k(ranked, k):
    """Keep at most K across both sources, allowing many true matches."""
    if k < 1:
        raise ValueError("K must be positive")
    return ranked[:k]


def rank_summary(ranks):
    """Ranks are one-based; percentiles use the nearest-rank definition."""
    if not ranks:
        return {"count": 0, "median": None, "p90": None, "p95": None,
                "p99": None, "maximum": None}
    ordered = sorted(ranks)
    return {"count": len(ordered), "median": statistics.median(ordered),
            **{f"p{p}": ordered[math.ceil(p / 100 * len(ordered)) - 1]
               for p in (90, 95, 99)}, "maximum": ordered[-1]}


def count_summary(counts):
    return {"average_candidates": statistics.mean(counts),
            "median_candidates": statistics.median(counts),
            "maximum_candidates": max(counts),
            "zero_candidate_percent": 100 * counts.count(0) / len(counts),
            "total_pairs": sum(counts)}


def evaluate_rankings(records, truth, index, prepared_pool):
    """Compute signals once per pair, then test three fixed formulas.

    Retain one S1's signals/scores at a time plus small aggregate counters and
    true-link rank lists. Evaluate all K values from the same sorted list.
    """
    rules = CONFIGURATIONS["C"]
    totals = {source: sum(sum(e.startswith(source + "-") for e in links)
                          for links in truth.values()) for source in ("S2", "S3")}
    broad_found = {"S2": 0, "S3": 0}
    counts = []
    found = {f: {k: {"S2": 0, "S3": 0} for k in K_VALUES} for f in FORMULAS}
    ranks = {f: {"S2": [], "S3": []} for f in FORMULAS}
    shared_seconds = 0.0
    formula_seconds = {f: 0.0 for f in FORMULAS}
    for number, record in enumerate(records, 1):
        start = time.perf_counter()
        candidates = generate_candidates(record, index, rules)
        query = prepare(record)
        evidence = [(entity_id, pair_signals(query, prepared_pool[entity_id]))
                    for entity_id in candidates]
        counts.append(len(candidates))
        true_ids = truth[record.entity_id]
        present = candidates & true_ids
        for entity_id in present:
            broad_found[entity_id.split("-", 1)[0]] += 1
        shared_seconds += time.perf_counter() - start
        for formula in FORMULAS:
            start = time.perf_counter()
            scored = [(entity_id, ranking_score(signals, formula))
                      for entity_id, signals in evidence]
            scored.sort(key=lambda item: (-item[1], item[0]))
            for rank, (entity_id, _) in enumerate(scored, 1):
                if entity_id in true_ids:
                    source = entity_id.split("-", 1)[0]
                    ranks[formula][source].append(rank)
                    for k in K_VALUES:
                        if rank <= k:
                            found[formula][k][source] += 1
            formula_seconds[formula] += time.perf_counter() - start
        if number % 1000 == 0:
            print(f"Ranked {number:,}/{len(records):,} S1 records", flush=True)
    total_links = sum(totals.values())
    if total_links == 0:
        raise ValueError("Sample contains no true links to evaluate")
    baseline = {**count_summary(counts), "found_links": broad_found,
                "true_links": totals, "overall_recall": sum(broad_found.values()) / total_links,
                **{f"{s}_recall": broad_found[s] / totals[s] if totals[s] else None for s in totals}}
    results = {}
    for formula in FORMULAS:
        metrics = {}
        for k in K_VALUES:
            kept = found[formula][k]
            metrics[k] = {
                **count_summary([min(k, count) for count in counts]),
                "overall_recall": sum(kept.values()) / total_links,
                **{f"{s}_recall": kept[s] / totals[s] if totals[s] else None for s in totals},
                "found_links": kept,
                "lost_due_to_pruning": sum(broad_found.values()) - sum(kept.values()),
                "lost_due_to_pruning_by_source": {s: broad_found[s] - kept[s] for s in totals},
                "total_missed_links": total_links - sum(kept.values()),
                "shared_sweep_runtime_seconds": shared_seconds + formula_seconds[formula],
            }
        results[formula] = {
            "weights": FORMULAS[formula], "top_k": metrics,
            "true_match_ranks": {"overall": rank_summary(ranks[formula]["S2"] + ranks[formula]["S3"]),
                                 **{s: rank_summary(ranks[formula][s]) for s in totals}},
            "unranked_true_links_absent_from_broad_pool": total_links - sum(broad_found.values()),
            "score_sort_evaluation_seconds": formula_seconds[formula],
        }
    return baseline, results, shared_seconds


def example_sets(records, pool, truth, index, prepared_pool, formula, k):
    """First ten successes and failures in S1 order; compact, explicit examples."""
    successes, failures = [], []
    for record in records:
        query = prepare(record)
        ids = generate_candidates(record, index, CONFIGURATIONS["C"])
        ranked = rank_candidates(query, ids, prepared_pool, formula)
        positions = {entity_id: (rank, score) for rank, (entity_id, score) in enumerate(ranked, 1)}
        true_ids = truth[record.entity_id]
        lost = sorted(e for e in true_ids & ids if positions[e][0] > k)
        is_success = bool(true_ids) and true_ids <= {e for e, _ in top_k(ranked, k)} and len(ids) > k
        if (is_success and len(successes) < 10) or (lost and len(failures) < 10):
            def describe(entity_id):
                rank, score = positions[entity_id]
                candidate = pool[entity_id]
                return {"entity_id": entity_id, "business_name": candidate.business_name,
                        "business_address": candidate.business_address, "rank": rank,
                        "score": score, "kept": rank <= k,
                        "label": "TRUE MATCH" if entity_id in true_ids else "FALSE CANDIDATE",
                        "signals": pair_signals(query, prepared_pool[entity_id])._asdict()}
            example = {"entity_id": record.entity_id, "business_name": record.business_name,
                       "business_address": record.business_address, "broad_count": len(ids),
                       "kept_count": min(k, len(ids)),
                       "true_candidates": [describe(e) for e in sorted(true_ids & ids)],
                       "blocking_misses": sorted(true_ids - ids),
                       "pruning_misses": lost,
                       "cutoff_candidate": describe(ranked[k - 1][0]) if len(ranked) >= k else None}
            if is_success and len(successes) < 10:
                successes.append(example)
            if lost and len(failures) < 10:
                failures.append(example)
        if len(successes) == 10 and len(failures) == 10:
            break
    return {"formula": formula, "k": k, "successes": successes, "failures": failures}


def peak_memory_mib():
    """Read the OS process high-water mark; no polling thread is needed."""
    import sys
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        read = ctypes.windll.psapi.GetProcessMemoryInfo
        read.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not read(current(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return counters.PeakWorkingSetSize / 1024**2
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**2 if sys.platform == "darwin" else 1024)


def write_reports(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    formula = report["selected_formula"]
    lines = ["# Phase 2C ranking experiment", "", f"Selected formula: {formula}.", "",
             "20,000 S1 records and deterministic Phase 2B pools. All metrics are development-sample results.", "",
             "| Formula | K | Recall | S2 recall | S3 recall | Mean | Median | Max | Zero % | Pruning losses | Shared sweep seconds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    b = report["baseline"]
    lines.append(f"| Phase 2B broad C | All | {b['overall_recall']:.4%} | {b['S2_recall']:.4%} | {b['S3_recall']:.4%} | {b['average_candidates']:.2f} | {b['median_candidates']:.0f} | {b['maximum_candidates']} | {b['zero_candidate_percent']:.3f} | 0 | Not re-timed separately |")
    for name, result in report["formulas"].items():
        for k, m in result["top_k"].items():
            lines.append(f"| {name} | {k} | {m['overall_recall']:.4%} | {m['S2_recall']:.4%} | {m['S3_recall']:.4%} | {m['average_candidates']:.2f} | {m['median_candidates']:.0f} | {m['maximum_candidates']} | {m['zero_candidate_percent']:.3f} | {m['lost_due_to_pruning']} | {m['shared_sweep_runtime_seconds']:.2f} |")
    lines += ["", "Each formula's six K values share one ranking pass. Sweep times include shared generation/signals plus that formula's scoring/sorting/evaluation; they are not six independent timings and exclude sample loading and index preparation.", "",
              "## Scoring", "",
              "All formulas are weighted sums of the following signals in this order: " + ", ".join(report["signal_order"]) + ".", "",
              *[f"- {name}: {result['weights']}" for name, result in report["formulas"].items()], "",
              *[f"- {name}: {definition}" for name, definition in report["signal_definitions"].items()], "",
              "Missing-address evidence is zero without an extra penalty or weight redistribution. Country equality is a gate, not a constant bonus. Ties use entity ID ascending. K is shared across S2 and S3, not applied separately per source.", "",
              f"Formula selection: {report['formula_selection_rule']}", "",
              f"K selection: {report['k_selection_rule']}", "",
              f"Recommended K: {report['recommended_k']}", "",
              f"0.5-percentage-point loss target met: {report['k_loss_target_met']}. When false, K=150 is only the highest-recall tested fallback, not a configuration meeting the target.", "",
              "True-match ranks are one-based and exclude links absent from broad blocking. Percentiles use nearest rank; median uses the usual midpoint.", "",
              "```json", json.dumps(report["formulas"][formula]["true_match_ranks"], indent=2), "```", "",
              f"Timing: {report['timing']}", "", f"Peak process working set/RSS: {report['peak_memory_mib']:.2f} MiB.", "",
              "## Limitations", "", *[f"- {item}" for item in report["limitations"]], "",
              "See report.json for exact formulas, baseline comparison, counts, and signal-level examples. See examples.md for ten successes and ten pruning failures.", "",
              "No trained model, full-scale/test-set work, candidate TSVs, predictions, or submission files were produced."]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    lines = [f"# Examples: {formula}, K={report['recommended_k']}", "",
             "First ten qualifying S1 records in sample order in each category. Success means all known true links survived and at least one candidate was removed. Failure means at least one broad-pool true link ranked below K.", ""]
    for category in ("successes", "failures"):
        lines += [f"## {category.title()}", ""]
        for e in report["examples"][category]:
            lines += [f"### {e['entity_id']} — {e['business_name']}", "", f"Address: {e['business_address'] or '(missing)'}", "",
                      f"Candidates: {e['broad_count']} → {e['kept_count']}. Blocking misses: {e['blocking_misses']}. Pruning misses: {e['pruning_misses']}.", ""]
            for c in e["true_candidates"]:
                lines.append(f"- {c['entity_id']} | {c['business_name']} | rank {c['rank']} | score {c['score']:.6f} | {'KEPT' if c['kept'] else 'PRUNED'} | address: {c['business_address']}")
            cutoff = e["cutoff_candidate"]
            if cutoff:
                lines += ["", f"At cutoff: {cutoff['entity_id']} | {cutoff['business_name']} | {cutoff['label']} | score {cutoff['score']:.6f}"]
            lines.append("")
    (output / "examples.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--baseline-report", type=Path, default=Path("output/candidate_baseline/report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/candidate_ranking"))
    args = parser.parse_args()
    started = time.perf_counter()
    previous = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    if previous["sample_size"] != 20000 or previous["seed"] != 42:
        raise ValueError("Expected Phase 2B sample of 20,000 rows with seed 42")
    records = load_s1_sample(args.train_dir / "train_source1.tsv", 20000)
    if len(records) != 20000:
        raise ValueError("Expected 20,000 S1 rows")
    truth = load_truth(args.train_dir / "train_ground_truth.tsv", {r.entity_id for r in records})
    required = set().union(*truth.values())
    pool = []
    stats = {}
    for source in (2, 3):
        selected, stats[f"S{source}"] = load_candidate_pool(
            args.train_dir / f"train_source{source}.tsv",
            {e for e in required if e.startswith(f"S{source}-")}, 20000, 42 + source)
        pool.extend(selected)
        del selected
        print(f"Loaded S{source}: {stats[f'S{source}']}", flush=True)
    if stats != previous["pool_stats"]:
        raise ValueError("Rebuilt pool counts differ from Phase 2B")
    loading_seconds = time.perf_counter() - started
    start = time.perf_counter()
    index = build_index(pool, CONFIGURATIONS["C"])
    prepared = {r.entity_id: prepare(r) for r in pool}
    prep_seconds = time.perf_counter() - start
    start = time.perf_counter()
    baseline, results, shared_seconds = evaluate_rankings(records, truth, index, prepared)
    sweep_seconds = time.perf_counter() - start
    for key in ("total_pairs", "found_links", "true_links", "maximum_candidates"):
        if baseline[key] != previous["configurations"]["C"][key]:
            raise ValueError(f"Broad baseline changed: {key}")
    # Fixed selection criteria, not a trained model or an exhaustive weight search.
    selected_formula = max(FORMULAS, key=lambda f: (results[f]["top_k"][50]["overall_recall"],
                                                   results[f]["top_k"][20]["overall_recall"]))
    eligible = [k for k in K_VALUES if baseline["overall_recall"] - results[selected_formula]["top_k"][k]["overall_recall"] <= 0.005]
    chosen_k = min(eligible) if eligible else max(K_VALUES)
    start = time.perf_counter()
    examples = example_sets(records, {r.entity_id: r for r in pool}, truth, index, prepared, selected_formula, chosen_k)
    example_seconds = time.perf_counter() - start
    report = {
        "sample_size": len(records), "seed": 42, "pool_stats": stats,
        "baseline_reproduced": True, "baseline": baseline, "formulas": results,
        "signal_order": list(SIGNAL_NAMES),
        "signal_definitions": {
            "name_exact": "1 for equal nonempty normalized names, else 0",
            "name_jaccard": "intersection/union of normalized name token sets; empty gives 0",
            "name_bigram_dice": "2*intersection/(size1+size2) of unique character bigrams after removing name whitespace; empty gives 0",
            "name_prefix": "common leading characters among the first four, divided by 4",
            "address_jaccard": "intersection/union of all normalized address tokens, including numbers; empty gives 0",
            "shared_number": "1 if any fully decimal address token is shared, else 0",
            "country_equal": "nonempty exact normalized country equality; eligibility gate",
            "missing_address": "either address empty; diagnostic only, with zero address evidence"},
        "tie_break": "entity_id ascending, after descending score",
        "selected_formula": selected_formula,
        "formula_selection_rule": "Highest development recall at K=50; tie broken by recall at K=20, then declared formula order",
        "recommended_k": chosen_k,
        "k_selection_rule": "Smallest tested K losing at most 0.5 percentage points versus broad recall; use 150 if none qualify",
        "k_loss_target_met": bool(eligible),
        "rank_scope": "True links present in broad candidate sets only; blocking misses have no finite rank",
        "timing": {"loading_seconds": loading_seconds, "preparation_seconds": prep_seconds,
                   "shared_generation_and_signals_seconds": shared_seconds,
                   "all_formula_sweeps_seconds": sweep_seconds, "examples_seconds": example_seconds,
                   "total_seconds_before_report_write": time.perf_counter() - started},
        "peak_memory_mib": peak_memory_mib(), "examples": examples,
        "limitations": ["Same first-row S1 sample and truth-enriched candidate pools as Phase 2B",
                        "Formula/K selection uses development labels; no held-out generalization claim",
                        "Full-scale or unseen-country behavior is not measured",
                        "No RapidFuzz in active runtime; no external dependencies added"],
    }
    write_reports(report, args.output_dir)
    print(json.dumps({"selected_formula": selected_formula, "recommended_k": chosen_k,
                      "metrics": results[selected_formula]["top_k"], "timing": report["timing"],
                      "peak_memory_mib": report["peak_memory_mib"]}), flush=True)


if __name__ == "__main__":
    main()
