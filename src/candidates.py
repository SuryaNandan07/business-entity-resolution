"""Training-sample blocking baseline. Run: python -m src.candidates

Streams training files to retain a bounded sample, then evaluates blocking.
Ground truth selects the evaluation pool and labels results only; it is never
used by the blocking rules. No similarities, models, or test data are used.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import statistics
import time

from src.normalize import normalize_address, normalize_name, normalize_text


CONFIGURATIONS = {
    "A": ("name_token",),
    "B": ("name_token", "address_number"),
    "C": ("name_token", "address_number", "name_prefix"),
}


@dataclass(slots=True)
class Record:
    entity_id: str
    business_name: str
    business_address: str
    country: str
    name: str
    address: str

    @classmethod
    def from_row(cls, row):
        name = row.get("business_name") or ""
        address = row.get("business_address") or ""
        return cls(row["entity_id"], name, address,
                   normalize_text(row.get("country")),
                   normalize_name(name), normalize_address(address))


def read_rows(path):
    """Read one TSV record at a time; never load a full dataframe."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def load_s1_sample(path, size):
    from itertools import islice
    records = [Record.from_row(row) for row in islice(read_rows(path), size)]
    if len({r.entity_id for r in records}) != len(records):
        raise ValueError("Duplicate S1 entity IDs in sample")
    return records


def load_truth(path, s1_ids):
    truth = {entity_id: set() for entity_id in s1_ids}
    covered = set()
    for row in read_rows(path):
        entity_id = row["source1_entity_id"]
        if entity_id in truth:
            covered.add(entity_id)
            for match in row["matched_entity_ids"].split(","):
                match = match.strip()
                if match:
                    if not match.startswith(("S2-", "S3-")):
                        raise ValueError(f"Unexpected truth ID: {match}")
                    truth[entity_id].add(match)
    if covered != s1_ids:
        raise ValueError(f"Missing ground truth for {len(s1_ids - covered)} S1 rows")
    return truth


def load_candidate_pool(path, required_ids, distractor_count, seed):
    """Retain every required match and a uniform reservoir of other rows.

    Distractors are not known matches of ANY sampled S1. A sequential scan is
    necessary to locate matches; only retained rows are normalized afterwards.
    """
    rng = random.Random(seed)
    required = {}
    reservoir = []
    other_count = 0
    scanned = 0
    for row in read_rows(path):
        scanned += 1
        entity_id = row["entity_id"]
        if entity_id in required_ids:
            required[entity_id] = row
        else:
            other_count += 1
            if len(reservoir) < distractor_count:
                reservoir.append(row)
            else:
                position = rng.randrange(other_count)
                if position < distractor_count:
                    reservoir[position] = row
    missing = required_ids - required.keys()
    if missing:
        raise ValueError(f"{len(missing)} true matches missing from {path}")
    records = [Record.from_row(row) for row in required.values()]
    records.extend(Record.from_row(row) for row in reservoir)
    if len({r.entity_id for r in records}) != len(records):
        raise ValueError(f"Duplicate entity IDs in retained pool: {path}")
    return records, {"scanned_rows": scanned, "true_match_records": len(required),
                     "distractors": len(reservoir), "retained_rows": len(records)}


def blocking_keys(record, rules):
    """Country equality plus first name token, address number, or 4-char prefix.

    Empty countries do not block together. Empty names/addresses produce no
    corresponding keys. Numeric tokens use isdecimal(), preserving Unicode.
    No bucket truncation or candidate cap silently removes possible matches.
    """
    if not record.country:
        return
    if "name_token" in rules and record.name:
        yield ("name_token", record.country, record.name.split()[0])
    if "address_number" in rules:
        for token in set(record.address.split()):
            if token.isdecimal():
                yield ("address_number", record.country, token)
    if "name_prefix" in rules and record.name:
        yield ("name_prefix", record.country, record.name[:4])


def build_index(records, rules):
    index = defaultdict(list)
    for record in records:
        for key in blocking_keys(record, rules):
            index[key].append(record.entity_id)
    return index


def generate_candidates(record, index, rules):
    """Union rule hits for one S1 only, removing duplicate candidate IDs."""
    candidates = set()
    for key in blocking_keys(record, rules):
        candidates.update(index.get(key, ()))
    return candidates


def recall(found, total):
    return found / total if total else None


def evaluate(records, pool, truth, rules, pairs_path):
    started = time.perf_counter()
    index = build_index(pool, rules)
    index_seconds = time.perf_counter() - started
    counts = []
    totals = {"S2": 0, "S3": 0}
    found = {"S2": 0, "S3": 0}
    with pairs_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_id", "candidate_source"])
        for record in records:
            candidates = generate_candidates(record, index, rules)
            counts.append(len(candidates))
            for entity_id in sorted(candidates):
                writer.writerow([record.entity_id, entity_id, entity_id.split("-", 1)[0]])
            for entity_id in truth[record.entity_id]:
                source = entity_id.split("-", 1)[0]
                totals[source] += 1
                found[source] += entity_id in candidates
    metrics = {
        "rules": list(rules),
        "overall_recall": recall(sum(found.values()), sum(totals.values())),
        "S2_recall": recall(found["S2"], totals["S2"]),
        "S3_recall": recall(found["S3"], totals["S3"]),
        "true_links": totals, "found_links": found,
        "average_candidates": statistics.mean(counts),
        "median_candidates": statistics.median(counts),
        "maximum_candidates": max(counts),
        "zero_candidate_percent": 100 * counts.count(0) / len(counts),
        "total_pairs": sum(counts), "index_seconds": index_seconds,
        "runtime_seconds": time.perf_counter() - started,
        "pairs_file": str(pairs_path),
    }
    return metrics


def make_examples(records, pool, truth, rules):
    """First ten S1 rows, without cherry-picking; include every candidate."""
    index = build_index(pool, rules)
    lookup = {record.entity_id: record for record in pool}
    examples = []
    for record in records[:10]:
        candidates = generate_candidates(record, index, rules)
        examples.append({
            "entity_id": record.entity_id, "business_name": record.business_name,
            "business_address": record.business_address,
            "missed_true_ids": sorted(truth[record.entity_id] - candidates),
            "candidates": [
                {"entity_id": entity_id, "business_name": lookup[entity_id].business_name,
                 "label": "TRUE MATCH" if entity_id in truth[record.entity_id]
                 else "FALSE CANDIDATE"}
                for entity_id in sorted(candidates)
            ],
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/candidate_baseline"))
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--distractors", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.sample_size <= 20000 or not 0 <= args.distractors <= 20000:
        parser.error("Sample-only guard: 1..20000 S1 rows and 0..20000 distractors/source")
    started = time.perf_counter()
    records = load_s1_sample(args.train_dir / "train_source1.tsv", args.sample_size)
    if not records:
        raise ValueError("Empty S1 sample")
    truth = load_truth(args.train_dir / "train_ground_truth.tsv", {r.entity_id for r in records})
    required = set().union(*truth.values())
    pool = []
    pool_stats = {}
    for source in (2, 3):
        selected, stats = load_candidate_pool(
            args.train_dir / f"train_source{source}.tsv",
            {entity_id for entity_id in required if entity_id.startswith(f"S{source}-")},
            args.distractors, args.seed + source,
        )
        pool.extend(selected)
        del selected
        pool_stats[f"S{source}"] = stats
        print(f"S{source}: {stats}", flush=True)
    loading_seconds = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for version, rules in CONFIGURATIONS.items():
        results[version] = evaluate(records, pool, truth, rules,
                                   args.output_dir / f"training_sample_pairs_{version}.tsv")
        print(json.dumps({version: results[version]}), flush=True)
    best = max(results, key=lambda v: (results[v]["overall_recall"],
                                      -results[v]["average_candidates"]))
    examples = make_examples(records, pool, truth, CONFIGURATIONS[best])
    report = {"sample_size": len(records), "sampling": "First S1 rows; all true matches plus seeded reservoir distractors",
              "seed": args.seed, "pool_stats": pool_stats, "configurations": results,
              "best_by_recall_then_candidate_count": best,
              "loading_seconds": loading_seconds,
              "total_runtime_seconds": time.perf_counter() - started,
              "examples": examples}
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# Training sample candidate examples — configuration {best}",
             "", "First ten sampled S1 records; complete candidate sets. Labels use training ground truth.", ""]
    for example in examples:
        lines.extend([f"## {example['entity_id']}", "",
                      f"Name: {example['business_name']}", "",
                      f"Address: {example['business_address'] or '(empty)'}", "",
                      f"Missed true IDs: {', '.join(example['missed_true_ids']) or '(none)'}", ""])
        for candidate in example["candidates"]:
            lines.append(f"- {candidate['entity_id']} | {candidate['business_name']} | {candidate['label']}")
        if not example["candidates"]:
            lines.append("No candidates.")
        lines.append("")
    (args.output_dir / "examples.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {args.output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
