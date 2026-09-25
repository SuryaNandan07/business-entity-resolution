"""Small synthetic checks for blocking, sampling, and link recall."""

import csv
from pathlib import Path
import tempfile
import unittest

from src.candidates import (CONFIGURATIONS, Record, build_index, evaluate,
                            generate_candidates, load_candidate_pool, load_truth)


def record(entity_id, name, address="", country="X"):
    return Record.from_row(dict(entity_id=entity_id, business_name=name,
                                business_address=address, country=country))


class CandidateTests(unittest.TestCase):
    def test_union_and_deduplication(self):
        query = record("S1-1", "Alpha Shop", "25 Road")
        pool = [record("S2-1", "Alpha Shop", "25 Road"),
                record("S2-2", "Other", "25 Street"),
                record("S3-1", "Alphabet", ""),
                record("S3-2", "Alpha Shop", "25 Road", "Y")]
        expected = [{"S2-1"}, {"S2-1", "S2-2"}, {"S2-1", "S2-2", "S3-1"}]
        for rules, ids in zip(CONFIGURATIONS.values(), expected):
            self.assertEqual(generate_candidates(query, build_index(pool, rules), rules), ids)

    def test_unicode_missing_values_and_country(self):
        rules = CONFIGURATIONS["C"]
        query = record("S1-1", "École", None)
        pool = [record("S2-1", "E\u0301cole"), record("S3-1", "राम दुकान")]
        index = build_index(pool, rules)
        self.assertEqual(generate_candidates(query, index, rules), {"S2-1"})
        self.assertEqual(generate_candidates(record("S1-2", "राम दुकान"), index, rules), {"S3-1"})
        self.assertEqual(generate_candidates(record("S1-3", "", None), index, rules), set())
        self.assertEqual(generate_candidates(record("S1-4", "École", country=""), index, rules), set())

    def test_one_to_many_recall_and_zero_candidates(self):
        queries = [record("S1-1", "Alpha"), record("S1-2", "Unseen")]
        pool = [record("S2-1", "Alpha"), record("S2-2", "Alpha"), record("S3-1", "Other")]
        truth = {"S1-1": {"S2-1", "S2-2", "S3-1"}, "S1-2": set()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.tsv"
            metrics = evaluate(queries, pool, truth, CONFIGURATIONS["A"], path)
            self.assertEqual(metrics["overall_recall"], 2 / 3)
            self.assertEqual(metrics["S2_recall"], 1)
            self.assertEqual(metrics["S3_recall"], 0)
            self.assertEqual(metrics["average_candidates"], 1)
            self.assertEqual(metrics["median_candidates"], 1)
            self.assertEqual(metrics["maximum_candidates"], 2)
            self.assertEqual(metrics["zero_candidate_percent"], 50)
            with path.open() as stream:
                pairs = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0]["candidate_source"], "S2")

    def test_streamed_pool_keeps_true_matches_and_bounded_distractors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tsv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["entity_id", "business_name", "business_address", "country"])
                for i in range(20):
                    writer.writerow([f"S2-{i}", "Name", "", "X"])
            pool, stats = load_candidate_pool(path, {"S2-19", "S2-18"}, 3, 42)
            again, _ = load_candidate_pool(path, {"S2-19", "S2-18"}, 3, 42)
            self.assertEqual(len(pool), 5)
            self.assertTrue({"S2-18", "S2-19"}.issubset({r.entity_id for r in pool}))
            self.assertEqual([r.entity_id for r in pool], [r.entity_id for r in again])
            self.assertEqual(stats["scanned_rows"], 20)
            with self.assertRaises(ValueError):
                load_candidate_pool(path, {"S2-missing"}, 3, 42)

    def test_truth_merges_duplicate_links_and_requires_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.tsv"
            path.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S2-1,S3-1\n", encoding="utf-8")
            self.assertEqual(load_truth(path, {"S1-1"}), {"S1-1": {"S2-1", "S3-1"}})
            with self.assertRaises(ValueError):
                load_truth(path, {"S1-2"})


if __name__ == "__main__":
    unittest.main()
