"""Small feature/split/streaming regressions; never read hackathon files."""

import csv
import gzip
from pathlib import Path
import tempfile
import unittest

from src.candidates import Record
from src.candidate_improvements import Frequencies
from src.features import FEATURE_NAMES, label_pair, pair_features, split_s1_entities
from src.build_features import FeatureStatistics, build_pair_files, load_frequencies, save_frequencies


def record(entity_id, name, address="", country="X"):
    return Record.from_row(dict(entity_id=entity_id, business_name=name,
                                business_address=address, country=country))


class FeatureTests(unittest.TestCase):
    def features(self, a, b):
        return pair_features(a, b, Frequencies.fit([a, b]))

    def test_exact_name_and_address(self):
        f = self.features(record("S1-1", "ABC Pvt. Ltd.", "25, Main Road"),
                          record("S2-1", "abc pvt ltd", "25 MAIN ROAD"))
        for name in ("name_exact", "name_token_jaccard", "name_bigram_dice", "name_trigram_dice",
                     "name_length_ratio", "name_rare_token_dice", "address_exact", "address_token_jaccard",
                     "address_bigram_dice", "address_rare_token_dice", "address_length_ratio"):
            self.assertEqual(f[name], 1, name)
        self.assertEqual(tuple(f), FEATURE_NAMES)

    def test_completely_different_names(self):
        f = self.features(record("S1-1", "AAAA"), record("S2-1", "ZZZZ"))
        for name in ("name_exact", "name_token_jaccard", "name_bigram_dice", "name_trigram_dice",
                     "name_prefix_similarity", "name_rare_token_dice"):
            self.assertEqual(f[name], 0)

    def test_missing_addresses_and_empty_names(self):
        f = self.features(record("S1-1", None, None), record("S3-1", "", ""))
        for name in ("name_exact", "name_token_jaccard", "address_exact", "address_token_jaccard",
                     "address_bigram_dice", "address_length_ratio", "numeric_token_jaccard"):
            self.assertEqual(f[name], 0)
        for name in ("s1_address_missing", "candidate_address_missing", "both_addresses_missing",
                     "s1_name_missing", "candidate_name_missing"):
            self.assertEqual(f[name], 1)
        f = self.features(record("S1-1", "Name", "25 Main"), record("S3-1", "Name", None))
        self.assertEqual((f["s1_address_missing"], f["candidate_address_missing"], f["both_addresses_missing"]), (0, 1, 0))

    def test_numeric_tokens_preserve_leading_zeroes(self):
        f = self.features(record("S1-1", "A", "25 25/7 001"), record("S2-1", "B", "25/7 1"))
        self.assertEqual(f["shared_numeric_token_count"], 2)
        self.assertEqual(f["numeric_token_jaccard"], .5)

    def test_unicode_and_non_latin(self):
        f = self.features(record("S1-1", "École Française"), record("S2-1", "E\u0301COLE FRANÇAISE"))
        self.assertEqual(f["name_exact"], 1)
        f = self.features(record("S1-1", "राम दुकान", "२५ सड़क"), record("S3-1", "राम दुकान", "२५ सड़क"))
        self.assertEqual(f["name_trigram_dice"], 1)
        self.assertEqual(f["shared_numeric_token_count"], 1)

    def test_multiple_true_matches_and_label_separation(self):
        truth = {"S1-1": {"S2-1", "S3-1"}}
        self.assertEqual([label_pair("S1-1", e, truth) for e in ("S2-1", "S3-1", "S2-2")], [1, 1, 0])
        f = self.features(record("S1-1", "A"), record("S2-1", "B"))
        self.assertNotIn("label", f)
        self.assertNotIn("source1_entity_id", f)
        with self.assertRaises(KeyError):
            label_pair("S1-missing", "S2-1", truth)

    def test_country_equality_and_weighted_bounds(self):
        a, b = record("S1-1", "Same", "25 Road", "X"), record("S3-1", "Same", "25 Road", "Y")
        frequencies = Frequencies.fit([a] + [record(f"S2-{i}", "Same", "25 Road", "Y") for i in range(50)])
        f = pair_features(a, b, frequencies)
        self.assertEqual(f["country_equal"], 0)
        self.assertLessEqual(f["name_rare_token_dice"], 1)
        self.assertLessEqual(f["address_rare_token_dice"], 1)
        self.assertEqual((f["candidate_is_s2"], f["candidate_is_s3"]), (0, 1))
        self.assertEqual(self.features(a, record("S2-1", "", country="x"))["country_equal"], 1)
        self.assertEqual(self.features(record("S1-1", "", country=""), record("S2-1", "", country=""))["country_equal"], 0)

    def test_determinism_and_cached_features(self):
        from src.candidate_improvements import prepare_weighted
        a, b = record("S1-1", "Alpha", "25 Main"), record("S2-1", "Alpha Ltd", "25 Main")
        freq = Frequencies.fit([a, b])
        expected = pair_features(a, b, freq, .8, 2)
        self.assertEqual(expected, pair_features(a, b, freq, .8, 2, prepare_weighted(a, freq), prepare_weighted(b, freq)))
        self.assertEqual(expected, pair_features(a, b, freq, .8, 2))
        self.assertEqual((expected["candidate_rank"], expected["ranking_available"]), (2, 1))
        with self.assertRaises(ValueError):
            pair_features(a, b, freq, .8, None)

    def test_split_is_entity_disjoint_and_input_order_independent(self):
        ids = [f"S1-{i}" for i in range(100)]
        splits = split_s1_entities(ids)
        self.assertEqual(splits, split_s1_entities(list(reversed(ids))))
        self.assertEqual(list(splits.values()).count("train"), 80)
        self.assertEqual(list(splits.values()).count("validation"), 20)
        self.assertNotEqual(splits, split_s1_entities(ids, seed=43))
        with self.assertRaises(ValueError):
            split_s1_entities(["S1-1", "S1-1"])

    def test_exact_feature_statistics(self):
        stats = FeatureStatistics()
        for value in (0., .25, .5, .75, 1.):
            stats.add("train", 1, {f: value for f in FEATURE_NAMES})
        summary = stats.summarize()["train"]["1"]["name_exact"]
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["mean"], .5)
        self.assertEqual(summary["median"], .5)
        self.assertAlmostEqual(summary["p90"], .9)

    def test_compressed_pipeline_split_labels_and_frequency_roundtrip(self):
        queries = [record("S1-1", "Alpha", "25 Main"), record("S1-2", "Beta", "25 Main")]
        pool = [record("S2-1", "Alpha", "25 Main"), record("S3-1", "Alpha", "25 Main"),
                record("S2-2", "Beta", "25 Main")]
        truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": {"S2-2"}}
        assignments = {"S1-1": "train", "S1-2": "validation"}
        frequencies = Frequencies.fit(pool[:2])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_frequencies(frequencies, output / "frequencies.gz")
            self.assertEqual(frequencies, load_frequencies(output / "frequencies.gz"))
            balances, summaries, _ = build_pair_files(queries, pool, truth, assignments, frequencies, output, k=3)
            self.assertEqual(balances["train"]["positive_pairs"], 2)
            self.assertEqual(balances["validation"]["positive_pairs"], 1)
            self.assertEqual(balances["train"]["negative_pairs"], 1)
            self.assertEqual(summaries["validation"]["0"]["candidate_rank"]["count"], 2)
            for split, entity_id in (("train", "S1-1"), ("validation", "S1-2")):
                with gzip.open(output / f"{split}_features.csv.gz", "rt", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual({r["source1_entity_id"] for r in rows}, {entity_id})
                self.assertEqual(len({r["candidate_entity_id"] for r in rows}), 3)
                self.assertFalse((output / f"{split}_features.csv.gz.part").exists())


if __name__ == "__main__":
    unittest.main()
