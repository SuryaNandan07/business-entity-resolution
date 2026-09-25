"""Small synthetic tests; never load hackathon datasets."""

import unittest

from src.candidates import CONFIGURATIONS, Record, build_index
from src.ranking import (FORMULAS, SIGNAL_NAMES, evaluate_rankings, example_sets, jaccard, pair_signals,
                         prepare, rank_candidates, rank_summary, ranking_score, top_k)


def record(entity_id, name, address="", country="X"):
    return Record.from_row({"entity_id": entity_id, "business_name": name,
                           "business_address": address, "country": country})


class RankingTests(unittest.TestCase):
    def test_report_signal_metadata(self):
        self.assertEqual(SIGNAL_NAMES, ("name_exact", "name_jaccard", "name_bigram_dice",
                                       "name_prefix", "address_jaccard", "shared_number"))

    def test_scoring_formula_and_bounds(self):
        item = prepare(record("S1-1", "Alpha Clinic", "25 Main Street"))
        signals = pair_signals(item, item)
        self.assertEqual(signals[:6], (1, 1, 1, 1, 1, 1))
        for formula, weights in FORMULAS.items():
            self.assertAlmostEqual(sum(weights), 1)
            self.assertAlmostEqual(ranking_score(signals, formula), 1)
        partial = pair_signals(item, prepare(record("S2-1", "Alpha Dental", "25 Other")))
        self.assertAlmostEqual(partial.name_jaccard, 1 / 3)
        self.assertAlmostEqual(partial.address_jaccard, 1 / 4)
        self.assertEqual(partial.shared_number, 1)
        self.assertAlmostEqual(ranking_score(partial, "balanced"),
                               .25 / 3 + .25 * partial.name_bigram_dice + .05 + .30 / 4 + .05)

    def test_name_evidence_beats_shared_number(self):
        query = prepare(record("S1-1", "Alpha Clinic", "25 Main Road"))
        name = prepare(record("S2-1", "Alpha Clinic"))
        number = prepare(record("S2-2", "Unrelated Hardware", "25 Other Street"))
        for formula in FORMULAS:
            self.assertGreater(ranking_score(pair_signals(query, name), formula),
                               ranking_score(pair_signals(query, number), formula))

    def test_ranking_order_duplicates_and_ties(self):
        query = prepare(record("S1-1", "Alpha Clinic", "25 Main Road"))
        pool = {r.entity_id: prepare(r) for r in [
            record("S2-2", "Alpha Clinic", "25 Main Road"),
            record("S2-1", "Alpha Clinic", "25 Main Road"),
            record("S3-1", "Alpha Other", "25 Main Road")]}
        ranked = rank_candidates(query, ["S3-1", "S2-2", "S2-1", "S2-2"], pool)
        self.assertEqual([e for e, _ in ranked], ["S2-1", "S2-2", "S3-1"])
        self.assertEqual(rank_candidates(query, reversed(list(pool)), pool), ranked)

    def test_top_k_and_multiple_matches(self):
        ranked = [("S2-1", .9), ("S3-1", .8), ("S2-2", .7)]
        self.assertEqual(top_k(ranked, 2), ranked[:2])
        self.assertEqual(top_k(ranked, 20), ranked)
        self.assertEqual(top_k([], 20), [])
        with self.assertRaises(ValueError):
            top_k(ranked, 0)

    def test_missing_addresses_are_not_matches(self):
        query = prepare(record("S1-1", "École", None))
        other = prepare(record("S2-1", "E\u0301cole", ""))
        signals = pair_signals(query, other)
        self.assertTrue(signals.missing_address)
        self.assertEqual(signals.address_jaccard, 0)
        self.assertEqual(signals.shared_number, 0)
        self.assertEqual(signals.name_exact, 1)
        self.assertAlmostEqual(ranking_score(signals), .65)
        empty = prepare(record("S3-1", "", None))
        self.assertEqual(ranking_score(pair_signals(empty, empty)), 0)
        self.assertEqual(jaccard(frozenset(), frozenset()), 0)

    def test_country_gate_and_non_latin_names(self):
        query = prepare(record("S1-1", "राम दुकान"))
        pool = {r.entity_id: prepare(r) for r in [record("S2-1", "राम दुकान"),
                record("S3-1", "राम दुकान", country="Y"),
                record("S3-2", "राम दुकान", country="")]}
        ranked = rank_candidates(query, pool, pool)
        self.assertEqual([e for e, _ in ranked], ["S2-1"])

    def test_link_recall_distinguishes_blocking_and_pruning_misses(self):
        queries = [record("S1-1", "Alpha Clinic", "25 Main"), record("S1-2", "Zebra")]
        pool = [record(f"S2-{i:03}", "Alpha Clinic", "25 Main") for i in range(25)]
        pool += [record("S3-001", "Alpha Other", "25 Main")]
        truth = {"S1-1": {"S2-000", "S2-024", "S3-001", "S3-absent"}, "S1-2": set()}
        baseline, results, _ = evaluate_rankings(queries, truth,
            build_index(pool, CONFIGURATIONS["C"]), {r.entity_id: prepare(r) for r in pool})
        self.assertEqual(baseline["overall_recall"], .75)
        for result in results.values():
            twenty = result["top_k"][20]
            thirty = result["top_k"][30]
            self.assertEqual(twenty["overall_recall"], .25)
            self.assertEqual(twenty["lost_due_to_pruning"], 2)
            self.assertEqual(twenty["total_missed_links"], 3)
            self.assertEqual(twenty["zero_candidate_percent"], 50)
            self.assertEqual(twenty["average_candidates"], 10)
            self.assertEqual(thirty["overall_recall"], .75)
            self.assertEqual(thirty["found_links"], {"S2": 2, "S3": 1})
            self.assertEqual(result["true_match_ranks"]["overall"]["count"], 3)
            self.assertEqual(result["unranked_true_links_absent_from_broad_pool"], 1)

    def test_rank_distribution(self):
        summary = rank_summary(list(range(1, 101)))
        self.assertEqual(summary, {"count": 100, "median": 50.5, "p90": 90,
                                   "p95": 95, "p99": 99, "maximum": 100})
        self.assertIsNone(rank_summary([])["median"])

    def test_example_labels_ranks_and_pruning_reasons(self):
        queries = [record("S1-1", "Alpha"), record("S1-2", "Alpha")]
        pool = [record(f"S2-{i:03}", "Alpha") for i in range(25)]
        truth = {"S1-1": {"S2-000"}, "S1-2": {"S2-024"}}
        examples = example_sets(queries, {r.entity_id: r for r in pool}, truth,
                               build_index(pool, CONFIGURATIONS["C"]),
                               {r.entity_id: prepare(r) for r in pool}, "balanced", 20)
        self.assertEqual(len(examples["successes"]), 1)
        self.assertEqual(len(examples["failures"]), 1)
        success = examples["successes"][0]
        failure = examples["failures"][0]
        self.assertEqual(success["broad_count"], 25)
        self.assertEqual(success["kept_count"], 20)
        self.assertEqual(success["pruning_misses"], [])
        self.assertEqual(failure["pruning_misses"], ["S2-024"])
        self.assertEqual(failure["true_candidates"][0]["rank"], 25)
        self.assertFalse(failure["true_candidates"][0]["kept"])
        self.assertEqual(failure["cutoff_candidate"]["rank"], 20)
        self.assertEqual(failure["cutoff_candidate"]["label"], "FALSE CANDIDATE")


if __name__ == "__main__":
    unittest.main()
