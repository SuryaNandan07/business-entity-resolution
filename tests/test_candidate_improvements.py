"""Synthetic regression tests for Phase 2D; no training-file reads."""

import unittest

from src.candidates import Record, build_index, generate_candidates, CONFIGURATIONS
from src.candidate_improvements import (AdditionalIndex, Frequencies, char_ngrams,
                                        failure_categories, improved_score, improved_signals,
                                        prepare_weighted, script_profile, weighted_overlap)


def record(entity_id, name, address="", country="X"):
    return Record.from_row(dict(entity_id=entity_id, business_name=name,
                                business_address=address, country=country))


class ImprovementTests(unittest.TestCase):
    def test_idf_counts_documents_not_repeated_words(self):
        rows = [record("S2-1", "Rare Common Common"), record("S2-2", "Other Common")]
        f = Frequencies.fit(rows)
        self.assertEqual(f.name["x", "common"], 2)
        self.assertGreater(f.idf("x", "rare", "name"), f.idf("x", "common", "name"))

    def test_rare_tokens_recover_changed_first_word_and_reordering(self):
        query = record("S1-1", "Alpha Zephyr", "")
        pool = [record("S2-1", "Zephyr Alpha")]
        baseline = generate_candidates(query, build_index(pool, CONFIGURATIONS["C"]), CONFIGURATIONS["C"])
        self.assertEqual(baseline, set())
        tokens, _ = AdditionalIndex(pool, Frequencies.fit(pool)).candidates(query)
        self.assertEqual(tokens, {"S2-1"})

    def test_address_words_recover_script_mismatch_and_changed_number(self):
        query = record("S1-1", "Alpha Company", "25 Magnolia Cedar")
        pool = [record("S2-1", "राम दुकान", "30 Magnolia Cedar"),
                record("S2-2", "Other", "30 Magnolia Different"),
                record("S3-1", "राम दुकान", "30 Magnolia Cedar", "Y")]
        tokens, _ = AdditionalIndex(pool, Frequencies.fit(pool)).candidates(query)
        self.assertEqual(tokens, {"S2-1"})

    def test_ngram_index_recovers_spelling_with_no_shared_tokens(self):
        query = record("S1-1", "Zlphatech")
        pool = [record("S2-1", "Alphatech")]
        tokens, grams = AdditionalIndex(pool, Frequencies.fit(pool)).candidates(query)
        self.assertEqual(tokens, set())
        self.assertEqual(grams, {"S2-1"})
        self.assertTrue(char_ngrams("École"))

    def test_common_keys_are_omitted_not_truncated(self):
        pool = [record(f"S2-{i}", "Common", "Shared Road") for i in range(301)]
        index = AdditionalIndex(pool, Frequencies.fit(pool))
        tokens, grams = index.candidates(record("S1-1", "Common", "Shared Road"))
        self.assertEqual(tokens | grams, set())
        self.assertNotIn(("x", "common"), index.name)

    def test_union_keeps_original_and_multiple_new_candidates(self):
        query = record("S1-1", "Alpha Zephyr", "25 Main")
        pool = [record("S2-1", "Alpha Services"), record("S3-1", "Zephyr Alpha"),
                record("S3-2", "Zephyr New")]
        old = generate_candidates(query, build_index(pool, CONFIGURATIONS["C"]), CONFIGURATIONS["C"])
        tokens, grams = AdditionalIndex(pool, Frequencies.fit(pool)).candidates(query)
        expanded = old | tokens | grams
        self.assertTrue(old <= expanded)
        self.assertEqual(expanded, {"S2-1", "S3-1", "S3-2"})

    def test_rare_number_has_more_weight_than_common_number(self):
        pool = [record(f"S2-{i}", "Other", "1 Main") for i in range(200)]
        pool.append(record("S3-1", "Other", "98765 Main"))
        f = Frequencies.fit(pool)
        common = prepare_weighted(record("S1-1", "", "1"), f)
        rare = prepare_weighted(record("S1-2", "", "98765"), f)
        self.assertGreater(improved_signals(rare, rare, f)[2],
                           improved_signals(common, common, f)[2])

    def test_address_rescue_beats_misleading_suffix_and_number(self):
        query = record("S1-1", "Alpha Private Limited", "1 Cedar Magnolia")
        true = record("S2-1", "राम दुकान", "1 Cedar Magnolia")
        false = record("S2-2", "Other Private Limited", "1 Other Road")
        background = [record(f"S3-{i}", "Other Private Limited", "1 Other Road") for i in range(200)]
        f = Frequencies.fit([true, false] + background)
        q = prepare_weighted(query, f)
        a = improved_signals(q, prepare_weighted(true, f), f)
        b = improved_signals(q, prepare_weighted(false, f), f)
        self.assertGreater(improved_score(a), improved_score(b))
        self.assertGreater(improved_score(a), improved_score(a, "weighted"))

    def test_missing_address_country_gate_and_score_bounds(self):
        query = record("S1-1", "École", None)
        pool = [record("S2-1", "École"), record("S3-1", "École", country="Y")]
        f = Frequencies.fit(pool)
        q = prepare_weighted(query, f)
        signals = improved_signals(q, prepare_weighted(pool[0], f), f)
        self.assertEqual(signals, (1, 0, 0))
        self.assertAlmostEqual(improved_score(signals), .65)
        self.assertIsNone(improved_signals(q, prepare_weighted(pool[1], f), f))
        self.assertEqual(improved_score(None), float("-inf"))
        self.assertEqual(weighted_overlap({}, {}, 0, 0), (0, 0))
        self.assertLessEqual(improved_score((1, 1, 1)), 1)

    def test_script_categories_are_observational_and_overlap(self):
        a = record("S1-1", "Alpha", "25 Cedar")
        b = record("S2-1", "राम", "30 Cedar")
        flags = failure_categories(a, b, Frequencies.fit([a, b]))
        for category in ("different_dominant_scripts", "disjoint_letter_scripts",
                         "no_shared_name_tokens", "address_numbers_disjoint"):
            self.assertIn(category, flags)
        self.assertEqual(script_profile("École 25")['dominant'], "LATIN")
        self.assertEqual(script_profile("राम")['dominant'], "DEVANAGARI")
        self.assertIsNone(script_profile("123 !")['dominant'])


if __name__ == "__main__":
    unittest.main()
