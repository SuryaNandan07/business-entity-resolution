import unittest

from src.evaluate import entity_f05, macro_f05


class EvaluationTests(unittest.TestCase):
    def test_perfect_singleton(self):
        self.assertEqual(entity_f05(['a'], ['a']), 1)

    def test_false_singleton(self):
        self.assertEqual(entity_f05(['a'], ['b']), 0)

    def test_one_correct_of_multiple(self):
        self.assertAlmostEqual(entity_f05(['a', 'b'], ['a']), 5 / 6)

    def test_multiple_correct(self):
        self.assertEqual(entity_f05(['a', 'b'], ['a', 'b']), 1)

    def test_partial_prediction(self):
        self.assertAlmostEqual(entity_f05(['a', 'b', 'c'], ['a', 'b']), 2.5 / 2.75)

    def test_extra_false_matches(self):
        self.assertAlmostEqual(entity_f05(['a'], ['a', 'b']), 1.25 / 2.25)

    def test_empty_prediction(self):
        self.assertEqual(entity_f05(['a'], []), 0)

    def test_empty_truth(self):
        self.assertEqual(entity_f05([], []), 1)
        self.assertEqual(entity_f05([], ['a']), 0)

    def test_duplicates(self):
        self.assertEqual(entity_f05(['a', 'a'], ['a', 'a']), 1)

    def test_macro_not_pair_average(self):
        self.assertEqual(macro_f05({'s1': ['a'], 's2': ['b', 'c']}, {'s1': ['a']}), .5)

    def test_invalid_population(self):
        with self.assertRaises(ValueError):
            macro_f05({}, {})
        with self.assertRaises(ValueError):
            macro_f05({'a': []}, {'b': []})
