import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluate import macro_f05
from src.modeling import load_pairs, threshold_metrics, training_columns, tune_threshold


class ModelingTests(unittest.TestCase):
    def test_macro_matches_sets_with_missing_links_and_empty_entities(self):
        truth = {'a': {'x', 'y', 'missing'}, 'b': set(), 'c': {'z'}, 'd': set()}
        labels = np.array([1, 1, 0, 0])
        probabilities = np.array([.8, .7, .9, .1])
        result = threshold_metrics(probabilities, labels, np.array([0, 0, 1, 2]), np.array([3, 0, 1, 0]), .5)
        self.assertAlmostEqual(result['macro_f05'], macro_f05(truth, {'a': {'x', 'y'}, 'b': {'wrong'}}))
        self.assertEqual(result['predicted_links'], 3)
        self.assertEqual(result['pair_recall'], 1)
        self.assertEqual(result['full_truth_link_recall'], .5)

    def test_threshold_boundary_and_multiple_matches(self):
        result = threshold_metrics(np.array([.5, .5]), np.ones(2), np.zeros(2, dtype=int), np.array([2]), .5)
        self.assertEqual(result['macro_f05'], 1)
        self.assertEqual(result['average_links'], 2)

    def test_deterministic_tuning(self):
        args = (np.array([.9, .8, .2]), np.array([1, 1, 0]), np.array([0, 0, 0]), np.array([2]))
        self.assertEqual(tune_threshold(*args), tune_threshold(*args))
        self.assertEqual(tune_threshold(*args)[0]['macro_f05'], 1)

    def test_training_only_preprocessing(self):
        train = np.array([[0., 1., 0.], [2., 1., 1.]])
        columns = training_columns(train, ['a', 'constant', 'candidate_is_s3'])
        self.assertEqual(columns, [0])
        scaler = StandardScaler().fit(train[:, columns])
        scaler.transform(np.array([[100.]]))
        self.assertEqual(scaler.mean_[0], 1.)

    def test_duplicate_and_split_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pairs.csv.gz'
            frame = pd.DataFrame({'source1_entity_id': ['a', 'a'], 'candidate_entity_id': ['b', 'b'], 'label': [1, 1], 'feature': [1, 1]})
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_pairs(path, ['feature'], 2, {'a'})
            with self.assertRaisesRegex(ValueError, 'split'):
                load_pairs(path, ['feature'], 2, {'other'})
