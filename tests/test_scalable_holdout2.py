"""Regression checks for the frozen evaluation and resource guard."""
import unittest
from unittest.mock import patch, mock_open
from src.holdout2 import prior
from src.resource_guard import check, reset_guard_state
from src.scalable_holdout2 import summarize

class FrozenHoldoutTests(unittest.TestCase):
    def setUp(self):
        reset_guard_state()

    def test_prior_includes_holdout_one(self):
        rows=[{'entity_id':str(i)} for i in range(80000)]
        with patch('src.holdout2.read_rows',return_value=iter(rows)),patch('pathlib.Path.open',mock_open(read_data='source1_entity_id\nextra\n')):
            ids=prior()
        self.assertIn('59999',ids)
        self.assertIn('extra',ids)
        self.assertNotIn('60000',ids)

    def test_candidate_and_full_recall_differ(self):
        truth={'a':{'S2-1','S3-2'},'b':{'S3-3'}}
        candidates={'a':{'S2-1','S2-x'},'b':{'S3-3'}}
        predictions={'a':{'S2-1'},'b':set()}
        m=summarize(truth,predictions,candidates)
        self.assertEqual(m['candidate_recall'],2/3)
        self.assertEqual(m['pair_recall'],.5)
        self.assertEqual(m['full_ground_truth_link_recall'],1/3)
        self.assertEqual(m['pair_precision'],1)
        self.assertEqual(m['incorrect_singletons'],1)
        self.assertEqual(m['zero_prediction_s1_count'],1)

    def test_python_process_ram_over_limit_stops_immediately(self):
        safe = dict(rss=100, private=100, available=3*2**30)
        with patch('src.resource_guard.memory', return_value=safe):
            self.assertEqual(check(), safe)
        for field in ('rss', 'private'):
            reset_guard_state()
            with patch('src.resource_guard.memory', return_value={**safe, field: int(6 * 2**30)}):
                with self.assertRaises(MemoryError):
                    check()

    def test_transient_system_memory_dip_does_not_stop(self):
        safe = dict(rss=100, private=100, available=3*2**30)
        low = dict(rss=100, private=100, available=100*2**20)  # 100 MiB < 512 MiB
        with patch('src.resource_guard.memory', return_value=low):
            # Single observation should pass without raising MemoryError
            res = check()
            self.assertEqual(res, low)

    def test_sustained_critically_low_system_memory_stops(self):
        low = dict(rss=100, private=100, available=100*2**20)  # 100 MiB < 512 MiB
        with patch('src.resource_guard.memory', return_value=low):
            # 4 low observations should pass
            for _ in range(4):
                check()
            # 5th consecutive observation should raise MemoryError
            with self.assertRaises(MemoryError):
                check()

    def test_normal_memory_resets_low_observation_counter(self):
        safe = dict(rss=100, private=100, available=3*2**30)
        low = dict(rss=100, private=100, available=100*2**20)
        
        # 3 low observations
        with patch('src.resource_guard.memory', return_value=low):
            for _ in range(3):
                check()
        # 1 normal observation (resets count)
        with patch('src.resource_guard.memory', return_value=safe):
            check()
        # 3 more low observations (should not trigger stop since count was reset)
        with patch('src.resource_guard.memory', return_value=low):
            for _ in range(3):
                check()

if __name__=='__main__':
    unittest.main()
