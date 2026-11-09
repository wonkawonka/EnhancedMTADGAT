import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data.nc_battery import SnippetRecord
from src.data.external_protocol import load_external_protocol_data
from src.runners.run_brand1_external_comparison import expand, build_args
from src.runners.run_unified_external_baselines import _window_and_stride


class Brand1ComparisonTests(unittest.TestCase):
    def test_plan_jobs_and_native_windows(self):
        plan = json.loads(Path('configs/external/brand1_comparison_20261108.json').read_text())
        jobs = expand(plan)
        self.assertEqual(len(jobs), 8 * 5 * len(plan['seeds']))
        for job in jobs:
            args = build_args(plan, job, Path('/tmp/brand1_test_unused'))
            self.assertEqual(args.dataset, 'Brand1')
            self.assertEqual(args.seed, job['seed'])
            self.assertEqual(_window_and_stride(args),
                             (plan['native_windows'][job['method']], plan['native_strides'][job['method']]))
            self.assertEqual(args.train_sample_limit, 0)
            self.assertEqual(args.threshold_quantile, .95)

    def test_brand1_loader_preserves_vehicle_isolation_and_normal_only_scaler(self):
        records = [SnippetRecord(path=f'/fake/train/{car}.pkl', car=str(car), label=int(car >= 10), length=128)
                   for car in range(12)]
        def load(path):
            car = int(Path(path).stem)
            # Faults are far beyond the training range and must not set its extrema.
            return np.full((128, 7), car + (1000 if car >= 10 else 0), dtype='float32'), {}
        with patch('src.data.external_protocol.build_index', return_value=records), \
             patch('src.data.external_protocol.load_snippet', side_effect=load), \
             patch('src.data.nc_battery.load_snippet', side_effect=load):
            data = load_external_protocol_data('Brand1', brand_fold=0, brand_fold_seed=0,
                    brand_split_protocol='strict_normal_validation', brand_normalization='minmax')
        ids = data.metadata['sample_ids']
        cars = {p: {s.split(':')[0] for s in selected} for p, selected in ids.items()}
        self.assertFalse(cars['train'] & cars['validation'])
        self.assertFalse(cars['train'] & cars['test'])
        self.assertFalse(cars['validation'] & cars['test'])
        self.assertTrue(all(int(label[0]) == 0 for label in data.validation_labels))
        self.assertEqual(sum(int(y[0]) for y in data.test_labels), 2)
        self.assertLess(max(data.metadata['normalization']['data_max']), 1000)
        self.assertEqual(data.evaluation_kind, 'vehicle_ranking')


if __name__ == '__main__':
    unittest.main()
