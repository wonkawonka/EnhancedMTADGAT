"""Protocol checks for vehicle-level Brand2 extension; no GPU or training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.runners.run_brand_rcr_matrix import (
    evaluate, threshold_metrics, validate_plan, vehicle_aggregate, verify_mapping,
)
from src.runners.run_unified_external_baselines import _window_and_stride
from src.data.external_protocol import load_external_protocol_data


class BrandRCRProtocolTests(unittest.TestCase):
    def test_brand2_and_brand3_scalers_use_training_only(self):
        records = [SimpleNamespace(car=car, label=label, path=f'/tmp/{car}/x.pkl', length=128)
                   for car, label in [('a', 0), ('b', 0), ('c', 0), ('d', 1)]]
        arrays = {record.path: np.full((128, 7), i * 2., dtype=np.float32)
                  for i, record in enumerate(records)}
        arrays[records[0].path] = np.repeat(np.linspace(0., 1., 128)[:, None], 7, axis=1).astype(np.float32)
        split = dict(train=records[:1], validation=records[1:2], calibration=records[1:2], test=records[2:])
        load = lambda path: (arrays[str(path)], {})
        for brand in (2, 3):
            with patch('src.data.external_protocol.build_index', return_value=records) as index, \
                 patch('src.data.external_protocol.split_vehicle_folds', return_value=split), \
                 patch('src.data.external_protocol.resolve_dataset_root', return_value=Path('/tmp/mock-brand')), \
                 patch('src.data.external_protocol.load_snippet', side_effect=load), \
                 patch('src.data.nc_battery.load_snippet', side_effect=load):
                data = load_external_protocol_data(f'Brand{brand}', brand_split_protocol='strict_normal_validation', brand_normalization='minmax')
                self.assertEqual(index.call_args.args[1], brand)
                np.testing.assert_array_equal(data.metadata['normalization']['data_max'], [1.] * 7)
                self.assertTrue((data.validation_sequences[0] > 1.).all())
                self.assertEqual(data.metadata['brand'], brand)
                self.assertEqual(data.metadata['sample_ids']['test'], ['c:c/x.pkl', 'd:d/x.pkl'])

    def test_four_shards_cover_all_75_cells_once(self):
        plan = json.loads(Path('configs/internal/brand2_rcr_reference_fivefold_seed3407_20261024.json').read_text())
        validate_plan(plan)
        duplicate = copy.deepcopy(plan)
        duplicate['shards']['daisy_a']['jobs'].append(duplicate['shards']['daisy_b']['jobs'][0])
        with self.assertRaises(AssertionError):
            validate_plan(duplicate)

    def test_vehicle_top_five_percent_does_not_average_all_snippets(self):
        snippets = pd.DataFrame(dict(car=['normal'] * 20 + ['fault'] * 20,
            label=[0] * 20 + [1] * 20, score=[0.] * 19 + [2.] + [1.] * 20))
        vehicles = vehicle_aggregate(snippets).set_index('car')
        self.assertEqual(vehicles.loc['normal', 'score'], 2.)
        self.assertEqual(vehicles.loc['fault', 'score'], 1.)

    def test_thresholds_do_not_depend_on_test_fault_scores(self):
        validation = pd.DataFrame(dict(label=[0, 0, 0], score=[.1, .2, .3]))
        test = pd.DataFrame(dict(label=[0, 0, 1], score=[.1, .5, .8]))
        before = threshold_metrics(validation, test)
        changed = test.copy(); changed.loc[2, 'score'] = 800.
        after = threshold_metrics(validation, changed)
        self.assertEqual([r['threshold'] for r in before], [r['threshold'] for r in after])
        self.assertEqual((before[0]['tp'], before[0]['fp'], before[0]['fn'], before[0]['tn']), (1, 1, 0, 1))
        validation.loc[0, 'label'] = 1
        with self.assertRaises(AssertionError):
            threshold_metrics(validation, test)

    def mapping(self):
        return dict(brand=2, fold=0, smoke=True, feature_columns=list('abcdefg'),
            sample_ids=dict(train=['train0'], validation=['v0', 'v1'], test=['t0', 't1']),
            cars=dict(train=['a'], validation=['b', 'c'], test=['d', 'e']),
            labels=dict(train=[0], validation=[0, 0], test=[0, 1]),
            vehicles=dict(train=['a'], validation=['b', 'c'], test=['d', 'e']))

    def test_vehicle_overlap_is_rejected(self):
        mapping = self.mapping(); verify_mapping(mapping)
        mapping['vehicles']['test'] = ['a', 'e']; mapping['cars']['test'] = ['a', 'e']
        with self.assertRaises(AssertionError):
            verify_mapping(mapping)

    def test_artifacts_preserve_score_mapping_and_confusion_counts(self):
        mapping = self.mapping()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            evaluate(output, mapping, dict(validation=np.array([.1, .2]), test=np.array([.15, .8])),
                     127, 'raw', {'source': 'synthetic protocol test'})
            with np.load(output / 'test_windows.npz') as saved:
                self.assertEqual(saved['sample_ids'].tolist(), ['t0', 't1'])
                self.assertEqual(saved['end_indices'].tolist(), [126, 126])
            metrics = pd.read_csv(output / 'metrics.csv')
            self.assertTrue((metrics.tp + metrics.fp + metrics.fn + metrics.tn == 2).all())
            self.assertTrue(json.loads((output / 'complete.json').read_text())['smoke'])

    def test_brand2_preserves_brand3_native_deep_svdd_window(self):
        for brand in ['Brand2', 'Brand3']:
            args = SimpleNamespace(dataset=brand, method='deep_svdd', lookback=127, window_stride=128)
            self.assertEqual(_window_and_stride(args), (128, 128))


if __name__ == '__main__':
    unittest.main()
