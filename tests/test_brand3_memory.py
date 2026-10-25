"""CPU tests for Brand3 vehicle scoring, leakage guards and full A/B replay."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.models.rcr import LinearResidualCalibration
from src.runners.run_brand3_memory_probe import vehicle_scores, verify_source


class Brand3MemoryTests(unittest.TestCase):
    def test_vehicle_top_five_percent_and_id_join(self):
        ids = [f's{i}' for i in range(50)]
        mapping = dict(sample_ids=dict(test=ids), cars=dict(test=['a']*25+['b']*25),
                       labels=dict(test=[0]*25+[1]*25))
        scores = [0.]*23+[1., 9.]+[2.]*25
        frame = pd.DataFrame(dict(sample_id=ids, raw=scores, magnitude=scores, magnitude_cdf=scores))
        _, vehicles = vehicle_scores(frame.sample(frac=1, random_state=1), mapping, 'test')
        np.testing.assert_allclose(vehicles.magnitude, [5., 2.])
        np.testing.assert_array_equal(vehicles.label, [0, 1])
        with self.assertRaisesRegex(ValueError, 'Incomplete snippet'):
            vehicle_scores(frame.iloc[:-1], mapping, 'test')

    def test_full_synthetic_fold_and_unavailable_temporal_switching(self):
        rng = np.random.default_rng(3407)
        with tempfile.TemporaryDirectory(prefix='brand3-memory-') as temporary:
            base = Path(temporary)
            source = base/'sources/fold0/full'
            source.mkdir(parents=True)
            (source/'protocol.json').write_text(json.dumps(dict(dataset='Brand3', fold=0, smoke=False,
                                                                window=127, stride=127)))
            (source/'backbone_complete.json').write_text(json.dumps(dict(checkpoint_hash='synthetic-frozen')))
            (source/'memory_source_complete.json').write_text(json.dumps(dict(full_data=True)))
            mapping = dict(feature_columns=['pack_voltage', 'current', 'soc', 'max_cell_voltage',
                'min_cell_voltage', 'max_temperature', 'min_temperature'], control_indices=[1, 2],
                sample_ids={}, cars={}, labels={}, windows={})
            metadata = []
            for partition, n_cars in [('train', 3), ('validation', 2), ('test', 4)]:
                ids = [f'{partition}_{i}' for i in range(n_cars*25)]
                cars = [f'{partition}_car{i//25}' for i in range(n_cars*25)]
                labels = [int(partition == 'test' and i >= 50) for i in range(n_cars*25)]
                mapping['sample_ids'][partition] = ids
                mapping['cars'][partition] = cars
                mapping['labels'][partition] = labels
                mapping['windows'][partition] = len(ids)
                q = rng.normal(size=(len(ids), 8))
                residual = rng.normal(0, .03, size=(len(ids), 127, 7))+q[:, 0, None, None]*.02
                residual += np.asarray(labels)[:, None, None]*.5
                p = source/'residuals'/partition
                p.mkdir(parents=True)
                np.save(p/'residual.npy', residual)
                np.save(p/'condition.npy', q)
                np.save(p/'references.npy', np.column_stack((np.arange(len(ids)), np.full(len(ids), 126))))
                (p/'complete.json').write_text(json.dumps(dict(backbone_hash='synthetic-frozen',
                                                               windows=len(ids), sample_ids=ids)))
                metadata.extend(dict(sample_id=sample, car=car, sample_label=label, partition=partition)
                                for sample, car, label in zip(ids, cars, labels))
                if partition == 'train':
                    (source/'lite_continuous').mkdir()
                    LinearResidualCalibration().fit(q, residual).save(source/'lite_continuous/calibration.npz')
            (source/'mapping.json').write_text(json.dumps(mapping))
            pd.DataFrame(metadata).to_csv(source/'metadata.csv', index=False)
            command = [sys.executable, '-m', 'src.runners.run_brand3_memory_probe', '--root', str(base),
                       '--folds', '0', '--n-init', '2', '--stability-seeds', '3408', '--threads', '2']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            p = base/'experiments/fold0'
            self.assertTrue(json.loads((p/'artifact_audit.json').read_text())['complete'])
            summary = pd.read_csv(p/'summary_metrics.csv')
            self.assertEqual(len(summary), 36)
            for k in [1, 2, 4, 6, 8]:
                unit = p/f'B_k{k}'
                d = json.loads((unit/'diagnostics.json').read_text())
                self.assertFalse(d['temporal_switching_available'])
                self.assertIsNone(d['partitions']['test']['temporal_switch_rate'])
                vehicle = pd.read_csv(unit/'test_vehicle_prototypes.csv')
                self.assertTrue(vehicle.temporal_switch_rate.isna().all())
                self.assertEqual(vehicle.snippets.sum(), 100)
                self.assertEqual(len(pd.read_csv(unit/'test_vehicle_scores.csv')), 4)
            # A changed split must fail before any scoring/fitting.
            mapping['cars']['test'][0] = mapping['cars']['train'][0]
            (source/'mapping.json').write_text(json.dumps(mapping))
            with self.assertRaisesRegex(ValueError, 'Vehicle or snippet leakage'):
                verify_source(source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
