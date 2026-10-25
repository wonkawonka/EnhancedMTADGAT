"""CPU mechanism checks and a complete synthetic-cache A/B integration test."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.models.rcr import LinearResidualCalibration
from src.models.rcr_memory import PrototypeResidualCalibration, residual_moments


class MemoryTests(unittest.TestCase):
    def test_k1_matches_global_current_estimator(self):
        rng = np.random.default_rng(3407)
        q = rng.normal(size=(120, 8))
        residual = rng.normal(size=(120, 5, 7))*.2+.4
        global_cal = LinearResidualCalibration(.05).fit(np.zeros_like(q), residual)
        with threadpool_limits(limits=2):
            prototype = PrototypeResidualCalibration(1).fit(q, residual)
        np.testing.assert_allclose(prototype.transform(q, residual),
                                   global_cal.transform(np.zeros_like(q), residual), atol=1e-13)
        means, variances = residual_moments(residual, batch=7)
        np.testing.assert_allclose(means, residual.mean(1))
        np.testing.assert_allclose(variances, residual.var(1))

    def test_multistate_recovery_and_no_response_routing(self):
        rng = np.random.default_rng(42)
        state = np.repeat([-1, 1], 90)
        q = state[:, None]*3+rng.normal(0, .1, size=(180, 8))
        residual = state[:, None, None]*2+rng.normal(0, .1, size=(180, 8, 7))
        with threadpool_limits(limits=2):
            single = PrototypeResidualCalibration(1).fit(q, residual)
            multiple = PrototypeResidualCalibration(2).fit(q, residual)
        self.assertLess(np.square(residual-multiple.parameters(q)[0][:, None, :]).mean(),
                        .05*np.square(residual-single.parameters(q)[0][:, None, :]).mean())
        old_parameters = [v.copy() for v in multiple.parameters(q)]
        old_query = multiple.query(q)[0].copy()
        fault_score = np.square(multiple.transform(q, residual+10)).mean()
        self.assertGreater(fault_score, 1000)
        for before, after in zip(old_parameters, multiple.parameters(q)):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(old_query, multiple.query(q)[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'calibration.npz'
            multiple.save(path)
            restored = PrototypeResidualCalibration.load(path)
            np.testing.assert_array_equal(restored.transform(q, residual), multiple.transform(q, residual))

    def test_degenerate_controls_reject_empty_prototypes(self):
        import warnings
        with warnings.catch_warnings(), threadpool_limits(limits=2):
            warnings.simplefilter('ignore')
            with self.assertRaisesRegex(ValueError, 'Empty normal-state'):
                PrototypeResidualCalibration(2).fit(np.zeros((10, 8)), np.zeros((10, 4, 7)))

    def test_complete_probe_and_partition_boundaries(self):
        rng = np.random.default_rng(3407)
        with tempfile.TemporaryDirectory(prefix='rcr-memory-test-') as temporary:
            root = Path(temporary)/'source'
            output = Path(temporary)/'output'
            root.mkdir()
            (root/'protocol.json').write_text(json.dumps(dict(smoke=False, window=4, stride=1)))
            (root/'backbone_complete.json').write_text(json.dumps(dict(checkpoint_hash='frozen-synthetic')))
            mapping = dict(control_indices=[1, 2], feature_columns=['x', 'SUM_CURRENT', 'SOC', 'a', 'b', 'c', 'd'],
                           sample_ids={}, windows={}, train_vins=['train'], validation_vins=['validation'],
                           test_normal_vins=['test'])
            rows, train = [], None
            for partition, samples in [('train', 8), ('validation', 2), ('test', 4)]:
                ids = [f'{partition}_{i}' for i in range(samples)]
                mapping['sample_ids'][partition] = ids
                mapping['windows'][partition] = samples*24
                q = rng.normal(size=(samples*24, 8))
                residual = rng.normal(0, .1, size=(samples*24, 4, 7))+q[:, 0, None, None]*.2
                if partition == 'test':
                    residual[48:] += 2
                if partition == 'train':
                    train = q, residual
                directory = root/'residuals'/partition
                directory.mkdir(parents=True)
                np.save(directory/'condition.npy', q)
                np.save(directory/'residual.npy', residual)
                np.save(directory/'references.npy', np.column_stack((np.repeat(np.arange(samples), 24),
                                                                      np.tile(np.arange(3, 27), samples))))
                (directory/'complete.json').write_text(json.dumps(dict(backbone_hash='frozen-synthetic',
                    windows=samples*24, sample_ids=ids)))
                for i, sample in enumerate(ids):
                    label = int(partition == 'test' and i >= 2)
                    rows.append(dict(sample_id=sample, partition=partition, sample_label=label,
                        vin=partition, cycle_kind='charge' if i % 2 == 0 else 'discharge',
                        fault_type='fault' if label else 'normal'))
            (root/'mapping.json').write_text(json.dumps(mapping))
            pd.DataFrame(rows).to_csv(root/'metadata.csv', index=False)
            (root/'lite_continuous').mkdir()
            LinearResidualCalibration().fit(*train).save(root/'lite_continuous/calibration.npz')
            command = [sys.executable, '-m', 'src.runners.run_rcr_memory_probe', '--root', str(root),
                       '--output', str(output), '--n-init', '2', '--stability-seeds', '3408', '--threads', '2']
            done = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout+'\n'+done.stderr)
            self.assertTrue((output/'complete.json').exists())
            summary = pd.read_csv(output/'summary_metrics.csv')
            self.assertEqual(len(summary), 6*3*2)
            for k in [1, 2, 4, 6, 8]:
                unit = output/f'B_k{k}'
                with np.load(unit/'test_transitions.npz') as transitions:
                    self.assertEqual(transitions['all'].sum(), 4*23)
                    self.assertEqual(transitions['normal'].sum(), 2*23)
                    self.assertEqual(transitions['fault'].sum(), 2*23)
                labels = np.load(unit/'test_assignments.npy')
                refs = np.load(output/'test_references.npy')
                self.assertEqual(len(labels), len(refs))
                samples = pd.read_csv(unit/'test_sample_prototypes.csv')
                self.assertEqual(samples.adjacent_pairs.sum(), 4*23)
            a_scores = np.load(output/'A_current_rcr/test_window_scores.npy')
            b_scores = np.load(output/'B_k6/test_window_scores.npy')
            np.testing.assert_array_equal(a_scores[:, 0], b_scores[:, 0])
            a = LinearResidualCalibration.load(root/'lite_continuous/calibration.npz')
            r = np.load(root/'residuals/test/residual.npy')
            q = np.load(root/'residuals/test/condition.npy')
            np.testing.assert_allclose(a_scores[:, 1], np.square(a.transform(q, r)).mean((1, 2)))
            rerun = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            changed = subprocess.run(command+['--perturb-std', '.02'], capture_output=True, text=True)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn('Source or protocol changed', changed.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
