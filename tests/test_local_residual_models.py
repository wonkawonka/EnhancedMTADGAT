"""Mechanism and CPU integration tests for the two normal-only models."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from src.models.rcr import LinearResidualCalibration
from src.models.local_residual_models import (fit_reliability, ReliableConditionalCalibration,
    augmented_second, gram_from_moments, ConditionalLOORelation)
from src.runners.run_local_residual_models import run, NAMES


class LocalModelTests(unittest.TestCase):
    def test_score_csv_roundtrip_preserves_adjacent_floats(self):
        from src.runners.analyze_local_residual_models import read_table
        value = .81269795470123
        scores = np.array([value, np.nextafter(value, 1.), 1/3, np.nextafter(1/3, 1.)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'scores.csv'
            pd.DataFrame(dict(score=scores)).to_csv(path, index=False)
            np.testing.assert_array_equal(read_table(path).score.to_numpy(), scores)

    def test_vehicle_crossfit_gate_bounds_and_serialization(self):
        rng = np.random.default_rng(3407)
        q = rng.normal(size=(180, 8))
        residual = .2*q[:, 0, None, None]+rng.normal(0, .02, (180, 6, 7))
        base = LinearResidualCalibration().fit(q, residual)
        means, variance = residual.mean(1), residual.var(1)
        vehicles, groups = np.repeat(np.arange(12), 15), np.tile(np.arange(3), 60)
        with threadpool_limits(limits=2):
            fitted = fit_reliability(q, means, vehicles, groups, base)
            cal = ReliableConditionalCalibration(base, fitted, 'reliable', '.001').fit_scale(q, means, variance, groups)
        for vehicle in np.unique(vehicles):
            self.assertEqual(len(np.unique(fitted['folds'][vehicles == vehicle])), 1)
        self.assertTrue(((cal.strength(q, groups) >= 0) & (cal.strength(q, groups) <= 1)).all())
        self.assertLess(cal.strength(q+100, groups).mean(), .01)
        self.assertTrue((cal.parameters(q, groups)[1] >= .001).all())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'cal.npz'
            cal.save(p)
            restored = ReliableConditionalCalibration.load(p)
            np.testing.assert_array_equal(restored.transform(q, residual, groups), cal.transform(q, residual, groups))

    def test_gram_matches_explicit_all_point_design(self):
        rng = np.random.default_rng(12)
        q, z = rng.normal(size=(19, 2)), rng.normal(size=(19, 5, 3))
        c = np.column_stack((np.ones(len(q)), q))
        values = np.concatenate((np.ones((19, 5, 1)), z), axis=2)
        design = np.einsum('na,nlj->nlaj', c, values).reshape(19*5, -1)
        streamed = gram_from_moments(q, augmented_second(z))
        np.testing.assert_allclose(streamed, design.T@design/5, rtol=1e-12, atol=1e-12)

    def test_conditional_joint_anomaly_and_no_target_leakage(self):
        rng = np.random.default_rng(42)
        def sample(n, fault=False):
            q = rng.choice([-1., 1.], size=(n, 1))
            a = rng.normal(size=(n, 7))
            sign = -1 if fault else 1
            z = np.stack((a, sign*q*a+rng.normal(0, .03, a.shape), rng.normal(size=a.shape)), axis=-1)
            return q, z
        tq, tz = sample(500)
        vq, vz = sample(300)
        tg = gram_from_moments(tq, augmented_second(tz))/len(tq)
        vg = gram_from_moments(vq, augmented_second(vz))/len(vq)
        stats = np.zeros(1), np.ones(1), np.array([-1.]), np.array([1.])
        with threadpool_limits(limits=2):
            model = ConditionalLOORelation.fit(tg, vg, 1, 3, stats, True)
        altered = vz.copy(); altered[:, :, 1] += 100
        np.testing.assert_allclose(model.predict(vq, vz)[:, :, 1], model.predict(vq, altered)[:, :, 1])
        fq, fz = sample(300, True)
        second = np.concatenate((augmented_second(vz), augmented_second(fz)))
        q = np.concatenate((vq, fq))
        y = np.repeat([0, 1], 300)
        relation = model.score_moments(q, second)
        diagonal = model.diagonal_score(second)
        self.assertGreater(roc_auc_score(y, relation), .95)
        self.assertLess(roc_auc_score(y, diagonal), .65)
        direct_error = np.square(vz-model.predict(vq, vz)).mean(1)/model.variances
        np.testing.assert_allclose(model.score_moments(vq, augmented_second(vz)), direct_error.mean(1), rtol=1e-9, atol=1e-9)

    def test_complete_two_model_pipeline_on_synthetic_cache(self):
        rng = np.random.default_rng(3407)
        with tempfile.TemporaryDirectory(prefix='local-residual-models-') as tmp:
            root, out = Path(tmp)/'source', Path(tmp)/'out'
            root.mkdir()
            (root/'protocol.json').write_text(json.dumps(dict(smoke=False, window=4, stride=1)))
            (root/'backbone_complete.json').write_text(json.dumps(dict(checkpoint_hash='frozen')))
            mapping = dict(sample_ids={}, windows={}, feature_columns=[str(k) for k in range(7)])
            meta, training = [], None
            arrays = {}
            for partition, snippets in [('train', 36), ('validation', 18), ('test', 36)]:
                ids = [f'{partition}_{i}' for i in range(snippets)]
                mapping['sample_ids'][partition] = ids
                mapping['windows'][partition] = snippets*2
                q = rng.normal(size=(snippets*2, 8))
                residual = rng.normal(0, .03, (snippets*2, 4, 7))+.02*q[:, 0, None, None]
                if partition == 'test': residual[snippets:] += .2
                p = root/'residuals'/partition; p.mkdir(parents=True)
                np.save(p/'residual.npy', residual)
                np.save(p/'condition.npy', q)
                np.save(p/'references.npy', np.column_stack((np.repeat(np.arange(snippets), 2), np.tile([3, 4], snippets))))
                (p/'complete.json').write_text(json.dumps(dict(backbone_hash='frozen', windows=len(q))))
                arrays[partition] = q, residual
                for i, sample in enumerate(ids):
                    label = int(partition == 'test' and i >= snippets//2)
                    meta.append(dict(sample_id=sample, partition=partition, vin=f'{partition}_vin{i//2}',
                        sample_label=label, cycle_kind='charge' if i%2 else 'discharge', fault_type='fault' if label else 'normal'))
            (root/'mapping.json').write_text(json.dumps(mapping))
            pd.DataFrame(meta).to_csv(root/'metadata.csv', index=False)
            lite = root/'lite_continuous'; lite.mkdir()
            calibration = LinearResidualCalibration().fit(*arrays['train'])
            calibration.save(lite/'calibration.npz')
            for partition, (q, residual) in arrays.items():
                p = lite/partition; p.mkdir()
                score = np.zeros((len(q), 6))
                score[:, 0] = np.square(residual).mean((1, 2))
                score[:, 1] = np.square(calibration.transform(q, residual)).mean((1, 2))
                np.save(p/'scores.npy', score)
            args = SimpleNamespace(output=str(out), batch=19, threads=2)
            with patch('src.runners.run_local_residual_models.source_for', return_value=(root, lite)), threadpool_limits(limits=2):
                run('NCM', args)
            p = out/'NCM'
            self.assertTrue((p/'complete.json').exists())
            self.assertFalse(json.loads((p/'scale_selection.json').read_text())['test_labels_used'])
            scores = np.load(p/'test_window_scores.npy')
            original = np.load(lite/'test/scores.npy')
            np.testing.assert_array_equal(scores[:, NAMES.index('A')], original[:, 1])
            np.testing.assert_array_equal(scores[:, NAMES.index('Raw')], original[:, 0])
            self.assertEqual(len(pd.read_csv(p/'metrics.csv')), 2*len(NAMES))


if __name__ == '__main__':
    unittest.main(verbosity=2)
