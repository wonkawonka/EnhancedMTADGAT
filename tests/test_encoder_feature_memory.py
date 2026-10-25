"""Frozen H extraction, residual-free memory and unchanged-RCR fusion tests."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from src.models.rcr import RCRBackbone, EmpiricalCDF
from src.models.rcr_feature_memory import EncoderFeatureMemory, frozen_encoder_output
from src.runners.run_brand3_encoder_memory import SCORE_NAMES, aggregate_vehicle, fuse_percentiles, run_fold
from src.runners.run_rcr import state_hash


class FeatureMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3407)

    def test_encoder_is_before_decoder_and_preserves_backbone(self):
        model = RCRBackbone(7).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        x = torch.randn(3, 11, 7)
        before = state_hash(model)
        h, reconstruction = frozen_encoder_output(model, x)
        torch.testing.assert_close(reconstruction, model(x), rtol=0, atol=0)
        self.assertEqual(h.shape, (3, 11, 32))
        self.assertEqual(state_hash(model), before)
        self.assertFalse(h.requires_grad)
        self.assertEqual(len(model.temporal._forward_hooks), 0)
        changed_x = x.clone()
        changed_x[:, :, 3] += 5  # Response input is allowed in the raw-X Encoder.
        self.assertFalse(torch.allclose(h, frozen_encoder_output(model, changed_x)[0]))
        with torch.no_grad():
            model.decoder[-1].bias.add_(10)
        new_h, new_reconstruction = frozen_encoder_output(model, x)
        torch.testing.assert_close(new_h, h, rtol=0, atol=0)
        self.assertFalse(torch.allclose(reconstruction, new_reconstruction))
        model.train()
        with self.assertRaisesRegex(ValueError, 'eval-mode frozen'):
            frozen_encoder_output(model, x)

    def test_memory_distance_and_serialization_without_residuals(self):
        rng = np.random.default_rng(3407)
        h = rng.normal(0, .1, (100, 32))+np.repeat([-2, 2], 50)[:, None]
        with threadpool_limits(limits=2):
            memory = EncoderFeatureMemory(2).fit(h)
        normalized = memory.standardize(h)
        direct = np.square(normalized[:, None, :]-memory.centers[None, :, :]).mean(-1).min(1)
        np.testing.assert_array_equal(memory.score(h), direct)
        centers = memory.centers.copy()
        self.assertGreater(memory.score(h+20).mean(), 100*memory.score(h).mean())
        np.testing.assert_array_equal(memory.centers, centers)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'memory.npz'
            memory.save(p)
            restored = EncoderFeatureMemory.load(p)
            np.testing.assert_array_equal(restored.score(h), memory.score(h))

    def test_fusion_occurs_before_vehicle_topk(self):
        a, b = np.zeros(20), np.zeros(20)
        a[0], b[1] = 1, 1
        fused = fuse_percentiles(a, b)
        values = np.zeros((20, len(SCORE_NAMES)))
        values[:, SCORE_NAMES.index('RCR_CDF')] = a
        values[:, SCORE_NAMES.index('Memory_CDF_k6')] = b
        values[:, SCORE_NAMES.index('Fusion_k6')] = fused
        mapping = dict(sample_ids=dict(test=[f's{i}' for i in range(20)]),
            cars=dict(test=['car']*20), labels=dict(test=[0]*20))
        _, vehicle = aggregate_vehicle(values, mapping, 'test')
        self.assertEqual(vehicle.RCR_CDF.iloc[0], 1)
        self.assertEqual(vehicle.Memory_CDF_k6.iloc[0], 1)
        self.assertEqual(vehicle.Fusion_k6.iloc[0], .5)
        with self.assertRaises(ValueError):
            fuse_percentiles(a+2, b)

    def test_full_hidden_fold_runs_without_any_residual_files(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory(prefix='encoder-memory-') as tmp:
            base, output = Path(tmp)/'base', Path(tmp)/'memory'
            source = base/'sources/fold0/full'
            baseline = base/'experiments/fold0/A_current_rcr'
            features = output/'features/fold0'
            for p in [source, baseline, features]:
                p.mkdir(parents=True)
            (source/'backbone_complete.json').write_text(json.dumps(dict(checkpoint_hash='frozen')))
            (features/'complete.json').write_text(json.dumps(dict(backbone_hash='frozen', backbone_unchanged=True)))
            (features/'protocol_manifest.json').write_text(json.dumps(dict(uses_residuals=False, uses_calibration=False)))
            mapping = dict(sample_ids={}, cars={}, labels={})
            cdf = None
            baseline_bytes = {}
            for partition, n_cars in [('train', 3), ('validation', 2), ('test', 4)]:
                n = n_cars*20
                ids, cars = [f'{partition}:{i}' for i in range(n)], [f'{partition}_car{i//20}' for i in range(n)]
                labels = [int(partition == 'test' and i >= 40) for i in range(n)]
                mapping['sample_ids'][partition], mapping['cars'][partition], mapping['labels'][partition] = ids, cars, labels
                h = rng.normal(size=(n, 32))+np.asarray(labels)[:, None]*3
                np.save(features/f'{partition}_features.npy', h)
                magnitude = rng.uniform(0, 1, n)+np.asarray(labels)*.2
                if partition == 'train':
                    cdf = EmpiricalCDF().fit(magnitude)
                score = np.column_stack((magnitude, magnitude, cdf.transform(magnitude)))
                path = baseline/f'{partition}_window_scores.npy'
                np.save(path, score)
                baseline_bytes[partition] = path.read_bytes()
                p = source/'residuals'/partition
                p.mkdir(parents=True)
                np.save(p/'references.npy', np.column_stack((np.arange(n), np.full(n, 126))))
                # There is deliberately no residual.npy or condition.npy.
                values = np.zeros((n, len(SCORE_NAMES)))
                values[:, :2] = score[:, 1:3]
                _, vehicle = aggregate_vehicle(values, mapping, partition)
                vehicle[['car', 'label', 'snippets', 'RCR', 'RCR_CDF']].rename(
                    columns={'RCR': 'magnitude', 'RCR_CDF': 'magnitude_cdf'}).to_csv(
                    baseline/f'{partition}_vehicle_scores.csv', index=False)
            (source/'mapping.json').write_text(json.dumps(mapping))
            args = SimpleNamespace(seed=3407, n_init=2, max_iter=200, stability_seeds=[3408], threads=2, batch=17)
            with threadpool_limits(limits=2):
                run_fold(base, output, 0, args)
            directory = output/'experiments/fold0'
            record = json.loads((directory/'complete.json').read_text())
            self.assertFalse(record['memory_fit_uses_residuals'])
            self.assertTrue(record['original_RCR_window_scores_unchanged'])
            self.assertEqual(len(pd.read_csv(directory/'metrics.csv')), 2*len(SCORE_NAMES))
            for partition in ['train', 'validation', 'test']:
                self.assertEqual((baseline/f'{partition}_window_scores.npy').read_bytes(), baseline_bytes[partition])


if __name__ == '__main__':
    unittest.main(verbosity=2)
