import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.residual_scale_probe import (shrink_variance, scale_components,
    centered_energy, select_normal_only)
from src.runners.probe_scale_relation_evidence import weighted_metric, error_overlap


class EvidenceTests(unittest.TestCase):
    def test_variance_shrinkage_and_state_decomposition(self):
        global_scale = np.array([.002, .02, .1])
        conditional = global_scale*np.array([[2, 3, 4], [.5, .8, 1.2]])
        np.testing.assert_allclose(shrink_variance(conditional, global_scale, 0),
                                   np.broadcast_to(global_scale, conditional.shape))
        np.testing.assert_allclose(shrink_variance(conditional, global_scale, 1), conditional)
        shared, shape = scale_components(conditional, global_scale)
        np.testing.assert_allclose(shared*shape/global_scale, conditional)
        np.testing.assert_allclose(np.exp(np.log(shape/global_scale).mean(1)), 1)
        with self.assertRaises(ValueError):
            shrink_variance(conditional, global_scale, -1)

    def test_moment_energy_matches_explicit_residual(self):
        rng = np.random.default_rng(42)
        r, mu = rng.normal(size=(11, 7, 3)), rng.normal(size=(11, 3))
        np.testing.assert_allclose(centered_energy(r.mean(1), r.var(1), mu),
                                   np.square(r-mu[:, None, :]).mean(1))

    def test_vin_equal_selection_and_one_se_avoid_long_vehicle_domination(self):
        vins = np.array(['long']*100+['short_a', 'short_b', 'short_c'])
        a = np.r_[np.ones(100)*.8, 0, 0, 0]
        b = np.r_[np.zeros(100), 1, 1, 1]
        winner, conservative, records = select_normal_only({'global': a, 'conditional': b},
            vins, {'global': 0, 'conditional': 1})
        self.assertEqual(winner, 'global')
        self.assertEqual(conservative, 'global')
        self.assertLess(b.mean(), a.mean())

    def test_weighted_rank_metrics_preserve_ties(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        s = np.array([.2, .2, .6, .6, .9, .9])
        w = np.array([2, 4, 1, 3, 0, 2])
        result = weighted_metric(y, s, .5, w)
        self.assertAlmostEqual(result[0], average_precision_score(y, s, sample_weight=w))
        self.assertAlmostEqual(result[1], roc_auc_score(y, s, sample_weight=w))

    def test_overlap_reports_rescue_cost_and_unique_vins(self):
        f = pd.DataFrame(dict(vin=['fault_a', 'fault_a', 'fault_b', 'normal_a', 'normal_b'],
            sample_label=[1, 1, 1, 0, 0], a=[0, 0, 1, 0, 1], b=[1, 1, 0, 1, 1]))
        r = error_overlap(f, 'a', 'b', .5, .5, 'test', .95)
        self.assertEqual(r['fault_relation_only'], 2)
        self.assertEqual(r['fault_unique_vins_rescued'], 1)
        self.assertEqual(r['normal_relation_only'], 1)
        self.assertEqual(r['fault_base_only'], 1)
        self.assertEqual(r['added_FP_per_rescued_fault'], .5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
