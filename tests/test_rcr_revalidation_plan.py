import copy
import json
from pathlib import Path
import unittest
import tempfile

import numpy as np
import pandas as pd

from src.models.rcr import LinearResidualCalibration
from src.runners.run_rcr_revalidation import expand_plan, fit_scale, score_names
from src.runners.full_validation_support import ranked_frames


class RCRRevalidationTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(Path('configs/internal/rcr_revalidation_20261107_v2.json').read_text())

    def test_job_counts_defaults_and_frozen_split(self):
        before = copy.deepcopy(self.plan)
        jobs = expand_plan(self.plan)
        self.assertEqual(self.plan, before)
        self.assertEqual(len(jobs), 77)
        self.assertEqual(sum(j['stage']=='multiseed' for j in jobs), 21)
        self.assertEqual(sum(j['stage']=='ablation' for j in jobs), 14)
        self.assertEqual(sum(j['stage']=='sensitivity' for j in jobs), 42)
        self.assertEqual(sum(len(score_names(j,self.plan)) for j in jobs), 126)
        self.assertEqual(len({j['id'] for j in jobs}),77)
        self.assertTrue(all(j['args']['fresh_initialization'] for j in jobs))
        self.assertTrue(all('split_seed' not in j['args'] for j in jobs))

    def test_sensitivity_changes_exactly_one_parameter(self):
        for j in expand_plan(self.plan):
            if j['stage']=='sensitivity':
                changed=[k for k,v in self.plan['common_args'].items() if j['args'][k]!=v]
                self.assertEqual(changed,[j['axis']])
                self.assertEqual(j['seed'],3407)

    def test_scoring_ablations_never_create_training_jobs(self):
        jobs=expand_plan(self.plan)
        self.assertTrue(all('calibration' not in j['name'] and 'min_std' not in j['name'] for j in jobs))
        for j in jobs:
            names=score_names(j,self.plan)
            self.assertNotIn('cal_abs',names)
            self.assertEqual(names[0],'scale')
            self.assertEqual('no_calibration' in names,j['name']=='R1_full' and j['seed']==3407)

    def test_brand1_plan_omits_scale_floor_sweep(self):
        plan = json.loads(Path('configs/internal/rcr_brand1_revalidation_20261108.json').read_text())
        jobs = expand_plan(plan)
        self.assertEqual(len(jobs), 55)
        self.assertEqual(sum(len(score_names(j, plan)) for j in jobs), 60)
        self.assertEqual(plan['units'], [f'Brand1_fold{i}' for i in range(5)])
        self.assertTrue(all(not n.startswith('scale_floor_') for j in jobs for n in score_names(j, plan)))
        self.assertEqual(plan['scoring']['min_std'], .05)
        self.assertTrue(all(j['args']['epochs'] == 30 for j in jobs))

    def test_brand1_scores_aggregate_to_vehicle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, ids, scores = [], {}, {}
            for partition in ['train', 'validation', 'test']:
                ids[partition] = [partition + str(i) for i in range(4)]
                metadata.extend(dict(sample_id=s, car=str(i // 2 + 1), sample_label=i // 2)
                                for i, s in enumerate(ids[partition]))
                output = root / 'residuals' / partition
                output.mkdir(parents=True)
                np.save(output / 'references.npy', np.column_stack([np.arange(4), np.full(4,126)]))
                scores[partition] = np.array([[1.], [3.], [2.], [6.]])
            pd.DataFrame(metadata).to_csv(root / 'metadata.csv', index=False)
            frames, snippets = ranked_frames(root, dict(dataset='Brand1', sample_ids=ids), scores, ['scale'])
            self.assertEqual(len(snippets['test']), 4)
            self.assertEqual(len(frames['test']), 2)
            self.assertEqual(frames['test']['scale'].tolist(), [3., 6.])

    def test_moment_fit_matches_original_explicit_residual_estimator(self):
        rng=np.random.default_rng(3407)
        q=rng.normal(size=(100,8))
        r=rng.normal(size=(100,12,7))*.04+q[:,0,None,None]*.01
        moments=np.stack((r.mean(1),r.var(1)),axis=1)
        for floor in [.005,.05,.1]:
            direct=LinearResidualCalibration(floor).fit(q,r)
            moment=fit_scale(q,moments,floor)
            for field in ['q_mean','q_std','mean_coef','scale_coef','scale_low','scale_high']:
                np.testing.assert_allclose(getattr(moment,field),getattr(direct,field),rtol=1e-10,atol=1e-12)
            mu,sigma=moment.parameters(q)
            energy=((moments[:,1]+(moments[:,0]-mu)**2)/sigma**2).mean(1)
            np.testing.assert_allclose(energy,(direct.transform(q,r)**2).mean((1,2)),rtol=1e-10,atol=1e-12)


if __name__=='__main__':unittest.main()
