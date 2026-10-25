import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.models.mscm_reconstruction import MSCMReconstructor
from src.models.rcr import condition_features
from src.runners.run_mscm_full_validation import evaluate_model
from src.runners.run_mcr_full_validation import run_unit,HEADS

torch.set_num_threads(1)


class CPUInputs:
    def __init__(self,path):
        self.arrays={p:np.load(path/f'{p}_x.npy') for p in ['train','validation','test']}
    def get(self,partition,index):return torch.from_numpy(np.array(self.arrays[partition][index]))


class FullRelationTests(unittest.TestCase):
    def test_actual_memory_full_loo_training_scoring_and_heldout_normalization_pipeline(self):
        rng=np.random.default_rng(3407)
        torch.manual_seed(3407)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run';cache=Path(folder)/'cache';source=Path(folder)/'source'
            parent=root/'model_one/NCM/A4';(parent/'training').mkdir(parents=True)
            cohort=root/'cohorts/NCM';cohort.mkdir(parents=True)
            source.mkdir();inputs=cache/'inputs/NCM';inputs.mkdir(parents=True)
            fit,cal=np.arange(8),np.arange(8,12)
            np.save(cohort/'fit_window_indices.npy',fit);np.save(cohort/'calibration_window_indices.npy',cal)
            (cohort/'normal_split.json').write_text(json.dumps(dict(fit=list(map(int,fit)),cal=list(map(int,cal)))))
            model=MSCMReconstructor('A4').eval()
            torch.save(model.state_dict(),parent/'training/best.pt')
            mapping={'sample_ids':{}};records=[]
            for partition,n in [('train',12),('validation',4),('test',8)]:
                ids=[partition+str(i) for i in range(n)];mapping['sample_ids'][partition]=ids
                records.extend(dict(sample_id=s,vin=partition+str(i),sample_label=int(partition=='test' and i>=4),
                    partition=partition,operating_regime='state0',
                    fault_type='fault' if partition=='test' and i>=4 else 'normal') for i,s in enumerate(ids))
                x=rng.uniform(.1,.9,(n,8,7)).astype('float32');np.save(inputs/f'{partition}_x.npy',x)
                with torch.inference_mode():
                    prediction,_,state,_=model(torch.from_numpy(x),details=True)
                    residual=(torch.from_numpy(x)-prediction).numpy().astype('float64')
                np.save(parent/f'{partition}_signed_residual.npy',residual.astype('float32'))
                np.save(parent/f'{partition}_memory_state.npy',state.numpy())
                moments=np.stack((residual.mean(1),residual.var(1),np.abs(residual).mean(1),np.abs(residual).var(1)),1)
                np.save(parent/f'{partition}_moments.npy',moments)
                p=source/'residuals'/partition;p.mkdir(parents=True)
                np.save(p/'condition.npy',condition_features(torch.from_numpy(x),1,2).numpy())
                np.save(p/'references.npy',np.column_stack((np.arange(n),np.full(n,7))))
            (source/'mapping.json').write_text(json.dumps(mapping))
            pd.DataFrame(records).to_csv(source/'metadata.csv',index=False)
            evaluate_model('NCM',parent,source,mapping,fit,cal)
            config=SimpleNamespace(cache=str(cache),batch=4,seed=3407,epochs=1,max_epochs=1,patience=1)
            with patch('src.runners.run_mcr_full_validation.InputArrays',CPUInputs), \
                 patch('src.runners.run_mcr_full_validation.source_for',return_value=(source,source)), \
                 patch.object(torch.nn.Module,'cuda',lambda self,*a,**kw:self), \
                 patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self), \
                 patch('torch.cuda.mem_get_info',return_value=(0,0)), \
                 patch('torch.cuda.empty_cache',return_value=None):
                run_unit(root,'NCM',config)
            output=root/'model_two/NCM'
            audit=json.loads((output/'conditions_complete.json').read_text())
            self.assertEqual(audit['raw_target_context_max_error'],0.)
            self.assertTrue(audit['calibration_vins_excluded_from_context_fitting'])
            for name in HEADS:
                a=json.loads((output/name/'mask_audit.json').read_text())
                self.assertEqual(a['trained_residual_target_perturbation_error'],0.)
                error=np.load(output/name/'train_variable_mae.npy')
                scale=np.load(output/name/'normal_relation_error_scale.npy')
                np.testing.assert_array_equal(scale,np.maximum(np.median(error[cal],axis=0),1e-5))
                self.assertTrue((output/name/'training/complete.json').exists())
            base=pd.read_csv(parent/'test_scores.csv',float_precision='round_trip')
            actual=pd.read_csv(output/'test_scores.csv',float_precision='round_trip')
            np.testing.assert_array_equal(base.cal_abs.to_numpy(),actual.B0.to_numpy())
            self.assertTrue(np.isfinite(actual.B5_ECDF).all())
            self.assertEqual(len(pd.read_csv(output/'error_overlap.csv')),10)
            # Exercise three paired seeds and both final report stages entirely
            # in a temporary cwd, never writing fixture metrics to real reports.
            from src.runners.run_mcr_priority5 import share_fixed_contexts,DIRECT
            from src.runners.analyze_mscm_priority5 import analyze_one,analyze_two,final_summary
            from src.runners.run_mscm_priority5 import ARMS
            old_cwd=Path.cwd();prior=Path(folder)/'prior';(prior/'NCM').mkdir(parents=True)
            for part in ['validation','test']:
                frame=pd.read_csv(parent/f'{part}_scores.csv').rename(columns={'cal_abs':'A','raw':'Raw'})
                frame.to_csv(prior/'NCM'/f'{part}_scores.csv',index=False)
            pd.read_csv(parent/'test_snippet_scores.csv').to_csv(prior/'NCM/test_snippet_scores.csv',index=False)
            core=cache;short=core/'NCM/A4';short.mkdir(parents=True)
            for part in ['validation','test']:shutil.copy2(parent/f'{part}_scores.csv',short/f'{part}_scores.csv')
            pd.DataFrame([dict(epoch=1,normal_validation_huber=.1),dict(epoch=2,normal_validation_huber=.09)]).to_csv(parent/'training/history.csv',index=False)
            for name,value in [('complete.json',dict(model_hash='fixture',best_epoch=2,epochs_executed=2,normal_validation_huber=.09)),
                               ('initial.json',dict(parameters=41407,trainable_parameters=8327)),
                               ('memory_epoch_001.json',dict(epoch=1,soft_occupancy=[[.125]*8]*3,hard_occupancy=[[.125]*8]*3))]:
                (parent/'training'/name).write_text(json.dumps(value))
            parent_metrics=pd.read_csv(parent/'metrics.csv').rename(columns={'method':'score'})
            parent_metrics.to_csv(parent/'metrics.csv',index=False)
            for arm in list(ARMS)+['A4_final']:
                if arm!='A4':shutil.copytree(parent,root/'model_one/NCM'/arm)
            (root/'model_one_protocol.json').write_text(json.dumps(dict(units=['NCM'],core=str(core),implementations={})))
            (root/'model_one_complete.json').write_text('{}')
            (root/'model_two_protocol.json').write_text(json.dumps(dict(units=['NCM'],implementations={},parent_signatures={'NCM':{}})))
            (root/'model_two_complete.json').write_text('{}')
            with patch('src.runners.run_mcr_full_validation.InputArrays',CPUInputs), \
                 patch('src.runners.run_mcr_full_validation.source_for',return_value=(source,source)), \
                 patch('src.runners.analyze_mscm_priority5.source_for',return_value=(source,source)), \
                 patch('src.runners.analyze_mscm_priority5.PRIOR',prior), \
                 patch.object(torch.nn.Module,'cuda',lambda self,*a,**kw:self), \
                 patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self), \
                 patch('torch.cuda.mem_get_info',return_value=(0,0)), \
                 patch('torch.cuda.empty_cache',return_value=None):
                for seed in [3408,3409]:
                    child=root/f'relation_seed{seed}';share_fixed_contexts(root,child,'NCM')
                    config.seed=seed;run_unit(child,'NCM',config,DIRECT)
                report_dir=Path(folder)/'report/analysis/rcr';report_dir.mkdir(parents=True)
                for name in ['MSCM_MODEL_ONE_CORE_20261025.md','MCR_GAT_MODEL_TWO_20261025.md',
                             'MSCM_MCR_IMPLEMENTATION_RESULTS_20261025.md','MSCM_MCR_ABLATION_INDEX_20261025.md']:
                    (report_dir/name).write_text('# old report\n\nold data\n')
                try:
                    os.chdir(folder)
                    analyze_one(root,10);analyze_two(root,10);final_summary(root)
                finally:os.chdir(old_cwd)
            self.assertTrue((root/'analysis_complete.json').exists())
            seed_summary=pd.read_csv(root/'model_two_seed_summary.csv')
            count=seed_summary.query('method=="B3_memory_gat"').auroc_count.to_numpy()
            np.testing.assert_array_equal(count,3)


if __name__=='__main__':unittest.main()
