import sys
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import pandas as pd
from src.models.mcr_gat import NormalContextMemoryPrior,MCRSequencePredictor,OriginalRCRLOOPredictor

torch.set_num_threads(1)


class MCRTests(unittest.TestCase):
    def make_prior(self):
        rng=np.random.default_rng(3407)
        q=rng.normal(size=(120,8))
        logit=rng.normal(size=(120,3,8))+.2*q[:,0,None,None]
        weights=np.exp(logit-logit.max(-1,keepdims=True));weights/=weights.sum(-1,keepdims=True)
        return NormalContextMemoryPrior.fit(q,weights.reshape(120,24)),torch.from_numpy(q[:3].astype('float32'))

    def test_control_memory_prior_excludes_target_control_and_roundtrips(self):
        prior,q=self.make_prior()
        baseline=prior(q)
        changed=q.clone();changed[:,:4]+=100
        torch.testing.assert_close(prior(changed)[:,1],baseline[:,1],rtol=0,atol=0)
        changed=q.clone();changed[:,4:]-=100
        torch.testing.assert_close(prior(changed)[:,2],baseline[:,2],rtol=0,atol=0)
        torch.testing.assert_close(baseline[:,:,8:].reshape(3,7,3,8).sum(-1),torch.ones(3,7,3))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'prior.npz';prior.save(p)
            torch.testing.assert_close(NormalContextMemoryPrior.load(p)(q),baseline,rtol=0,atol=0)

    def test_sequence_predictors_never_read_their_target_coordinate(self):
        torch.manual_seed(3407)
        prior,q=self.make_prior();condition=prior(q)
        z=torch.randn(3,32,7)
        for mode in ['gat','memory_gat','memory_mlp','q_only_gat']:
            model=MCRSequencePredictor(mode,32).eval()
            first=model(z,condition)
            changed=z.clone();changed[:,:,3]+=100
            second=model(changed,condition)
            self.assertEqual(first.shape,z.shape)
            torch.testing.assert_close(first[:,:,3],second[:,:,3],rtol=0,atol=0)
            values=z.clone().requires_grad_()
            model(values,condition)[:,:,3].sum().backward()
            torch.testing.assert_close(values.grad[:,:,3],torch.zeros_like(values.grad[:,:,3]),rtol=0,atol=0)
            self.assertGreater(float(values.grad.abs().sum()),0)

    def test_star_attention_excludes_self_and_uses_condition_edge_bias(self):
        torch.manual_seed(3407)
        model=MCRSequencePredictor('memory_gat',32).eval()
        z,c=torch.randn(2,32,7),torch.randn(2,7,32)
        first,attention,bias=model(z,c,details=True)
        _,other,other_bias=model(z,c+5,details=True)
        torch.testing.assert_close(attention.diagonal(dim1=-2,dim2=-1),torch.zeros(2,2,7),rtol=0,atol=0)
        torch.testing.assert_close(attention.sum(-1),torch.ones(2,2,7))
        self.assertGreater(float((bias-other_bias).abs().max()),1e-6)
        self.assertGreater(float((attention-other).abs().max()),1e-7)

    def test_original_head_is_strict_loo_and_sampled_target_agrees(self):
        torch.manual_seed(3407)
        model=OriginalRCRLOOPredictor().eval()
        z,c=torch.randn(2,8,7),torch.randn(2,7,32)
        first=model(z,c)
        changed=z.clone();changed[:,:,4]+=100
        torch.testing.assert_close(model(changed,c)[:,:,4],first[:,:,4],rtol=0,atol=0)
        targets=torch.tensor([2,5])
        sampled=model(z,c,targets=targets)
        selected=first.gather(-1,targets[:,None,None].expand(2,8,1)).squeeze(-1)
        torch.testing.assert_close(sampled,selected,rtol=1e-5,atol=1e-6)

    def test_brand_sequence_length_is_preserved(self):
        z,c=torch.randn(2,127,7),torch.randn(2,7,32)
        model=MCRSequencePredictor('memory_gat',127).eval()
        self.assertEqual(model(z,c).shape,z.shape)

    def test_q_only_control_ignores_memory_and_matches_parameter_count(self):
        model=MCRSequencePredictor('q_only_gat',32).eval()
        memory=MCRSequencePredictor('memory_gat',32)
        self.assertEqual(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in memory.parameters()))
        z,c=torch.randn(2,32,7),torch.randn(2,7,32)
        changed=c.clone();changed[:,:,8:]+=100
        torch.testing.assert_close(model(z,c),model(z,changed),rtol=0,atol=0)

    def test_inherited_scores_fusion_and_full_evaluation_pipeline(self):
        from src.runners.run_mcr_development import evaluate_unit,HEADS
        from src.runners.run_mscm_development import SCORES
        from src.runners.run_rcr_calibration_sensitivity import aggregate
        from src.runners.probe_scale_relation_evidence import read
        import json
        rng=np.random.default_rng(3407)
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);core=folder/'core';root=folder/'result';source=folder/'source'
            parent=core/'NCM/A4';parent.mkdir(parents=True)
            output=root/'NCM';output.mkdir(parents=True)
            mapping={'sample_ids':{}}
            for partition,count in [('train',6),('validation',6),('test',8)]:
                ids=[f'{partition}_{i}' for i in range(count)];mapping['sample_ids'][partition]=ids
                label=np.array([int(partition=='test' and i>=count//2) for i in range(count)])
                frame=pd.DataFrame(dict(sample_id=ids,vin=[f'{partition}_vin{i//2}' for i in range(count)],
                    sample_label=label,fault_type=np.where(label,'fault','normal')))
                refs=np.column_stack((np.repeat(np.arange(count),2),np.tile([3,4],count)))
                p=source/'residuals'/partition;p.mkdir(parents=True)
                np.save(p/'references.npy',refs)
                values=rng.uniform(.1,1,(count*2,4))+np.repeat(label,2)[:,None]
                snippets=frame.merge(aggregate(values,refs,ids,SCORES),on='sample_id')
                snippets.to_csv(parent/f'{partition}_snippet_scores.csv',index=False)
                snippets.to_csv(parent/f'{partition}_scores.csv',index=False)
                for name in HEADS:
                    p=output/name;p.mkdir(exist_ok=True)
                    np.save(p/f'{partition}_window_scores.npy',rng.uniform(.1,1,count*2)+np.repeat(label,2))
                np.save(output/f'{partition}_signed_diagonal_score.npy',values[:,2])
            (source/'mapping.json').write_text(json.dumps(mapping))
            with patch('src.runners.run_mcr_development.source_for',return_value=(source,source)):
                evaluate_unit(core,root,'NCM')
            got=read(output/'test_scores.csv');original=read(parent/'test_scores.csv')
            np.testing.assert_array_equal(got.B0,original.cal_abs)
            np.testing.assert_array_equal(got.B5_ECDF_0p5,.5*(got.B0_ECDF+got.B3_ECDF))
            self.assertTrue((output/'prediction_sets.csv').exists())
            self.assertEqual(len(read(output/'metrics.csv')),2*(11+len(HEADS)))

    def test_final_report_pipeline_with_paired_intervals_and_supplement_controls(self):
        from src.runners.analyze_mcr_development import run,PAIRS
        from src.runners.probe_scale_relation_evidence import error_overlap
        from src.runners.run_rcr_memory_probe import digest
        from scripts.evaluate_rcr import metrics
        import json
        rng=np.random.default_rng(3407)
        main=['B0','B0_raw','B0_signed','B0_legacy','B0_signed_L1','B1_original','B2_gat',
            'B3_memory_gat','B4_memory_mlp','B3_q_only_gat','B0_ECDF','B3_ECDF','B5_ECDF_0p5',
            'B0_MAD','B3_MAD','B5_MAD_0p5']
        supplementary=['B3_scaled_marginal','B0_diagonal_ECDF_0p5']
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);root=folder/'result';core=folder/'core';output=root/'NCM'
            output.mkdir(parents=True);parent=core/'NCM/A4';(parent/'training').mkdir(parents=True)
            for name in ['training/best.pt','calibration_signed.npz','calibration_absolute.npz']:
                (parent/name).write_bytes(b'isolated synthetic report fixture')
            protocol=dict(task='TEST-MCR-REPORT',core=str(core),units=['NCM'],epochs=2,patience=3,
                implementations={},parent_signatures={'NCM':dict(checkpoint=digest(parent/'training/best.pt'),
                calibration=digest(parent/'calibration_signed.npz'),absolute_calibration=digest(parent/'calibration_absolute.npz'))})
            (root/'protocol_manifest.json').write_text(json.dumps(protocol));(root/'complete.json').write_text('{}')
            val=pd.DataFrame(dict(vin=[f'v{i}' for i in range(10)],sample_label=np.zeros(10,dtype=int)))
            test=pd.DataFrame(dict(vin=[f't{i}' for i in range(12)],sample_label=np.repeat([0,1],6)))
            for name in main+supplementary:
                val[name]=rng.uniform(.1,1,len(val));test[name]=rng.uniform(.1,1,len(test))+test.sample_label*.5
            thresholds={name:{str(q):float(val[name].quantile(q)) for q in [.95,.99]} for name in main+supplementary}
            rows=[]
            for name in main+supplementary:
                for q in [.95,.99]:
                    threshold=thresholds[name][str(q)]
                    rows.append(dict(unit='NCM',method=name,q=q,threshold=threshold,validation_fpr=.1,
                        recall_at_test_fpr05=.5,**metrics(test.sample_label,test[name],threshold)))
            records=pd.DataFrame(rows)
            records[records.method.isin(main)].to_csv(output/'metrics.csv',index=False)
            records[records.method.isin(supplementary)].to_csv(output/'marginal_control_metrics.csv',index=False)
            overlap=error_overlap(test,'B0','B3_memory_gat',thresholds['B0']['0.95'],thresholds['B3_memory_gat']['0.95'],'NCM',.95)
            pd.DataFrame([overlap]).to_csv(output/'error_overlap.csv',index=False)
            frames={'validation':val,'test':test}
            original_cwd=Path.cwd()
            try:
                os.chdir(folder);Path('report/analysis/rcr').mkdir(parents=True)
                with patch('src.runners.analyze_mcr_development.audit_unit',return_value=({'unit':'NCM'},[])), \
                     patch('src.runners.analyze_mcr_development.local_analysis'), \
                     patch('src.runners.analyze_mcr_development.context_quality',return_value=[]), \
                     patch('src.runners.analyze_mcr_development.marginal_controls',return_value=(frames,thresholds)):
                    run(root,10)
                self.assertTrue((root/'analysis_complete.json').exists())
                self.assertTrue(Path('report/analysis/rcr/MCR_GAT_MODEL_TWO_20261025.md').exists())
                intervals=pd.read_csv(root/'paired_VIN_intervals.csv')
                self.assertEqual(len(intervals),len(PAIRS)*5*2)
            finally:os.chdir(original_cwd)


if __name__=='__main__':unittest.main(verbosity=2)
