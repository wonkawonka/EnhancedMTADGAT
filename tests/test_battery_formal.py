"""Regression checks for methodological risks in the formal experiment runner."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from src.experiments.battery_data import window_starts
from src.experiments.artifacts import Efficiency,package_cell
from src.experiment_preflight import Preflight
from src.models.rcr import RCRBackbone
from src.runners.run_battery_formal import train_model,seed_all,object_scores,evaluate,stage_better


class FakeBundle:
    def __init__(self):
        self.lengths={'train':np.array([12,15]),'validation':np.array([11,13])}
        self.arrays={p:np.linspace(.01,.9,int(ns.sum())*7,dtype=np.float32).reshape(-1,7) for p,ns in self.lengths.items()}
    def batch(self,part,positions,span,dtype):
        return self.arrays[part][np.asarray(positions)[:,None]+np.arange(span)[None,:]].astype(dtype)


class FormalTests(unittest.TestCase):
    def test_no_cross_boundary_and_complete_tail(self):
        ns=np.array([105,109,128]);ends=np.cumsum(ns)
        for span in (10,32,100,101,105):
            for stride in (1,4,5,100,128):
                pos,counts=window_starts(ns,span,stride)
                owner=np.searchsorted(ends,pos,side='right')
                self.assertTrue(np.all(pos+span<=ends[owner]))
                for i,end in enumerate(ends):
                    self.assertEqual(pos[owner==i][-1]+span,end)
                self.assertEqual(sum(counts),len(pos))

    def test_entity_threshold_and_author_AT_tie_comparator(self):
        v=np.array([.1,.9,.2,.8]);m=dict(labels=[0,0,1,1],entities=['n','n','f','f'],sample_ids=['a','b','c','d'])
        values,labels,ids=object_scores(v,m,'Brand1')
        np.testing.assert_allclose(values,[.9,.8]);self.assertEqual(ids,['n','f'])
        metrics=evaluate(np.array([.1,.9]),np.array([.2,.4]),np.array([0,1]))
        self.assertEqual(metrics['threshold'],.39)
        self.assertTrue(stage_better('at',[1.,2.],[1.,2.]))
        self.assertFalse(stage_better('rcr',[1.],[1.]))
        self.assertFalse(stage_better('at',[.5,3.],[1.,2.]))

    def test_archive_allowlist_and_lossless_roundtrip(self):
        import zipfile,io
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cell=root/'cell';cell.mkdir()
            (cell/'metrics.json').write_text('{"AP": 0.5}')
            (cell/'secret.key').write_text('must not export')
            scores=np.array([1.2345678901234567,2.],dtype=np.float64)
            np.savez_compressed(cell/'test_scores.npz',scores=scores)
            np.save(cell/'test_native.npy',scores)
            index=package_cell(cell,root/'export','fixture')
            for row in index['archives']:
                with zipfile.ZipFile(root/'export'/row['file']) as z:
                    self.assertNotIn('secret.key',z.namelist())
                    if 'test_scores.npz' in z.namelist():
                        a=np.load(io.BytesIO(z.read('test_scores.npz')))['scores']
                        self.assertEqual(a.dtype,scores.dtype);np.testing.assert_array_equal(a,scores)

    def test_IF_full_entry_exports_recomputable_results(self):
        from src.experiments.battery_data import finish_bundle,write_json
        from src.runners.run_battery_formal import main,source_manifest
        from src.experiment_preflight import sha256
        import sys,zipfile,copy
        plan=json.loads(Path('configs/external/battery_kaggle_formal_20261112_v1.json').read_text())
        job=copy.deepcopy(next(j for j in plan['cells'] if j['dataset']=='LFP' and j['method']=='iforest'))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'inputs/LFP';data.mkdir(parents=True);mapping={}
            for part,n in [('train',3),('validation',2),('test',4)]:
                values=np.random.default_rng(n).normal(size=(n*105,7)).astype(np.float32)
                labels=[0]*n if part!='test' else [0,0,1,1]
                np.save(data/f'{part}_raw.npy',values);np.save(data/f'{part}_lengths.npy',np.full(n,105))
                mapping[part]=dict(sample_ids=[part+str(i) for i in range(n)],entities=[part+str(i) for i in range(n)],labels=labels)
            finish_bundle(data,'LFP',['x'+str(i) for i in range(7)],mapping,{'fixture':True})
            plan['data']['LFP'].update(train=3,validation=2,test_normal=2,test_fault=2,train_points=315,validation_points=210,test_points=420)
            plan['bundle_manifests']['LFP']=sha256(data/'bundle_manifest.json');plan['cells']=[job]
            file=root/'plan.json';write_json(file,plan)
            evidence=root/'evidence.json'
            write_json(evidence,dict(status='passed',files=source_manifest(plan)['files'],dataset='LFP',
                experiments={'IsolationForest':dict(config=job['config'],architecture_overrides={})}))
            argv=['formal','--plan',str(file),'--bundle',str(root/'inputs'),'--output',str(root/'out'),
                  '--dataset','LFP','--experiment','IsolationForest','--device','cpu','--smoke-evidence',str(evidence)]
            with patch.object(sys,'argv',argv):self.assertEqual(main(),0)
            cell=root/'out/LFP/IsolationForest/seed3407'
            self.assertEqual(json.loads((cell/'complete.json').read_text())['status'],'complete')
            index=json.loads((root/'out/archives/LFP_IsolationForest_s3407_archive_index.json').read_text())
            result=next(x for x in index['archives'] if x['group']=='results')
            with zipfile.ZipFile(root/'out/archives'/result['file']) as z:
                self.assertIn('complete.json',z.namelist());self.assertIn('efficiency.json',z.namelist());self.assertIn('test_scores.npz',z.namelist())

    @patch('torch.cuda.is_available',return_value=False)
    def test_mid_epoch_resume_matches_uninterrupted_training(self,_):
        torch.set_num_threads(1)
        cfg=dict(span=4,train_stride=2,validation_stride=2,batch=3,validation_batch=4,
                 precision='float32',shuffle=True,seed=3407,optimizer='AdamW',lr=.001,weight_decay=.0001,formal_epochs=2)
        policy=dict(early_stopping_patience=None,selection='minimum normal validation')
        def model():return RCRBackbone(7,hidden=8,node_dim=2,layers=1,heads=2,ffn_dim=16,dropout=.1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=root/'plan.json';plan.write_text('{}');device=torch.device('cpu')
            def run(out,net,limit):
                out.mkdir(exist_ok=True);gate=Preflight(out,plan,'resume fixture');meter=Efficiency(out,device)
                try:return train_model(net,'rcr',cfg,policy,FakeBundle(),out,device,meter,gate,{'fixture':1},limit)
                finally:meter.close()
            seed_all(3407);a=model();ready,_=run(root/'a',a,0);self.assertTrue(ready)
            seed_all(3407);b=model();ready,_=run(root/'b',b,1e-9);self.assertFalse(ready)
            state=torch.load(root/'b/latest.pt',weights_only=False)
            self.assertEqual(state['next_step'],1)
            c=model();ready,_=run(root/'b',c,0);self.assertTrue(ready)
            for key,tensor in a.state_dict().items():torch.testing.assert_close(tensor,c.state_dict()[key],atol=0,rtol=0)
            ha=json.loads((root/'a/history.json').read_text());hb=json.loads((root/'b/history.json').read_text())
            self.assertEqual(ha,hb)

if __name__=='__main__':unittest.main()
