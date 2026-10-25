import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.models.mscm_reconstruction import MSCMReconstructor
from src.runners.full_validation_support import normal_split, TrainingSubset, train_network
from src.runners.run_mscm_full_validation import evaluate_model

torch.set_num_threads(1)


class Arrays:
    def __init__(self):
        generator=torch.Generator().manual_seed(3407)
        self.arrays={p:torch.rand(n,8,7,generator=generator).numpy() for p,n in [('train',12),('validation',4)]}
        self.seen=[]

    def get(self,partition,index):
        if partition=='train':self.seen.extend(np.arange(len(self.arrays[partition]))[index].tolist())
        return torch.from_numpy(np.array(self.arrays[partition][index]))


class FullValidationTests(unittest.TestCase):
    def test_continuation_preserves_original_weights_optimizer_and_history(self):
        from src.runners.run_mscm_priority5 import import_continuation
        from src.runners.run_rcr_memory_probe import digest
        arrays=Arrays();teacher=MSCMReconstructor('A1').eval()
        for p in teacher.parameters():p.requires_grad_(False)
        model=MSCMReconstructor('A4');model.load_state_dict(teacher.state_dict(),strict=False)
        first=SimpleNamespace(batch=4,seed=3407,epochs=1,max_epochs=1,patience=1)
        longer=SimpleNamespace(batch=4,seed=3407,epochs=3,max_epochs=3,patience=3)
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'old';output=Path(folder)/'new'
            train_network(model,arrays,source,first,'memory',teacher=teacher)
            signature=digest(source/'last.pt')
            continued=MSCMReconstructor('A4')
            import_continuation(continued,source,output,longer)
            old=torch.load(source/'last.pt',weights_only=False)
            new=torch.load(output/'last.pt',weights_only=False)
            for k,v in old['model'].items():torch.testing.assert_close(new['model'][k],v,rtol=0,atol=0)
            self.assertEqual(new['epoch'],old['epoch'])
            self.assertEqual(len(new['optimizer']['state']),len(old['optimizer']['state']))
            train_network(continued,arrays,output,longer,'memory',teacher=teacher)
            history=pd.read_csv(output/'history.csv')
            self.assertEqual(history.epoch.tolist(),[1,2,3])
            self.assertEqual(digest(source/'last.pt'),signature)

    def test_full_loo_gradient_accumulation_matches_effective_batch(self):
        class Relations(Arrays):
            def get(self,partition,index):
                z=super().get(partition,index)
                return z,torch.zeros(len(z),7,32)
        class Predictor(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(7,7)
                torch.nn.init.zeros_(self.linear.weight);torch.nn.init.zeros_(self.linear.bias)
            def forward(self,z,c):return self.linear(z)
        states=[]
        for micro in [1,4]:
            model=Predictor();arrays=Relations()
            config=SimpleNamespace(batch=4,microbatch=micro,seed=3407,epochs=1,max_epochs=1,patience=1)
            with tempfile.TemporaryDirectory() as folder:
                train_network(model,arrays,Path(folder),config,'relation',relation=True,memory_weight=0.)
                self.assertTrue(torch.all(model.linear.bias.grad!=0))
                history=pd.read_csv(Path(folder)/'history.csv')
                self.assertTrue(history.all_target_variables.all())
                self.assertEqual(history.full_training_windows.iloc[0],12)
            states.append({k:v.clone() for k,v in model.state_dict().items()})
        for k in states[0]:torch.testing.assert_close(states[0][k],states[1][k],atol=1e-7,rtol=1e-6)

    def test_vin_split_is_disjoint_reproducible_and_covers_all_training_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source';source.mkdir()
            pd.DataFrame(dict(sample_id=[f's{i}' for i in range(8)],vin=[f'v{i//2}' for i in range(8)],
                sample_label=0,partition='train')).to_csv(source/'metadata.csv',index=False)
            mapping=dict(sample_ids={'train':[f's{i}' for i in range(8)]})
            (source/'mapping.json').write_text(json.dumps(mapping))
            (source/'residuals/train').mkdir(parents=True)
            refs=np.column_stack((np.repeat(np.arange(8),2),np.tile([3,7],8)))
            np.save(source/'residuals/train/references.npy',refs)
            fit,cal,_=normal_split(source,Path(folder)/'split')
            self.assertEqual(set(fit)|set(cal),set(range(16)))
            self.assertFalse(set(fit)&set(cal))
            labels=pd.read_csv(source/'metadata.csv').vin.to_numpy()[refs[:,0]]
            self.assertFalse(set(labels[fit])&set(labels[cal]))
            a,b,_=normal_split(source,Path(folder)/'split')
            np.testing.assert_array_equal(a,fit);np.testing.assert_array_equal(b,cal)

    def test_reconstruction_training_excludes_calibration_and_freezes_teacher_and_encoder(self):
        arrays=Arrays();fit=np.arange(8);training=TrainingSubset(arrays,fit)
        teacher=MSCMReconstructor('A1').eval()
        for p in teacher.parameters():p.requires_grad_(False)
        model=MSCMReconstructor('A4')
        model.load_state_dict(teacher.state_dict(),strict=False)
        frozen={k:v.clone() for k,v in model.backbone.state_dict().items() if not k.startswith('decoder.')}
        prototypes=[p.prototypes.detach().clone() for p in model.memories]
        config=SimpleNamespace(batch=4,seed=3407,epochs=2,max_epochs=2,patience=2)
        with tempfile.TemporaryDirectory() as folder:
            train_network(model,training,Path(folder),config,'memory',teacher=teacher)
            self.assertTrue(set(arrays.seen)<=set(fit))
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            for k,v in frozen.items():torch.testing.assert_close(model.backbone.state_dict()[k],v,rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(p.prototypes,v) for p,v in zip(model.memories,prototypes)))
            self.assertTrue((Path(folder)/'memory_epoch_001.json').exists())
            state={k:v.clone() for k,v in model.state_dict().items()}
            train_network(model,training,Path(folder),config,'memory',teacher=teacher)
            for k,v in state.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)

    def test_main_calibration_never_fits_reconstruction_training_vins(self):
        rng=np.random.default_rng(3407)
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source';output=Path(folder)/'result'
            source.mkdir();output.mkdir();mapping={'sample_ids':{}}
            records=[]
            for part,n in [('train',8),('validation',4),('test',6)]:
                ids=[part+str(i) for i in range(n)];mapping['sample_ids'][part]=ids
                records.extend(dict(sample_id=s,vin=part+str(i),sample_label=int(part=='test' and i>=3),partition=part)
                    for i,s in enumerate(ids))
                path=source/'residuals'/part;path.mkdir(parents=True)
                np.save(path/'condition.npy',rng.normal(size=(n,8)))
                np.save(path/'references.npy',np.column_stack((np.arange(n),np.full(n,7))))
                values=rng.uniform(.1,1,(n,4,7));np.save(output/f'{part}_moments.npy',values)
            pd.DataFrame(records).to_csv(source/'metadata.csv',index=False)
            fit,cal=np.arange(4),np.arange(4,8)
            evaluate_model('NCM',output,source,mapping,fit,cal)
            with np.load(output/'calibration_absolute.npz') as v:first={k:v[k].copy() for k in v.files}
            values=np.load(output/'train_moments.npy');values[fit]*=100
            np.save(output/'train_moments.npy',values)
            evaluate_model('NCM',output,source,mapping,fit,cal)
            with np.load(output/'calibration_absolute.npz') as second:
                for k,v in first.items():np.testing.assert_array_equal(second[k],v)


if __name__=='__main__':unittest.main()
