import unittest
import torch

from src.models.mscm_reconstruction import MSCMReconstructor
from src.models.mcr_direct_context import DirectMaskedMemoryContext
from src.models.mcr_gat import MCRSequencePredictor

torch.set_num_threads(1)


class DirectContextTests(unittest.TestCase):
    def make_context(self):
        torch.manual_seed(3407)
        return DirectMaskedMemoryContext(MSCMReconstructor('A4'),torch.zeros(7),torch.zeros(8),
            torch.ones(8),torch.full((8,),-3.),torch.full((8,),3.))

    def test_actual_memory_context_excludes_each_raw_target_including_controls(self):
        context=self.make_context()
        x=torch.randn(2,8,7)
        audit=context.audit(x)
        self.assertEqual(audit['raw_target_context_max_error'],0.)
        weights=context(x)[:,:,8:].reshape(2,7,3,8)
        torch.testing.assert_close(weights.sum(-1),torch.ones(2,7,3))
        values=x.clone().requires_grad_()
        context(values)[:,3].sum().backward()
        torch.testing.assert_close(values.grad[:,:,3],torch.zeros_like(values.grad[:,:,3]),rtol=0,atol=0)
        self.assertGreater(float(values.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in context.reconstructor.parameters()))

    def test_memory_weights_are_actual_masked_retrieval_and_change_other_targets(self):
        context=self.make_context();x=torch.randn(2,8,7)
        output=context(x)
        masked=x.clone();masked[:,:,3]=context.neutral[3]
        expected=[]
        for patch,memory in zip(context.reconstructor.patches(masked),context.reconstructor.memories):
            expected.append(memory(patch)[2].mean(1))
        torch.testing.assert_close(output[:,3,8:],torch.cat(expected,-1),rtol=0,atol=0)
        changed=x.clone();changed[:,:,4]+=2
        self.assertGreater(float((context(changed)[:,3,8:]-output[:,3,8:]).abs().max()),0.)

    def test_end_to_end_relation_target_is_masked_in_z_and_raw_memory_context(self):
        context=self.make_context();model=MCRSequencePredictor('memory_gat',8).eval()
        x,z=torch.randn(2,8,7),torch.randn(2,8,7)
        before=model(z,context(x))
        for j in range(7):
            xx,zz=x.clone(),z.clone();xx[:,:,j]+=20;zz[:,:,j]-=20
            after=model(zz,context(xx))
            torch.testing.assert_close(before[:,:,j],after[:,:,j],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
