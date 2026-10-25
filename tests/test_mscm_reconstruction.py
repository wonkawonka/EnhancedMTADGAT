import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.models.rcr import RCRBackbone
from src.models.mscm_reconstruction import (MSCMReconstructor, PatchMemory, patchify,
    overlap_average, FixedRobustResidualCalibration)

torch.set_num_threads(1)


class MSCMTests(unittest.TestCase):
    def test_patch_tail_coverage_and_autograd(self):
        for length in [3, 8, 32, 127]:
            for patch in [4,8,16]:
                x = torch.ones(2,length,4,requires_grad=True)
                patches = patchify(x,patch,4)
                filled = overlap_average(patches,length,patch,4)
                torch.testing.assert_close(filled,x)
                filled.sum().backward()
                self.assertTrue(torch.all(x.grad > 0))

    def test_exposed_backbone_interface_preserves_original_equation(self):
        b = RCRBackbone(7).eval(); x = torch.randn(2,32,7)
        h = b.value(x.unsqueeze(-1))+b.variable_embedding
        h = b.fusion(b.graph(h).flatten(-2))
        import math
        position = torch.arange(32,dtype=h.dtype)[:,None]
        freq = torch.exp(torch.arange(0,32,2,dtype=h.dtype)*(-math.log(10000)/32))
        pe = torch.zeros_like(h[0]); pe[:,0::2] = (position*freq).sin(); pe[:,1::2] = (position*freq).cos()
        with torch.no_grad():
            expected = b.decoder(b.temporal(h+pe))
            torch.testing.assert_close(b(x),expected,rtol=0,atol=0)
            m = MSCMReconstructor('A0').eval(); m.warm_start(b.state_dict())
            torch.testing.assert_close(m(x),expected,rtol=0,atol=0)

    def test_memory_and_ordinary_bottleneck_parameter_counts_match(self):
        memory, ordinary = MSCMReconstructor('A4'), MSCMReconstructor('A6')
        self.assertEqual(sum(p.numel() for p in memory.parameters()),sum(p.numel() for p in ordinary.parameters()))
        for m in [memory,ordinary]:
            m.configure_stage('memory')
        self.assertEqual(sum(p.numel() for p in memory.parameters() if p.requires_grad),
                         sum(p.numel() for p in ordinary.parameters() if p.requires_grad))

    def test_memory_changes_reconstruction_and_teacher_has_no_gradient(self):
        torch.manual_seed(3407)
        m = MSCMReconstructor('A4',fusion='pure').eval()
        x = torch.randn(2,32,7)
        targets = [v.detach().requires_grad_() for v in m.patches(x)]
        y, loss, state, weights = m(x,details=True,teacher_patches=targets)
        self.assertEqual(state.shape,(2,24))
        for w in weights: torch.testing.assert_close(w.sum(-1),torch.ones_like(w[...,0]))
        (y.square().mean()+loss).backward()
        for target in targets: self.assertIsNone(target.grad)
        self.assertIsNotNone(m.memories[0].prototypes.grad)
        before = y.detach().clone()
        with torch.no_grad():
            m.memories[0].prototypes.add_(3.)
            after = m(x)
        self.assertGreater(float((before-after).abs().max()),1e-4)

    def test_teacher_loss_sums_dimensions_and_scales_as_specified(self):
        m = MSCMReconstructor('A4').eval(); x = torch.randn(2,32,7)
        targets = [p.detach()+.1 for p in m.patches(x)]
        expected = x.new_zeros(())
        for query,target,memory in zip(m.patches(x),targets,m.memories):
            _,retrieved,_ = memory(query)
            reference = (target-memory.center)/memory.scale
            expected += (retrieved-reference).square().sum(-1).mean()
        actual = m(x,details=True,teacher_patches=targets)[1]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_only_requested_modules_train_in_each_stage(self):
        m = MSCMReconstructor('A4')
        m.configure_stage('pretrain')
        self.assertFalse(m.backbone.value.weight.requires_grad)
        self.assertFalse(m.backbone.graph.nodes.requires_grad)
        self.assertTrue(m.backbone.temporal.layers[0].linear1.weight.requires_grad)
        self.assertFalse(m.memories[0].prototypes.requires_grad)
        m.configure_stage('memory')
        self.assertFalse(m.backbone.temporal.layers[0].linear1.weight.requires_grad)
        self.assertFalse(m.convolutions[0][0].weight.requires_grad)
        self.assertTrue(m.memories[0].prototypes.requires_grad)
        self.assertTrue(m.backbone.decoder[0].weight.requires_grad)

    def test_memory_condition_cannot_read_its_target_input(self):
        m = MSCMReconstructor('A4').eval(); x = torch.randn(2,16,7)
        neutral = torch.zeros(7)
        first = m.masked_memory_condition(x,neutral)
        changed = x.clone(); changed[:,:,3] += 100
        second = m.masked_memory_condition(changed,neutral)
        torch.testing.assert_close(first[:,3],second[:,3],rtol=0,atol=0)

    def test_signed_and_absolute_calibration_are_distinct(self):
        rng = np.random.default_rng(3407)
        q = rng.normal(size=(50,8)); r = rng.normal(size=(50,8,7))
        signed = FixedRobustResidualCalibration().fit(q,r.mean(1),r.var(1))
        absolute = FixedRobustResidualCalibration().fit(q,np.abs(r).mean(1),np.abs(r).var(1))
        self.assertGreater(float(np.abs(signed.coef-absolute.coef).max()),.2)
        z = absolute.transform(q,np.abs(r))
        np.testing.assert_allclose(absolute.energy(q,np.abs(r).mean(1),np.abs(r).var(1)),
                                   np.square(z).mean((1,2)),rtol=1e-12,atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
