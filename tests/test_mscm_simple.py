"""CPU structural checks before expensive normal-only screening."""
import unittest

import torch

from src.models.mscm_simple import SimpleMSCM


class SimpleMSCMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3407)

    def test_single_shared_bank_and_weak_decoder(self):
        model = SimpleMSCM('A3')
        x = torch.randn(3, 17, 7)
        h = model.encoder(x)
        y, a = model.decode(h)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(a.shape, (3, 17, 8))
        torch.testing.assert_close(a.sum(-1), torch.ones(3, 17))
        self.assertEqual(tuple(model.memory.prototypes.shape), (8, 32))
        self.assertFalse(any(isinstance(m, torch.nn.TransformerEncoder) for m in model.modules()))

    def test_capacity_control_and_common_initial_reconstruction(self):
        pre = SimpleMSCM('pre_multi')
        a3, a5 = SimpleMSCM('A3'), SimpleMSCM('A5')
        a3.load_pretrained(pre)
        a5.load_pretrained(pre)
        self.assertEqual(sum(p.numel() for p in a3.parameters()), sum(p.numel() for p in a5.parameters()))
        for m in [a3, a5]:
            m.configure_stage('adapt')
        self.assertEqual(sum(p.numel() for p in a3.parameters() if p.requires_grad),
                         sum(p.numel() for p in a5.parameters() if p.requires_grad))
        x = torch.randn(2, 13, 7)
        torch.testing.assert_close(a3(x), pre(x))
        torch.testing.assert_close(a5(x), pre(x))
        self.assertTrue(all(not p.requires_grad for p in a3.encoder.parameters()))

    def test_memory_effect_reaches_decoder_without_changing_query(self):
        model = SimpleMSCM('A3')
        h = torch.randn(2, 11, 32)
        original, _ = model.decode(h)
        replaced, _ = model.decode(h, reference=torch.zeros_like(h))
        self.assertGreater(float((original - replaced).abs().max()), 1e-7)

    def test_transformer_control_is_before_single_memory_and_frozen(self):
        model = SimpleMSCM('A3-T')
        self.assertEqual(len(model.encoder.temporal.layers), 2)
        self.assertEqual(tuple(model.memory.prototypes.shape), (8, 32))
        model.configure_stage('adapt')
        self.assertTrue(all(not p.requires_grad for p in model.encoder.temporal.parameters()))
        self.assertEqual(model(torch.randn(2, 15, 7)).shape, (2, 15, 7))

    def test_single_scale_transformer_memory_control(self):
        pretrained = SimpleMSCM('pre_single_T')
        model = SimpleMSCM('A2-T')
        model.load_pretrained(pretrained)
        self.assertEqual(model.encoder.kernels, (7,))
        self.assertEqual(len(model.encoder.temporal.layers), 2)
        self.assertEqual(tuple(model.memory.prototypes.shape), (8, 32))
        x = torch.randn(2, 15, 7)
        pretrained.eval()
        model.eval()
        torch.testing.assert_close(model(x), pretrained(x))
        model.configure_stage('adapt')
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertEqual(model(x).shape, x.shape)



if __name__ == '__main__':
    unittest.main()
