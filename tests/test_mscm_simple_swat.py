import unittest

import numpy as np
import torch

from src.models.mscm_simple import SimpleMSCM
from src.runners.run_mscm_simple_swat import actuator_context, temporal_fit_calibration


class SwatScreenTests(unittest.TestCase):
    def test_temporal_calibration_has_no_boundary_crossing_windows(self):
        starts = np.arange(0, 901, 10)
        fit, calibration, cut = temporal_fit_calibration(starts, 100, 1000)
        self.assertEqual(cut, 800)
        self.assertEqual(len(starts) - len(fit) - len(calibration), 9)
        self.assertLess(starts[fit[-1]] + 99, cut)
        self.assertGreaterEqual(starts[calibration[0]], cut)

    def test_actuator_context_ignores_sensor_response(self):
        x = torch.randn(2, 9, 5)
        before = actuator_context(x, [1, 3])
        x[:, :, [0, 2, 4]] += 1000
        torch.testing.assert_close(before, actuator_context(x, [1, 3]))
        self.assertEqual(before.shape, (2, 4))

    def test_forty_variable_decoder_and_capacity_control(self):
        torch.set_num_threads(2)
        a3, a5 = SimpleMSCM('A3', variables=40), SimpleMSCM('A5', variables=40)
        self.assertEqual(a3(torch.randn(2, 11, 40)).shape, (2, 11, 40))
        self.assertEqual(sum(p.numel() for p in a3.parameters()), sum(p.numel() for p in a5.parameters()))

    def test_forty_variable_single_scale_transformer_memory(self):
        model = SimpleMSCM('A2-T', variables=40)
        self.assertEqual(model.encoder.kernels, (7,))
        self.assertEqual(len(model.encoder.temporal.layers), 2)
        self.assertEqual(model(torch.randn(2, 11, 40)).shape, (2, 11, 40))


if __name__ == '__main__':
    unittest.main()
