"""Independent equation checks; these do NOT assert executed TensorFlow parity."""
import unittest

import numpy as np
import torch

from src.models.omni_anomaly_port import (
    OmniAnomalyPort, OmniConfig, PlanarFlow, TF1Adam, TF1GRU,
    clip_gradients_per_parameter, lgssm_log_prob_steps,
)


def array(t):
    return t.detach().numpy()


def dense(layer, x):
    return x @ array(layer.weight).T + array(layer.bias)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class OmniPortTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3407)
        torch.set_num_threads(1)
        self.model = OmniAnomalyPort(OmniConfig(input_dim=2, window=4, latent_dim=3,
            rnn_hidden=5, dense_dim=6, flow_layers=2), semantics="release").double()

    def test_tf_gru_reset_before_numpy(self):
        gru = TF1GRU(2, 3).double()
        x = torch.randn(2, 4, 2, dtype=torch.float64)
        state = np.zeros((2, 3)); expected = []
        for xt in array(x).transpose(1, 0, 2):
            reset, update = np.split(sigmoid(dense(gru.gates,
                np.concatenate((xt, state), -1))), 2, -1)
            state = update * state + (1 - update) * np.tanh(dense(
                gru.candidate, np.concatenate((xt, reset * state), -1)))
            expected.append(state.copy())
        np.testing.assert_allclose(array(gru(x)), np.stack(expected, 1), atol=1e-12)

    def test_release_recurrent_samples_and_reversed_density_numpy(self):
        model = self.model
        h = torch.randn(2, 4, 6, dtype=torch.float64)
        noise = torch.randn(2, 4, 3, dtype=torch.float64)
        out = model.posterior(h, noise)
        previous = np.zeros((2, 3)); samples = []
        for ht, eps in zip(array(h).transpose(1, 0, 2), array(noise).transpose(1, 0, 2)):
            inputs = np.concatenate((ht, previous), -1)
            std = np.logaddexp(0, dense(model.q_std, inputs)) + 1e-4
            previous = dense(model.q_mean, inputs) + std * eps
            samples.append(previous.copy())
        z = np.stack(samples, 1)
        inputs = np.concatenate((z, array(h)), -1)
        mu = dense(model.q_mean, inputs)
        std = np.logaddexp(0, dense(model.q_std, inputs)) + 1e-4
        logq = (-.5*np.log(2*np.pi) - np.log(std) - .5*((z-mu)/std)**2).sum(-1)
        np.testing.assert_allclose(array(out["base_z"]), z, atol=1e-12)
        np.testing.assert_allclose(array(out["base_logq"]), logq, atol=1e-12)
        self.assertFalse(out["base_z"].requires_grad)
        self.assertFalse(out["base_logq"].requires_grad)

    def test_planar_logdet_against_full_jacobian(self):
        flow = PlanarFlow(3).double()
        point = torch.randn(3, dtype=torch.float64, requires_grad=True)
        jac = torch.autograd.functional.jacobian(lambda x: flow(x)[0], point)
        torch.testing.assert_close(flow(point)[1], torch.linalg.slogdet(jac).logabsdet,
                                   atol=1e-12, rtol=1e-10)

    def test_lgssm_against_dense_multivariate_gaussian(self):
        z = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
        t = torch.arange(5, dtype=z.dtype)
        covariance = torch.minimum(t[:, None], t[None, :]) + 1 + torch.eye(5)
        ref = torch.distributions.MultivariateNormal(torch.zeros(5, dtype=z.dtype), covariance)
        expected = ref.log_prob(z.transpose(1, 2)).sum(-1)
        actual = lgssm_log_prob_steps(z).sum(-1)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), z, retain_graph=True)[0],
                                   torch.autograd.grad(expected.sum(), z)[0])

    def test_release_loss_gradients_and_gaussian_score(self):
        model = self.model
        x = torch.randn(2, 4, 2, dtype=torch.float64)
        out = model(x, include_prior_diagnostic=True)
        px = (-.5*np.log(2*np.pi) - np.log(array(out["x_std"]))
              - .5*((array(x)-array(out["x_mean"]))/array(out["x_std"]))**2).sum(-1)
        expected = np.mean(array(out["base_logq"])-array(out["logdet"])-px)
        self.assertAlmostEqual(float(model.loss(out)), expected, places=10)
        changed_prior = {**out, "logpz": out["logpz"] + 1000}
        torch.testing.assert_close(model.loss(out), model.loss(changed_prior))
        np.testing.assert_allclose(array(model.anomaly_score(out)), -px[:, -1], atol=1e-12)
        model.loss(out).backward()
        for name, p in model.named_parameters():
            if name.startswith(("encoder.", "q_mean.", "q_std.")):
                self.assertIsNone(p.grad, name)
            else:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_release_noise_shared_truncated_and_reproducible(self):
        h = torch.zeros(3, 100, 6, dtype=torch.float64)
        a = self.model._noise(h, torch.Generator().manual_seed(123))
        b = self.model._noise(h, torch.Generator().manual_seed(123))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a[0], a[1], rtol=0, atol=0)
        self.assertTrue((a.abs() <= 2).all())
        self.assertGreater(float(a.std()), .5)

    def test_clip_each_tensor_not_global(self):
        model = torch.nn.Linear(1, 1).double()
        model.weight.grad = torch.full_like(model.weight, 20.)
        model.bias.grad = torch.full_like(model.bias, 20.)
        clip_gradients_per_parameter(model, 10.)
        self.assertEqual(float(model.weight.grad), 10.)
        self.assertEqual(float(model.bias.grad), 10.)

    def test_tf_adam_numpy_and_checkpoint_resume(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
        opt = TF1Adam([p], eps=1e-5)
        expected = array(p).copy(); m=np.zeros(2); v=np.zeros(2)
        for i in range(1, 4):
            g = np.array([1e-4*i, -.02*i])
            p.grad = torch.tensor(g)
            m = .9*m + .1*g; v = .999*v + .001*g*g
            expected -= .001*np.sqrt(1-.999**i)/(1-.9**i)*m/(np.sqrt(v)+1e-5)
            opt.step()
            np.testing.assert_allclose(array(p), expected, atol=1e-12)
        clone = TF1Adam([torch.nn.Parameter(p.detach().clone())])
        clone.load_state_dict(opt.state_dict())
        self.assertEqual(clone.param_groups[0]["tf_step"], 3)

    def test_no_implicit_alternate_model(self):
        with self.assertRaises(ValueError):
            OmniAnomalyPort(OmniConfig(), semantics="full_elbo")
        with self.assertRaises(ValueError):
            self.model(torch.zeros(2, 3, 2, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
