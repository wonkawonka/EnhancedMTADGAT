"""Auditable PyTorch port of OmniAnomaly's connected stochastic model.

Reference: NetManAIOps/OmniAnomaly@7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4,
model.py, recurrent_distribution.py and wrapper.py; MIT license, copyright
(c) 2021 NetManAIOps-OmniAnomaly. Original license is retained under
external_baselines/reference_snapshots/omni/LICENSE.

This is a project port, NOT the author's executable or a proven TF equivalent.
The caller MUST select release semantics; no existing runner uses this implicitly.

release: reproduce inspected release's stopped posterior gradients, shared
truncated noise, current-z/reversed-input density, and omitted prior loss.
User decision: reproduce the author release, do not repair its objective.
Retains TF1 reset-before GRU equations, TWO LINEAR dense layers
(no added ReLU), 20 planar flows, Gaussian likelihood and last-point NLL.
Only the default single latent sample is supported. Cross-framework RNG,
and TF execution parity remain unverified. TF1Adam below preserves the epsilon
placement of tf.train.AdamOptimizer instead of silently using torch Adam's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OmniConfig:
    input_dim: int = 7
    window: int = 100
    latent_dim: int = 3
    rnn_hidden: int = 500
    dense_dim: int = 500
    flow_layers: int = 20
    std_epsilon: float = 1e-4

    def __post_init__(self):
        for name in ("input_dim", "window", "latent_dim", "rnn_hidden",
                     "dense_dim", "flow_layers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.std_epsilon <= 0:
            raise ValueError("std_epsilon must be positive")


class TF1GRU(nn.Module):
    """TF1 GRUCell, not torch.GRU (which applies reset AFTER hidden matmul)."""

    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Linear(input_dim + hidden, 2 * hidden)
        self.candidate = nn.Linear(input_dim + hidden, hidden)
        nn.init.xavier_uniform_(self.gates.weight)
        nn.init.ones_(self.gates.bias)
        nn.init.xavier_uniform_(self.candidate.weight)
        nn.init.zeros_(self.candidate.bias)

    def step(self, x: Tensor, state: Tensor) -> Tensor:
        reset, update = self.gates(torch.cat((x, state), -1)).sigmoid().chunk(2, -1)
        candidate = self.candidate(torch.cat((x, reset * state), -1)).tanh()
        return update * state + (1 - update) * candidate

    def forward(self, x: Tensor) -> Tensor:
        state = x.new_zeros(x.shape[0], self.hidden)
        outputs = []
        for xt in x.unbind(1):
            state = self.step(xt, state)
            outputs.append(state)
        return torch.stack(outputs, 1)


class FeatureNet(nn.Module):
    def __init__(self, input_dim: int, config: OmniConfig):
        super().__init__()
        self.gru = TF1GRU(input_dim, config.rnn_hidden)
        # tf.layers.dense defaults to activation=None in the reference.
        self.dense = nn.Sequential(nn.Linear(config.rnn_hidden, config.dense_dim),
                                   nn.Linear(config.dense_dim, config.dense_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.dense(self.gru(x))


class PlanarFlow(nn.Module):
    """Planar transform and invertibility correction from tfsnippet alpha1."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * .01)
        self.u = nn.Parameter(torch.randn(dim) * .01)
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        wu = (self.w * self.u).sum()
        # No ad-hoc logdet clipping: invalid numerical states must fail checks.
        uhat = self.u + (-1 + F.softplus(wu) - wu) * self.w / self.w.square().sum()
        activation = (z @ self.w + self.b).tanh()
        y = z + activation.unsqueeze(-1) * uhat
        logdet = (1 + (1 - activation.square()) * (self.w * uhat).sum()).abs().log()
        return y, logdet


def lgssm_log_prob_steps(z: Tensor) -> Tensor:
    """Kalman innovation log densities, B,T; identity F/H, Q=R=P0=I.

    This integrates out the LGSSM state; it is NOT just N(z_t; z_{t-1}, I).
    Each latent dimension is independent under this particular fixed prior.
    """
    state_mean = torch.zeros_like(z[:, 0])
    state_variance = z.new_ones(())
    result = []
    for observation in z.unbind(1):
        innovation_variance = state_variance + 1
        result.append(torch.distributions.Normal(
            state_mean, innovation_variance.sqrt()).log_prob(observation).sum(-1))
        gain = state_variance / innovation_variance
        state_mean = state_mean + gain * (observation - state_mean)
        state_variance = (1 - gain) * state_variance + 1
    return torch.stack(result, 1)


class OmniAnomalyPort(nn.Module):
    def __init__(self, config: OmniConfig, *, semantics: Literal["release"]):
        super().__init__()
        if semantics != "release":
            raise ValueError("Only the user-approved semantics='release' is supported")
        self.config = config
        self.semantics = semantics
        self.encoder = FeatureNet(config.input_dim, config)
        self.q_mean = nn.Linear(config.dense_dim + config.latent_dim, config.latent_dim)
        self.q_std = nn.Linear(config.dense_dim + config.latent_dim, config.latent_dim)
        self.flows = nn.ModuleList(PlanarFlow(config.latent_dim) for _ in range(config.flow_layers))
        self.decoder = FeatureNet(config.latent_dim, config)
        self.x_mean = nn.Linear(config.dense_dim, config.input_dim)
        self.x_std = nn.Linear(config.dense_dim, config.input_dim)
        # Match TF dense default initializers; do not overwrite GRU gate bias=1.
        for module in (self.encoder.dense, self.q_mean, self.q_std,
                       self.decoder.dense, self.x_mean, self.x_std):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _params(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        return self.q_mean(inputs), F.softplus(self.q_std(inputs)) + self.config.std_epsilon

    def _noise(self, h: Tensor, generator=None) -> Tensor:
        shape = (h.shape[0], h.shape[1], self.config.latent_dim)
        # Release tf.truncated_normal: one noise vector per time, shared by batch.
        noise = h.new_empty(1, h.shape[1], self.config.latent_dim)
        nn.init.trunc_normal_(noise, std=1., a=-2., b=2., generator=generator)
        return noise.expand(shape)

    def posterior(self, h: Tensor, noise: Tensor | None = None, *, generator=None):
        if noise is None:
            noise = self._noise(h, generator)
        if noise.shape != (h.shape[0], h.shape[1], self.config.latent_dim):
            raise ValueError("noise must have shape [batch, time, latent]")
        previous = h.new_zeros(h.shape[0], self.config.latent_dim)
        samples = []
        with torch.no_grad():
            for ht, eps in zip(h.unbind(1), noise.unbind(1)):
                mu, std = self._params(torch.cat((ht, previous), -1))
                sample = mu + std * eps
                samples.append(sample)
                previous = sample
            z0 = torch.stack(samples, 1)
            # Source scan(back_prop=False); log_prob_step also uses [current_z,h],
            # whereas sample_step used [h,previous_z]. Preserve, do not hide this.
            z0 = z0.detach()
            mu, std = self._params(torch.cat((z0, h), -1))
            distance = (z0 - mu).abs().clamp_max(1e8)
            base_logq = (-.9189385332046727 - std.log()
                         - .5 * distance.square() / std.square()).sum(-1).detach()
        z, logdet = z0, torch.zeros_like(base_logq)
        for flow in self.flows:
            z, increment = flow(z)
            logdet = logdet + increment
        return {"base_z": z0, "z": z, "base_logq": base_logq,
                "logdet": logdet, "logq": base_logq - logdet}

    def forward(self, x: Tensor, *, noise: Tensor | None = None, generator=None,
                include_prior_diagnostic: bool = False):
        if x.ndim != 3 or x.shape[1:] != (self.config.window, self.config.input_dim):
            raise ValueError("Expected [batch, configured window, input_dim]")
        # Stopped release posterior is evaluated without retaining useless graphs.
        with torch.no_grad():
            h = self.encoder(x)
        posterior = self.posterior(h, noise, generator=generator)
        hidden = self.decoder(posterior["z"])
        mean = self.x_mean(hidden)
        std = F.softplus(self.x_std(hidden)) + self.config.std_epsilon
        logpx_dim = torch.distributions.Normal(mean, std).log_prob(x)
        result = {**posterior, "x_mean": mean, "x_std": std, "logpx_dim": logpx_dim,
                  "logpx": logpx_dim.sum(-1)}
        # TF builds the prior but the fetched release loss does not depend on it.
        if include_prior_diagnostic:
            result["logpz"] = lgssm_log_prob_steps(posterior["z"])
        return result

    def loss(self, outputs: dict[str, Tensor]) -> Tensor:
        terms = outputs["logq"] - outputs["logpx"]
        return terms.mean()

    @staticmethod
    def anomaly_score(outputs: dict[str, Tensor], *, last_point_only=True) -> Tensor:
        # Negative Gaussian reconstruction log probability, not MSE or ELBO.
        score = -outputs["logpx"]
        return score[:, -1] if last_point_only else score


def clip_gradients_per_parameter(model: nn.Module, max_norm: float = 10.) -> None:
    """Author training.py clips EACH tensor; a global clip is not equivalent."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    for parameter in model.parameters():
        if parameter.grad is not None:
            norm = parameter.grad.norm()
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient norm")
            parameter.grad.mul_(max_norm / norm.clamp_min(max_norm))


class TF1Adam(torch.optim.Optimizer):
    """Dense TF1 Adam update, no weight decay; unused gradients remain unused.

    TF1: lr_t=lr*sqrt(1-beta2**t)/(1-beta1**t); delta=lr_t*m/(sqrt(v)+eps).
    PyTorch default Adam places epsilon after bias-correcting sqrt(v).
    """

    def __init__(self, params, lr=1e-3, betas=(.9, .999), eps=1e-8):
        if lr <= 0 or eps <= 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("Invalid Adam parameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, tf_step=0))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group["tf_step"] += 1
            t = group["tf_step"]
            b1, b2 = group["betas"]
            rate = group["lr"] * (1 - b2 ** t) ** .5 / (1 - b1 ** t)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError("Only dense Omni gradients are supported")
                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                p.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-rate)
        return loss
