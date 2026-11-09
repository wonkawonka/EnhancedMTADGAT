"""Simplified MSCM V1: time-step shared Memory, concat and a weak decoder.

Independent of the historical patch/bounded-fusion MSCM experiments. No
Transformer follows retrieval; only normal reconstruction Huber is optimized.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.rcr import VariableGraph


class SimpleTemporalEncoder(nn.Module):
    def __init__(self, variables=7, hidden=32, multiscale=True, temporal=False):
        super().__init__()
        self.variables, self.hidden = variables, hidden
        self.value = nn.Linear(1, hidden)
        self.variable_embedding = nn.Parameter(torch.randn(variables, hidden) * .02)
        # Keep the same VariableGraph in every screening arm.
        self.graph = VariableGraph(variables, hidden)
        self.input_fusion = nn.Linear(variables * hidden, hidden)
        self.kernels = (3, 7, 15) if multiscale else (7,)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(hidden, hidden, k, padding=k // 2, groups=hidden),
                          nn.GELU(), nn.Conv1d(hidden, hidden, 1))
            for k in self.kernels])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in self.kernels])
        self.fusion = nn.Linear(hidden * len(self.kernels), hidden)
        self.temporal = nn.Identity()
        if temporal:
            layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=64,
                        dropout=.1, activation='gelu', batch_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.variables:
            raise ValueError('Expected [B,T,N] with the fixed variable schema')
        tokens = self.value(x.unsqueeze(-1)) + self.variable_embedding
        h = self.input_fusion(self.graph(tokens).flatten(-2))
        position = torch.arange(h.shape[1], device=h.device, dtype=h.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, self.hidden, 2, device=h.device,
                                           dtype=h.dtype) * (-math.log(10000) / self.hidden))
        pe = torch.zeros_like(h[0])
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
        h = h + pe
        scales = [norm(h + conv(h.transpose(1, 2)).transpose(1, 2))
                  for norm, conv in zip(self.norms, self.branches)]
        return self.temporal(self.fusion(torch.cat(scales, -1)))


class SharedNormalMemory(nn.Module):
    def __init__(self, hidden=32, slots=8, temperature=.25):
        super().__init__()
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.randn(slots, hidden) * .02)

    def forward(self, h):
        similarity = F.normalize(h, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
        weights = (similarity / self.temperature).softmax(-1)
        return weights @ self.prototypes, weights

    @torch.no_grad()
    def initialize(self, features, seed):
        from sklearn.cluster import MiniBatchKMeans
        values = np.asarray(features, dtype=np.float32)
        # Cluster directions with normalized dot-product retrieval. Keep center
        # magnitudes in the original encoded feature space for the decoder.
        directions = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)
        fit = MiniBatchKMeans(n_clusters=len(self.prototypes), random_state=seed,
                             n_init=3, batch_size=2048, max_iter=100).fit(directions)
        centers = np.stack([values[fit.labels_ == k].mean(0) if (fit.labels_ == k).any()
                            else values[np.argmax(directions @ fit.cluster_centers_[k])]
                            for k in range(len(self.prototypes))])
        self.prototypes.copy_(torch.as_tensor(centers, device=self.prototypes.device))


class SimpleMSCM(nn.Module):
    def __init__(self, arm='A1', variables=7, hidden=32, slots=8, temperature=.25):
        super().__init__()
        if arm not in ('A1', 'A2', 'A2-T', 'A3', 'A3-T', 'A5',
                       'pre_single', 'pre_single_T', 'pre_multi', 'pre_multi_T'):
            raise ValueError(arm)
        self.arm, self.hidden = arm, hidden
        self.encoder = SimpleTemporalEncoder(
            variables, hidden, arm not in ('A2', 'A2-T', 'pre_single', 'pre_single_T'),
            temporal=arm in ('A2-T', 'A3-T', 'pre_single_T', 'pre_multi_T'))
        self.memory = SharedNormalMemory(hidden, slots, temperature) if arm in ('A2', 'A2-T', 'A3', 'A3-T') else None
        self.ordinary = None
        if arm == 'A5':
            # Ordinary feature MLP: 2*d*(K/2) = K*d parameters, exactly matching
            # Memory. No retrieval, prototypes, PCA initialization or gate.
            self.ordinary = nn.Sequential(nn.Linear(hidden, slots // 2, bias=False),
                                          nn.GELU(), nn.Linear(slots // 2, hidden, bias=False))
        doubled = self.memory is not None or self.ordinary is not None
        self.decoder = nn.Sequential(nn.Linear(hidden * (2 if doubled else 1), hidden),
                                     nn.GELU(), nn.Linear(hidden, variables))

    def decode(self, h, reference=None):
        weights = None
        if self.memory is not None:
            retrieved, weights = self.memory(h)
            h = torch.cat((h, retrieved if reference is None else reference), -1)
        elif self.ordinary is not None:
            h = torch.cat((h, self.ordinary(h)), -1)
        return self.decoder(h), weights

    def forward(self, x):
        return self.decode(self.encoder(x))[0]

    def configure_stage(self, stage):
        for p in self.parameters():
            p.requires_grad_(True)
        if stage == 'adapt':
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        elif stage != 'pretrain':
            raise ValueError(stage)

    @torch.no_grad()
    def load_pretrained(self, pretrained):
        self.encoder.load_state_dict(pretrained.encoder.state_dict())
        self.decoder[0].weight.zero_()
        self.decoder[0].weight[:, :self.hidden].copy_(pretrained.decoder[0].weight)
        self.decoder[0].bias.copy_(pretrained.decoder[0].bias)
        self.decoder[2].load_state_dict(pretrained.decoder[2].state_dict())
