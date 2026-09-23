"""RCR stage 1 and Lite: reconstruction, control-only calibration, frozen relation.

Independent research implementation; does not alter existing model behavior.
All fitting APIs take normal training arrays only. Thresholds belong to the runner.
"""
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from sklearn.covariance import LedoitWolf


CONDITION_FEATURES = (
    'current_mean', 'current_std', 'current_range', 'current_abs_diff_mean',
    'soc_mean', 'soc_std', 'soc_range', 'soc_delta',
)


def condition_features(x, current_index, soc_index):
    """[B,L,D] -> [B,8]; never reads response channels. Input train-scaled."""
    if x.ndim != 3 or x.shape[1] < 2 or current_index == soc_index:
        raise ValueError('Expected windows of length >=2 and distinct control columns')
    current, soc = x[..., current_index], x[..., soc_index]
    return torch.stack((current.mean(1), current.std(1, unbiased=False),
                        current.amax(1)-current.amin(1), current.diff(dim=1).abs().mean(1),
                        soc.mean(1), soc.std(1, unbiased=False),
                        soc.amax(1)-soc.amin(1), soc[:, -1]-soc[:, 0]), dim=1)


class VariableGraph(nn.Module):
    """One global learned graph; the temporal dimension is never an attention axis."""
    def __init__(self, variables, hidden=32, node_dim=8):
        super().__init__()
        self.nodes = nn.Parameter(torch.randn(variables, node_dim)*0.1)
        self.message = nn.Linear(hidden, hidden, bias=False)

    def adjacency(self):
        return (self.nodes @ self.nodes.T / math.sqrt(self.nodes.shape[1])).softmax(-1)

    def forward(self, h):
        return h + torch.einsum('ij,...jh->...ih', self.adjacency(), self.message(h))


class RCRBackbone(nn.Module):
    """Variable embedding -> graph -> flattened-token projection -> temporal AE.

    Flatten/project preserves variable identity, unlike averaging node tokens.
    No raw-input shortcut, forecasting, variance head or self-conditioning.
    """
    def __init__(self, variables, hidden=32, feature_graph=True, temporal=True,
                 node_dim=8, layers=2, heads=4, ffn_dim=64, dropout=0.1):
        super().__init__()
        self.variables = variables
        self.value = nn.Linear(1, hidden)
        self.variable_embedding = nn.Parameter(torch.randn(variables, hidden)*0.02)
        self.graph = VariableGraph(variables, hidden, node_dim) if feature_graph else nn.Identity()
        self.fusion = nn.Linear(variables*hidden, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, ffn_dim, dropout,
                                          activation='gelu', batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False) if temporal else nn.Identity()
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, variables))

    def forward(self, x):
        if x.shape[-1] != self.variables:
            raise ValueError('Feature schema mismatch')
        h = self.value(x.unsqueeze(-1)) + self.variable_embedding
        h = self.fusion(self.graph(h).flatten(-2))
        position = torch.arange(h.shape[1], device=h.device, dtype=h.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, h.shape[-1], 2, device=h.device,
                                           dtype=h.dtype)*(-math.log(10000)/h.shape[-1]))
        pe = torch.zeros_like(h[0])
        pe[:, 0::2] = torch.sin(position*frequency)
        pe[:, 1::2] = torch.cos(position*frequency)
        return self.decoder(self.temporal(h+pe))


class LinearResidualCalibration:
    """OLS mean and log-RMS scale; only training-normal window moments are fitted.

    q is standardized using training statistics. Sigma clipping uses training
    predictions' P5/P95 with an absolute min_std floor. No response-dependent
    inference adaptation, validation fitting or test percentile clipping.
    """
    def __init__(self, min_std=0.05):
        self.min_std = float(min_std)

    def _design(self, q):
        return np.column_stack(((np.asarray(q, dtype=np.float64)-self.q_mean)/self.q_std,
                                np.ones(len(q))))

    def fit(self, q, residual):
        q, residual = np.asarray(q, dtype=np.float64), np.asarray(residual, dtype=np.float64)
        if residual.ndim != 3 or q.ndim != 2 or len(q) != len(residual):
            raise ValueError('Expected matching [N,Q] conditions and [N,L,D] residuals')
        if not np.isfinite(q).all() or not np.isfinite(residual).all():
            raise ValueError('Nonfinite normal fitting input')
        self.q_mean = q.mean(0)
        self.q_std = np.maximum(q.std(0), 1e-8)
        design = self._design(q)
        self.mean_coef = np.linalg.lstsq(design, residual.mean(1), rcond=None)[0]
        centered = residual-(design @ self.mean_coef)[:, None, :]
        target = np.log(np.maximum(np.sqrt(np.mean(centered**2, axis=1)), self.min_std))
        self.scale_coef = np.linalg.lstsq(design, target, rcond=None)[0]
        predicted = np.exp(np.clip(design @ self.scale_coef, -30, 30))
        self.scale_low = np.maximum(np.quantile(predicted, .05, axis=0), self.min_std)
        self.scale_high = np.maximum(np.quantile(predicted, .95, axis=0), self.scale_low)
        return self

    def parameters(self, q):
        design = self._design(q)
        mu = design @ self.mean_coef
        sigma = np.clip(np.exp(np.clip(design @ self.scale_coef, -30, 30)),
                        self.scale_low, self.scale_high)
        return mu, sigma

    def transform(self, q, residual):
        mu, sigma = self.parameters(q)
        return (np.asarray(residual)-mu[:, None, :])/sigma[:, None, :]

    def save(self, path):
        np.savez(path, min_std=self.min_std, q_mean=self.q_mean, q_std=self.q_std,
                 mean_coef=self.mean_coef, scale_coef=self.scale_coef,
                 scale_low=self.scale_low, scale_high=self.scale_high)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            obj = cls(float(data['min_std']))
            for name in data.files:
                if name != 'min_std':
                    setattr(obj, name, data[name].copy())
        return obj


class EmpiricalCDF:
    def fit(self, score):
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        if not len(score) or not np.isfinite(score).all():
            raise ValueError('CDF requires nonempty finite training-normal scores')
        self.reference = np.sort(score)
        return self

    def transform(self, score):
        return np.searchsorted(self.reference, score, side='right')/len(self.reference)


class RCRLite:
    """Same-window raw/magnitude/joint scoring followed by training-only CDF fusion."""
    def __init__(self, min_std=0.05, alpha=0.5):
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        self.calibration = LinearResidualCalibration(min_std)
        self.alpha = float(alpha)

    def fit(self, q, residual):
        self.calibration.fit(q, residual)
        z = self.calibration.transform(q, residual)
        fit = LedoitWolf().fit(z.reshape(-1, z.shape[-1]))
        self.covariance, self.precision = fit.covariance_, fit.precision_
        self.location, self.shrinkage = fit.location_, float(fit.shrinkage_)
        # Follow the specified z^T Omega z, not sklearn's recentered mahalanobis.
        mag, joint = self.components(z)
        self.mag_cdf = EmpiricalCDF().fit(mag)
        self.joint_cdf = EmpiricalCDF().fit(joint)
        return self

    def components(self, z):
        mag = np.square(z).mean(axis=(1, 2))
        joint = np.einsum('nld,de,nle->nl', z, self.precision, z).mean(1)/z.shape[-1]
        return mag, joint

    def score(self, q, residual, return_residual=False):
        z = self.calibration.transform(q, residual)
        mag, joint = self.components(z)
        pm, pj = self.mag_cdf.transform(mag), self.joint_cdf.transform(joint)
        result = dict(raw=np.square(residual).mean(axis=(1, 2)), magnitude=mag,
                      relation=joint, magnitude_percentile=pm, relation_percentile=pj,
                      combined=self.alpha*pm+(1-self.alpha)*pj)
        if return_residual:
            result.update(residual=residual, standardized_residual=z)
        return result

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.calibration.save(directory/'calibration.npz')
        np.savez(directory/'lite.npz', covariance=self.covariance, precision=self.precision,
                 location=self.location, shrinkage=self.shrinkage, alpha=self.alpha,
                 mag_reference=self.mag_cdf.reference, joint_reference=self.joint_cdf.reference)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with np.load(directory/'lite.npz', allow_pickle=False) as data:
            obj = cls(alpha=float(data['alpha']))
            for name in ['covariance', 'precision', 'location', 'shrinkage']:
                setattr(obj, name, data[name].copy())
            obj.mag_cdf = EmpiricalCDF().fit(data['mag_reference'])
            obj.joint_cdf = EmpiricalCDF().fit(data['joint_reference'])
        obj.calibration = LinearResidualCalibration.load(directory/'calibration.npz')
        return obj


class MaskedResidualGraphNet(nn.Module):
    """Predict residual coordinates, never a directly learned anomaly scalar.

    Each time point is an independent D-token graph. Values are masked BEFORE
    embedding/message passing, preventing target leakage through graph neighbors.
    Controls must be training-standardized fixed q, not response embeddings.
    """
    def __init__(self, variables, hidden=32, node_dim=8, condition_mode='mlp',
                 condition_dim=8, dropout=0.1):
        super().__init__()
        if condition_mode not in {'none', 'linear', 'mlp'}:
            raise ValueError(condition_mode)
        self.variables, self.condition_mode = variables, condition_mode
        self.value = nn.Linear(1, hidden)
        self.variable_embedding = nn.Parameter(torch.randn(variables, hidden)*.02)
        self.mask_embedding = nn.Parameter(torch.randn(hidden)*.02)
        self.graph = VariableGraph(variables, hidden, node_dim)
        if condition_mode == 'mlp':
            self.condition = nn.Sequential(nn.Linear(condition_dim, 16), nn.GELU(), nn.Linear(16, 16))
            self.film = nn.Linear(16, 2*hidden)
        elif condition_mode == 'linear':
            self.condition = nn.Identity()
            self.film = nn.Linear(condition_dim, 2*hidden)
        if condition_mode != 'none':
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        layer = nn.TransformerEncoderLayer(hidden, 2, 64, dropout, activation='gelu', batch_first=True)
        self.attention = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.output = nn.Linear(hidden, 1)

    def forward(self, z, q, mask):
        if z.ndim != 2 or mask.shape != z.shape or mask.dtype != torch.bool:
            raise ValueError('Expected residual points [N,D] and boolean same-shape mask')
        h = self.value(z.masked_fill(mask, 0).unsqueeze(-1)) + self.variable_embedding
        h = h + mask.unsqueeze(-1)*self.mask_embedding
        if self.condition_mode != 'none':
            gamma, beta = (.1*self.film(self.condition(q)).tanh()).chunk(2, dim=-1)
            h = (1+gamma[:, None])*h+beta[:, None]
        return self.output(self.attention(self.graph(h))).squeeze(-1)

    def masked_loss(self, z, q, mask_ratio=.3, generator=None):
        if not 0 < mask_ratio < 1:
            raise ValueError('mask_ratio must be in (0,1)')
        mask = torch.rand(z.shape, device=z.device, generator=generator) < mask_ratio
        empty = ~mask.any(1)
        # Avoid a zero-denominator batch without always exposing a fixed channel.
        chosen = torch.randint(self.variables, (int(empty.sum()),), device=z.device, generator=generator)
        mask[empty, chosen] = True
        prediction = self(z, q, mask)
        return (prediction-z.detach()).square()[mask].mean()

    @torch.no_grad()
    def leave_one_out(self, z, q):
        if self.training:
            raise RuntimeError('Call eval() before deterministic leave-one-out scoring')
        prediction = torch.empty_like(z)
        for i in range(self.variables):
            mask = torch.zeros_like(z, dtype=torch.bool)
            mask[:, i] = True
            prediction[:, i] = self(z, q, mask)[:, i]
        return prediction
