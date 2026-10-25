"""MSCM V1: pre-Transformer multiscale Patch Memory and fixed postprocessing.

Architecture follows the user's 2026-10-25 development specification.
Memory training uses reconstruction plus a frozen normal-representation teacher.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.rcr import RCRBackbone


def patch_starts(length, patch, stride):
    patch = min(length, patch)
    starts = list(range(0, length-patch+1, min(stride, patch)))
    if starts[-1] != length-patch:
        starts.append(length-patch)
    return starts, patch


def patchify(h, patch=8, stride=4):
    starts, patch = patch_starts(h.shape[1], patch, stride)
    regular = h.unfold(1, patch, min(stride, patch)).mean(-1)
    if len(starts) != regular.shape[1]:
        regular = torch.cat((regular, h[:, -patch:].mean(1, keepdim=True)), 1)
    return regular


def overlap_average(patches, length, patch=8, stride=4):
    starts, patch = patch_starts(length, patch, stride)
    indices = torch.as_tensor(starts, device=patches.device)[:, None]+torch.arange(patch, device=patches.device)
    values = patches[:, :, None].expand(-1, -1, patch, -1).flatten(1, 2)
    output = patches.new_zeros((len(patches), length, patches.shape[-1]))
    output.index_add_(1, indices.flatten(), values)
    count = torch.bincount(indices.flatten(), minlength=length).to(patches.dtype)
    if torch.any(count == 0):
        raise ValueError('Patch reconstruction leaves uncovered time steps')
    return output/count[None, :, None]


class PatchMemory(nn.Module):
    def __init__(self, hidden=32, slots=8, temperature=.25, ordinary=False):
        super().__init__()
        self.slots, self.temperature, self.ordinary = slots, temperature, ordinary
        self.register_buffer('center', torch.zeros(hidden))
        self.register_buffer('scale', torch.ones(hidden))
        if ordinary:
            # 2*d*(K/2) parameters exactly match the K*d prototype dictionary.
            self.bottleneck = nn.Sequential(nn.Linear(hidden, slots//2, bias=False),
                nn.GELU(), nn.Linear(slots//2, hidden, bias=False))
        else:
            self.prototypes = nn.Parameter(torch.randn(slots, hidden)*.02)

    def initialize(self, features, seed=3407):
        from sklearn.cluster import MiniBatchKMeans
        values = np.asarray(features, dtype=np.float64)
        center, scale = values.mean(0), np.maximum(values.std(0), 1e-4)
        standardized = (values-center)/scale
        with torch.no_grad():
            self.center.copy_(torch.as_tensor(center, device=self.center.device))
            self.scale.copy_(torch.as_tensor(scale, device=self.scale.device))
            if self.ordinary:
                _, _, vectors = np.linalg.svd(standardized, full_matrices=False)
                rank = self.slots//2
                self.bottleneck[0].weight.copy_(torch.as_tensor(vectors[:rank], device=self.center.device))
                self.bottleneck[2].weight.copy_(torch.as_tensor(vectors[:rank].T, device=self.center.device))
            else:
                clusters = MiniBatchKMeans(self.slots, random_state=seed, n_init=3,
                    batch_size=1024, max_iter=100).fit(standardized)
                self.prototypes.copy_(torch.as_tensor(clusters.cluster_centers_, device=self.center.device))

    def forward(self, query):
        q = (query-self.center)/self.scale
        if self.ordinary:
            retrieved = self.bottleneck(q)
            weights = None
        else:
            weights = ((q@self.prototypes.T)/(self.temperature*np.sqrt(q.shape[-1]))).softmax(-1)
            retrieved = weights@self.prototypes
        return retrieved*self.scale+self.center, retrieved, weights


class MSCMReconstructor(nn.Module):
    def __init__(self, variant, variables=7, hidden=32, slots=8, patch=8, stride=4,
                 gamma=.1, fusion='bounded'):
        super().__init__()
        if variant not in ['A0', 'A1', 'A2', 'A4', 'A6']:
            raise ValueError(variant)
        if fusion not in ['bounded', 'pure', 'direct']:
            raise ValueError(fusion)
        self.variant, self.variables = variant, variables
        self.patch, self.stride, self.gamma, self.fusion_mode = patch, stride, gamma, fusion
        self.multiscale = variant in ['A1', 'A4', 'A6']
        self.use_memory = variant in ['A2', 'A4', 'A6']
        self.backbone = RCRBackbone(variables, hidden=hidden)
        self.kernels = (3, 7, 15) if self.multiscale else (1,)
        self.convolutions = nn.ModuleList()
        self.norms = nn.ModuleList()
        if self.multiscale:
            for k in self.kernels:
                branch = nn.Sequential(nn.Conv1d(hidden, hidden, k, padding=k//2, groups=hidden),
                    nn.GELU(), nn.Conv1d(hidden, hidden, 1))
                # Start as a small residual correction of the pretrained features.
                nn.init.zeros_(branch[-1].weight); nn.init.zeros_(branch[-1].bias)
                self.convolutions.append(branch)
                self.norms.append(nn.LayerNorm(hidden))
        self.projections = nn.ModuleList([nn.Linear(hidden, hidden) for _ in self.kernels])
        for projection in self.projections:
            nn.init.eye_(projection.weight); nn.init.zeros_(projection.bias)
        self.fuse = nn.Linear(hidden*len(self.kernels), hidden)
        with torch.no_grad():
            self.fuse.weight.zero_(); self.fuse.bias.zero_()
            for i in range(len(self.kernels)):
                self.fuse.weight[:,i*hidden:(i+1)*hidden].copy_(torch.eye(hidden)/len(self.kernels))
        self.compensation = nn.ModuleList([nn.Linear(hidden, hidden) for _ in self.kernels])
        self.memories = nn.ModuleList([PatchMemory(hidden, slots, ordinary=variant=='A6')
            for _ in self.kernels]) if self.use_memory else nn.ModuleList()

    def warm_start(self, state):
        self.backbone.load_state_dict(state, strict=True)

    def features(self, x):
        h = self.backbone.embed(x)
        if not self.multiscale:
            return [h]
        return [norm(h+conv(h.transpose(1,2)).transpose(1,2))
                for conv, norm in zip(self.convolutions, self.norms)]

    def patches(self, x):
        return [patchify(projection(h), self.patch, self.stride)
                for h, projection in zip(self.features(x), self.projections)]

    def configure_stage(self, stage, finetune_encoder=False):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == 'pretrain':
            modules = [self.backbone.temporal, self.backbone.decoder, self.fuse,
                       self.convolutions, self.norms, self.projections]
        elif stage == 'memory':
            modules = [self.memories, self.fuse, self.compensation, self.backbone.decoder]
            if finetune_encoder:
                modules += [self.convolutions, self.norms, self.projections]
        else:
            raise ValueError(stage)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def forward(self, x, details=False, teacher_patches=None, bypass_memory=False):
        hs = self.features(x)
        retrieved_patches, weights, fused_scales = [], [], []
        teacher_loss = x.new_zeros(())
        if self.use_memory and not bypass_memory:
            for s, (h, projection, memory, compensation) in enumerate(zip(
                    hs, self.projections, self.memories, self.compensation)):
                query = patchify(projection(h), self.patch, self.stride)
                retrieved, normalized, w = memory(query)
                if teacher_patches is not None:
                    target = (teacher_patches[s].detach()-memory.center)/memory.scale
                    # User specification: sum over feature coordinates, mean
                    # over batch/patches, then sum the scale losses.
                    teacher_loss = teacher_loss+(normalized-target).square().sum(-1).mean()
                restored = overlap_average(retrieved, h.shape[1], self.patch, self.stride)
                if self.fusion_mode == 'bounded':
                    fused = restored+self.gamma*torch.tanh(compensation(h))
                elif self.fusion_mode == 'direct':
                    fused = .5*(restored+h)
                else:
                    fused = restored
                fused_scales.append(fused); weights.append(w); retrieved_patches.append(normalized)
        else:
            fused_scales = [projection(h) for projection, h in zip(self.projections, hs)]
        if self.variant == 'A0':
            fused = hs[0]
        else:
            fused = self.fuse(torch.cat(fused_scales, -1))
        reconstruction = self.backbone.decoder(self.backbone.encode_temporal(fused))
        if details:
            states = [w.mean(1) for w in weights if w is not None]
            state = torch.cat(states, -1) if states else x.new_zeros((len(x),0))
            return reconstruction, teacher_loss, state, weights
        return reconstruction

    @torch.no_grad()
    def masked_memory_condition(self, x, neutral):
        conditions = []
        for j in range(self.variables):
            masked = x.clone(); masked[:,:,j] = neutral[j]
            _, _, state, _ = self(masked, details=True)
            conditions.append(state)
        return torch.stack(conditions, 1)


class FixedRobustResidualCalibration:
    """OLS conditional mean plus fixed median normal-window centered RMS."""
    def fit(self, q, means, variances):
        self.q_mean = np.mean(q, 0)
        self.q_std = np.maximum(np.std(q, 0), 1e-8)
        self.coef = np.linalg.lstsq(self.design(q), means, rcond=None)[0]
        energy = variances+np.square(means-self.location(q))
        self.scale = np.maximum(np.median(np.sqrt(np.maximum(energy, 0)), 0), 1e-5)
        return self

    def design(self, q):
        return np.column_stack(((q-self.q_mean)/self.q_std, np.ones(len(q))))

    def location(self, q):
        return self.design(q)@self.coef

    def transform(self, q, r):
        return (r-self.location(q)[:,None])/self.scale

    def energy(self, q, mean, variance):
        return ((variance+np.square(mean-self.location(q)))/self.scale**2).mean(1)

    def save(self, path):
        np.savez(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path) as values:
            obj.__dict__.update({k:values[k].copy() for k in values.files})
        return obj
