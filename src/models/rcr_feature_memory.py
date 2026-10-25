"""Normal memory in frozen Encoder feature space, independent of residuals."""
import numpy as np
import torch
from sklearn.cluster import KMeans

from src.models.rcr_fitting import slices


@torch.inference_mode()
def frozen_encoder_output(model, x):
    """Capture H immediately before the existing decoder, without editing it.

    H is temporal-Encoder output [batch,time,hidden]. Neither residuals nor
    calibration parameters are inputs. The returned reconstruction only audits
    that extraction preserved the original RCR computation.
    """
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Feature extraction requires an eval-mode frozen Backbone')
    outputs = []
    handle = model.temporal.register_forward_hook(lambda module, inputs, output: outputs.append(output))
    try:
        reconstruction = model(x)
    finally:
        handle.remove()
    if len(outputs) != 1 or outputs[0].ndim != 3:
        raise ValueError('Expected one [batch,time,hidden] Encoder output')
    return outputs[0], reconstruction


class EncoderFeatureMemory:
    """K-means on normal mean-pooled H; score = nearest standardized distance.

    This is an initial MEMTO-inspired representation probe, not its learned
    memory-constrained reconstruction architecture. No residual is accepted.
    """
    def __init__(self, k, seed=3407, n_init=5, max_iter=200):
        if min(k, n_init, max_iter) < 1:
            raise ValueError('Positive memory settings required')
        self.k, self.seed, self.n_init, self.max_iter = int(k), int(seed), int(n_init), int(max_iter)

    def standardize(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.feature_mean) or not np.isfinite(features).all():
            raise ValueError('Expected finite pooled Encoder features')
        return (features-self.feature_mean)/self.feature_std

    def fit(self, normal_features):
        h = np.asarray(normal_features, dtype=np.float64)
        if h.ndim != 2 or len(h) < self.k or not np.isfinite(h).all():
            raise ValueError('Insufficient or nonfinite normal Encoder features')
        self.feature_mean, self.feature_std = h.mean(0), np.maximum(h.std(0), 1e-8)
        fit = KMeans(n_clusters=self.k, random_state=self.seed, n_init=self.n_init,
                     max_iter=self.max_iter, algorithm='lloyd').fit(self.standardize(h))
        self.centers, self.counts = fit.cluster_centers_, np.bincount(fit.labels_, minlength=self.k)
        if (self.counts == 0).any():
            raise ValueError('Empty feature-memory slot; reduce K')
        self.inertia, self.n_iter = float(fit.inertia_), int(fit.n_iter_)
        return self

    def query_standardized(self, features):
        # Mean squared distance keeps the score's units independent of H width.
        distance = np.square(features[:, None, :]-self.centers[None, :, :]).mean(-1)
        labels = distance.argmin(1)
        nearest = distance[np.arange(len(features)), labels]
        if self.k == 1:
            margin = np.ones(len(features))
        else:
            second = np.partition(distance, 1, axis=1)[:, 1]
            margin = (second-nearest)/np.maximum(second, 1e-12)
        return labels, nearest, margin

    def query(self, features):
        labels, distances, margins = np.empty(len(features), dtype=np.int16), np.empty(len(features)), np.empty(len(features))
        for ix in slices(len(features)):
            labels[ix], distances[ix], margins[ix] = self.query_standardized(self.standardize(features[ix]))
        return labels, distances, margins

    def score(self, features):
        return self.query(features)[1]

    def save(self, path):
        np.savez(path, k=self.k, seed=self.seed, n_init=self.n_init, max_iter=self.max_iter,
            feature_mean=self.feature_mean, feature_std=self.feature_std, centers=self.centers,
            counts=self.counts, inertia=self.inertia, n_iter=self.n_iter)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            obj = cls(int(data['k']), int(data['seed']), int(data['n_init']), int(data['max_iter']))
            for name in ['feature_mean', 'feature_std', 'centers', 'counts']:
                setattr(obj, name, data[name].copy())
            obj.inertia, obj.n_iter = float(data['inertia']), int(data['n_iter'])
        return obj
