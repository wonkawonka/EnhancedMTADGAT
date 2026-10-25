"""Fixed normal-state memory, queried by training-standardized controls only.

No residual/response channel participates in routing. This is a calibration
replacement for a frozen backbone, not a reproduction of MEMTO.
"""
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from src.models.rcr_fitting import slices


def residual_moments(residual, batch=4096):
    """Keep only per-window mean and within-window variance in memory."""
    if residual.ndim != 3 or not len(residual):
        raise ValueError('Expected nonempty [windows,time,variables] residuals')
    means = np.empty((len(residual), residual.shape[-1]), dtype=np.float64)
    variances = np.empty_like(means)
    for ix in slices(len(residual), batch):
        block = np.asarray(residual[ix], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite normal training residuals')
        means[ix] = block.mean(1)
        variances[ix] = np.square(block-means[ix, None, :]).mean(1)
    return means, variances


class PrototypeResidualCalibration:
    """Hard K-means query with a constant mean/log-RMS model per prototype.

    For each cluster, use the intercept-only counterpart of the existing RCR
    estimator: mean(window residual mean), exp(mean(log(window centered RMS))).
    Thus K=1 is a global normal calibration. Long sequences retain their native
    window weighting, exactly as in the current RCR calibration.
    """
    def __init__(self, k, min_std=.05, seed=3407, n_init=10, max_iter=200):
        if k < 1 or min_std <= 0 or n_init < 1 or max_iter < 1:
            raise ValueError('Invalid prototype calibration settings')
        self.k, self.min_std, self.seed = int(k), float(min_std), int(seed)
        self.n_init, self.max_iter = int(n_init), int(max_iter)

    def standardize(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != len(self.q_mean) or not np.isfinite(q).all():
            raise ValueError('Expected finite aligned control features')
        return (q-self.q_mean)/self.q_std

    def fit(self, q, residual=None, moments=None):
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or len(q) < self.k or not np.isfinite(q).all():
            raise ValueError('Insufficient or nonfinite normal control features')
        means, variances = residual_moments(residual) if moments is None else moments
        if means.ndim != 2 or means.shape != variances.shape or len(means) != len(q):
            raise ValueError('Misaligned residual moments')
        if not np.isfinite(means).all() or not np.isfinite(variances).all() or (variances < 0).any():
            raise ValueError('Invalid normal residual moments')
        self.q_mean, self.q_std = q.mean(0), np.maximum(q.std(0), 1e-8)
        zq = self.standardize(q)
        fit = KMeans(n_clusters=self.k, random_state=self.seed, n_init=self.n_init,
                     max_iter=self.max_iter, algorithm='lloyd').fit(zq)
        self.centers = fit.cluster_centers_
        labels = fit.labels_
        self.counts = np.bincount(labels, minlength=self.k)
        if (self.counts == 0).any():
            raise ValueError('Empty normal-state prototypes; reduce K')
        self.inertia, self.n_iter = float(fit.inertia_), int(fit.n_iter_)
        self.mean = np.stack([np.bincount(labels, weights=means[:, j], minlength=self.k)
                              for j in range(means.shape[1])], axis=1)/self.counts[:, None]
        log_scale = np.zeros_like(self.mean)
        self.floor_counts = np.zeros_like(self.mean, dtype=np.int64)
        for ix in slices(len(q)):
            cluster = labels[ix]
            rms = np.sqrt(variances[ix]+np.square(means[ix]-self.mean[cluster]))
            target = np.log(np.maximum(rms, self.min_std))
            for j in range(means.shape[1]):
                log_scale[:, j] += np.bincount(cluster, weights=target[:, j], minlength=self.k)
                self.floor_counts[:, j] += np.bincount(cluster, weights=rms[:, j] <= self.min_std,
                                                      minlength=self.k).astype(np.int64)
        self.scale = np.maximum(np.exp(log_scale/self.counts[:, None]), self.min_std)
        return self

    def query_standardized(self, q):
        # Direct differences avoid catastrophic cancellation near a center.
        distance = np.sqrt(np.square(q[:, None, :]-self.centers[None, :, :]).sum(-1))
        labels = distance.argmin(1)
        nearest = distance[np.arange(len(q)), labels]
        if self.k == 1:
            margin = np.ones(len(q))  # No competing prototype.
        else:
            second = np.partition(distance, 1, axis=1)[:, 1]
            margin = (second-nearest)/np.maximum(second, 1e-12)
        return labels, nearest, margin

    def query(self, q):
        # Keep full-cache queries bounded even for multi-million-window runs.
        labels = np.empty(len(q), dtype=np.int64)
        distance, margin = np.empty(len(q)), np.empty(len(q))
        for ix in slices(len(q)):
            labels[ix], distance[ix], margin[ix] = self.query_standardized(self.standardize(q[ix]))
        return labels, distance, margin

    def parameters(self, q):
        labels, _, _ = self.query(q)
        return self.mean[labels], self.scale[labels]

    def transform(self, q, residual):
        mean, scale = self.parameters(q)
        return (np.asarray(residual, dtype=np.float64)-mean[:, None, :])/scale[:, None, :]

    def save(self, path):
        np.savez(Path(path), k=self.k, min_std=self.min_std, seed=self.seed,
                 n_init=self.n_init, max_iter=self.max_iter, q_mean=self.q_mean,
                 q_std=self.q_std, centers=self.centers, counts=self.counts,
                 mean=self.mean, scale=self.scale, floor_counts=self.floor_counts,
                 inertia=self.inertia, n_iter=self.n_iter)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            obj = cls(int(data['k']), float(data['min_std']), int(data['seed']),
                      int(data['n_init']), int(data['max_iter']))
            for name in ('q_mean', 'q_std', 'centers', 'counts', 'mean', 'scale', 'floor_counts'):
                setattr(obj, name, data[name].copy())
            obj.inertia, obj.n_iter = float(data['inertia']), int(data['n_iter'])
        return obj
