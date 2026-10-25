"""Normal-only reliability calibration and conditional leave-one-out relations.

These are minimal statistical mechanism models, not new neural architectures.
No fitting function accepts fault labels or test scores.
"""
import numpy as np
from sklearn.model_selection import KFold

from src.models.rcr_fitting import slices


FLOORS = (.05, .01, .001, .0001, .00001)


def fit_reliability(q, means, vehicles, groups, base, seed=3407):
    """Vehicle-held-out normal mean prediction determines a fixed g(c).

    Minimize normal cross-fitted squared error for a scalar shrinkage of mu.
    Each VIN has equal weight within a state; sparse states shrink to a global
    rule. An independent train-q P99 support rule attenuates extrapolation.
    """
    q = np.asarray(q, dtype=np.float64)
    design = base._design(q)
    vehicles = np.asarray(vehicles, dtype=str)
    unique = np.unique(vehicles)
    if len(unique) < 5:
        raise ValueError('Need at least five normal training vehicles')
    prediction = np.empty_like(means)
    folds = np.empty(len(q), dtype=np.int16)
    for fold, (_, held) in enumerate(KFold(5, shuffle=True, random_state=seed).split(unique)):
        mask = np.isin(vehicles, unique[held])
        coef = np.linalg.lstsq(design[~mask], means[~mask], rcond=None)[0]
        prediction[mask] = design[mask]@coef
        folds[mask] = fold
    numerator = (means*prediction).mean(1)
    denominator = np.square(prediction).mean(1)

    def estimate(mask):
        _, codes, counts = np.unique(vehicles[mask], return_inverse=True, return_counts=True)
        weight = 1/counts[codes]
        num, den = float(weight@numerator[mask]), float(weight@denominator[mask])
        return float(np.clip(num/max(den, 1e-30), 0, 1)), len(counts), num, den

    global_gate, _, _, _ = estimate(np.ones(len(q), dtype=bool))
    gates = np.full(int(groups.max())+1, global_gate)
    audit = []
    for group in range(len(gates)):
        mask = groups == group
        if not mask.any():
            audit.append(dict(group=group, vehicles=0, gate=global_gate, fallback=True))
            continue
        optimal, n, num, den = estimate(mask)
        gates[group] = (n*optimal+10*global_gate)/(n+10) if n >= 5 else global_gate
        audit.append(dict(group=group, vehicles=n, windows=int(mask.sum()),
            unshrunk_gate=optimal, gate=float(gates[group]), fallback=n < 5,
            normal_crossfit_e_mu=num, normal_crossfit_mu_square=den))
    radius = np.square((q-base.q_mean)/base.q_std).mean(1)
    return dict(gates=gates, global_gate=global_gate, support_p99=max(1e-12, float(np.quantile(radius, .99))),
                audit=audit, folds=folds)


class ReliableConditionalCalibration:
    def __init__(self, base, reliability, family, scale_spec):
        if family not in ('full', 'reliable'):
            raise ValueError('Unknown calibration family')
        self.q_mean, self.q_std = base.q_mean.copy(), base.q_std.copy()
        self.mean_coef = base.mean_coef.copy()
        self.gates = np.asarray(reliability['gates'])
        self.global_gate, self.support_p99 = reliability['global_gate'], reliability['support_p99']
        self.family, self.scale_spec = family, scale_spec

    def design(self, q):
        return np.column_stack(((np.asarray(q, dtype=np.float64)-self.q_mean)/self.q_std, np.ones(len(q))))

    def strength(self, q, groups):
        if self.family == 'full':
            return np.ones(len(q))
        gate = np.full(len(q), self.global_gate)
        known = (groups >= 0) & (groups < len(self.gates))
        gate[known] = self.gates[groups[known]]
        radius = np.square((np.asarray(q)-self.q_mean)/self.q_std).mean(1)
        return gate*np.minimum(1., self.support_p99/np.maximum(radius, 1e-30))

    def location(self, q, groups):
        return (self.design(q)@self.mean_coef)*self.strength(q, groups)[:, None]

    def fit_scale(self, q, means, variances, groups):
        rms = np.sqrt(np.maximum(variances+np.square(means-self.location(q, groups)), 0))
        if self.scale_spec == 'robust':
            self.min_std = 1e-5
            global_scale = np.maximum(np.median(rms, axis=0), self.min_std)
            self.state_scale = np.tile(global_scale, (len(self.gates), 1))
            for group in np.unique(groups):
                mask = groups == group
                if mask.sum() >= 50:
                    self.state_scale[group] = np.maximum(np.median(rms[mask], axis=0), self.min_std)
            self.fallback_scale = global_scale
        else:
            self.min_std = float(self.scale_spec)
            design = self.design(q)
            target = np.log(np.maximum(rms, self.min_std))
            self.scale_coef = np.linalg.lstsq(design, target, rcond=None)[0]
            predicted = np.exp(np.clip(design@self.scale_coef, -30, 30))
            self.scale_low = np.maximum(np.quantile(predicted, .05, axis=0), self.min_std)
            self.scale_high = np.maximum(np.quantile(predicted, .95, axis=0), self.scale_low)
        return self

    def parameters(self, q, groups):
        mean = self.location(q, groups)
        if self.scale_spec == 'robust':
            scale = np.tile(self.fallback_scale, (len(q), 1))
            known = (groups >= 0) & (groups < len(self.state_scale))
            scale[known] = self.state_scale[groups[known]]
        else:
            scale = np.clip(np.exp(np.clip(self.design(q)@self.scale_coef, -30, 30)), self.scale_low, self.scale_high)
        return mean, scale

    def energy(self, q, means, variances, groups):
        mean, scale = self.parameters(q, groups)
        return ((variances+np.square(means-mean))/np.square(scale)).mean(1)

    def normal_nll(self, q, means, variances, groups):
        mean, scale = self.parameters(q, groups)
        return float((np.log(scale)+.5*(variances+np.square(means-mean))/np.square(scale)).mean())

    def transform(self, q, residual, groups):
        mean, scale = self.parameters(q, groups)
        return (np.asarray(residual, dtype=np.float64)-mean[:, None, :])/scale[:, None, :]

    def save(self, path):
        np.savez(path, **{key: value for key, value in self.__dict__.items()})

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                value = data[key].copy()
                setattr(obj, key, value.item() if value.ndim == 0 else value)
        return obj


def augmented_second(z):
    """Per-window E[(1,z)(1,z)^T], preserving all within-window points."""
    n, length, d = z.shape
    result = np.empty((n, d+1, d+1), dtype=np.float64)
    result[:, 0, 0] = 1
    result[:, 0, 1:] = result[:, 1:, 0] = z.mean(1)
    result[:, 1:, 1:] = np.einsum('nli,nlj->nij', z, z)/length
    return result


def gram_from_moments(q_features, second):
    """Full-data Kronecker sufficient moments, without expanding time points."""
    c = np.column_stack((np.ones(len(q_features)), q_features))
    qq = np.einsum('na,nb->nab', c, c).reshape(len(c), -1)
    moment = qq.T@second.reshape(len(c), -1)
    p, d = c.shape[1], second.shape[1]
    return moment.reshape(p, p, d, d).transpose(0, 2, 1, 3).reshape(p*d, p*d)


class ConditionalLOORelation:
    """Ridge Gaussian P(z_i | z_-i,c), with c*z_-i interaction predictors.

    Every feature containing the target variable is excluded from its predictor.
    Normal validation NLL selects ridge strength; test labels are never used.
    """
    @staticmethod
    def condition(q, mean, std, low, high):
        return np.clip((np.asarray(q)-mean)/std, low, high)

    @classmethod
    def fit(cls, train_gram, val_gram, qdim, variables, q_stats, conditional):
        d, cdim = variables, qdim+1
        candidates, diagnostics = [], []
        target_indices = np.arange(1, d+1)
        y2 = np.diag(train_gram)[target_indices]
        for penalty in (1e-4, .01, 1.):
            coefficients = np.zeros((d, cdim, d+1))
            variances, losses = [], []
            for i in range(d):
                if conditional:
                    chosen = np.array([a*(d+1)+j for a in range(cdim) for j in range(d+1) if j != i+1])
                else:
                    chosen = np.array([j for j in range(d+1) if j != i+1])
                target = i+1
                gram = train_gram[np.ix_(chosen, chosen)]
                mean = gram[0, 1:]
                std = np.maximum(np.sqrt(np.maximum(np.diag(gram)[1:]-mean*mean, 0)), 1e-8)
                transform = np.eye(len(chosen))
                transform[1:, 1:] = np.diag(1/std)
                transform[0, 1:] = -mean/std
                ridge = np.eye(len(chosen))*penalty
                ridge[0, 0] = 0
                cross = train_gram[chosen, target]
                beta = transform@np.linalg.solve(transform.T@gram@transform+ridge, transform.T@cross)
                train_mse = float(y2[i]-2*beta@cross+beta@gram@beta)
                variance = max(train_mse, .05**2*y2[i], 1e-16)
                val_mse = float(val_gram[target, target]-2*beta@val_gram[chosen, target]+
                                beta@val_gram[np.ix_(chosen, chosen)]@beta)
                losses.append(.5*(np.log(variance)+max(0, val_mse)/variance))
                variances.append(variance)
                coefficients.reshape(d, -1)[i, chosen] = beta
            candidates.append((coefficients, np.asarray(variances)))
            diagnostics.append(dict(ridge=penalty, normal_validation_nll=float(np.mean(losses))))
        winner = int(np.argmin([v['normal_validation_nll'] for v in diagnostics]))
        obj = cls()
        obj.coefficients, obj.variances = candidates[winner]
        obj.q_mean, obj.q_std, obj.q_low, obj.q_high = q_stats
        obj.conditional, obj.ridge = bool(conditional), diagnostics[winner]['ridge']
        obj.selection = diagnostics
        obj.normal_z_mean = train_gram[0, target_indices]
        obj.normal_z_std = np.maximum(np.sqrt(np.maximum(y2-obj.normal_z_mean**2, 0)), 1e-8)
        return obj

    def condition_features(self, q):
        return self.condition(q, self.q_mean, self.q_std, self.q_low, self.q_high)

    def dynamic_coefficients(self, q):
        c = np.column_stack((np.ones(len(q)), self.condition_features(q)))
        return np.einsum('na,iaj->nij', c, self.coefficients)

    def score_moments(self, q, second):
        errors = -self.dynamic_coefficients(q)
        d = errors.shape[1]
        errors[:, np.arange(d), np.arange(d)+1] += 1
        mse = np.einsum('nij,njk,nik->ni', errors, second, errors)
        return (np.maximum(mse, 0)/self.variances).mean(1)

    def diagonal_score(self, second):
        mean = second[:, 0, 1:]
        square = np.diagonal(second[:, 1:, 1:], axis1=1, axis2=2)
        return ((square-2*mean*self.normal_z_mean+self.normal_z_mean**2)/self.normal_z_std**2).mean(1)

    def predict(self, q, z):
        coef = self.dynamic_coefficients(q)
        return coef[:, None, :, 0]+np.einsum('nlj,nij->nli', z, coef[:, :, 1:])

    def save(self, path):
        np.savez(path, coefficients=self.coefficients, variances=self.variances,
            q_mean=self.q_mean, q_std=self.q_std, q_low=self.q_low, q_high=self.q_high,
            conditional=self.conditional, ridge=self.ridge,
            normal_z_mean=self.normal_z_mean, normal_z_std=self.normal_z_std)

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                value = data[key].copy()
                setattr(obj, key, value.item() if value.ndim == 0 else value)
        return obj
