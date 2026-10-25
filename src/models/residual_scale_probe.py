"""Counterfactual variance shrinkage for a frozen conditional mean model."""
import numpy as np


LAMBDAS = (0., .25, .5, .75, 1.)


def shrink_variance(conditional_scale, global_scale, strength):
    if not 0 <= strength <= 1:
        raise ValueError('Variance shrinkage strength must lie in [0,1]')
    return np.sqrt((1-strength)*np.square(global_scale)+strength*np.square(conditional_scale))


def centered_energy(means, variances, location):
    return np.maximum(variances+np.square(means-location), 0.)


def gaussian_nll(energy, scale):
    return (np.log(scale)+.5*energy/np.square(scale)).mean(axis=1)


def scale_components(conditional_scale, global_scale):
    """Separate a shared state scale from state-dependent variable weights."""
    ratio = conditional_scale/global_scale
    shared = np.exp(np.log(ratio).mean(axis=1, keepdims=True))
    return global_scale*shared, conditional_scale/shared


def select_normal_only(losses, vehicles, strengths):
    """VIN-equal normal-validation NLL; conservative one-SE scale choice.

    Within one SE of the minimum, prefer the smallest conditional strength.
    Family ties use normal-validation NLL, never detection labels.
    """
    names = list(losses)
    unique, codes = np.unique(np.asarray(vehicles, dtype=str), return_inverse=True)
    counts = np.bincount(codes)
    vehicle_loss = np.stack([np.bincount(codes, weights=losses[n])/counts for n in names])
    means = vehicle_loss.mean(axis=1)
    standard_errors = vehicle_loss.std(axis=1, ddof=1)/np.sqrt(len(unique))
    winner = int(np.argmin(means))
    eligible = np.flatnonzero(means <= means[winner]+standard_errors[winner]+1e-14)
    conservative = min(eligible, key=lambda j: (strengths[names[j]], means[j], names[j]))
    records = [dict(method=n, normal_validation_vin_nll=float(means[j]),
                    normal_validation_window_nll=float(np.mean(losses[n])),
                    vin_nll_se=float(standard_errors[j]), strength=float(strengths[n]),
                    within_one_se=bool(j in eligible)) for j, n in enumerate(names)]
    return names[winner], names[conservative], records
