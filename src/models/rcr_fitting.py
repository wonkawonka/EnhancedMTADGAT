"""Memory-bounded, full-data RCR-Lite fitting. No statistical subsampling."""
import numpy as np
from scipy.linalg import pinvh
from src.models.rcr import RCRLite, EmpiricalCDF


def slices(length, batch=4096):
    for start in range(0, length, batch):
        yield slice(start, min(start+batch, length))


def fit_lite_streaming(q, residual, min_std=.05, batch=4096):
    """Same estimator as RCRLite.fit, using sufficient moments for Ledoit-Wolf.

    Small window-level design/target arrays remain in RAM. The much larger
    [windows,time,variables] residual and z arrays are read in bounded blocks.
    Tests compare covariance, shrinkage, precision and scores with sklearn.
    """
    n, length, d = residual.shape
    if n < 2 or len(q) != n:
        raise ValueError('Need at least two aligned normal windows')
    q = np.asarray(q, dtype=np.float64)
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite training controls')
    lite = RCRLite(min_std=min_std)
    cal = lite.calibration
    cal.q_mean, cal.q_std = q.mean(0), np.maximum(q.std(0), 1e-8)
    design = cal._design(q)
    means = np.empty((n, d), dtype=np.float64)
    for ix in slices(n, batch):
        block = np.asarray(residual[ix], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite training residuals')
        means[ix] = block.mean(1)
    cal.mean_coef = np.linalg.lstsq(design, means, rcond=None)[0]
    for ix in slices(n, batch):
        centered = np.asarray(residual[ix], dtype=np.float64)-(design[ix] @ cal.mean_coef)[:, None]
        means[ix] = np.log(np.maximum(np.sqrt((centered*centered).mean(1)), min_std))
    cal.scale_coef = np.linalg.lstsq(design, means, rcond=None)[0]
    predicted = np.exp(np.clip(design @ cal.scale_coef, -30, 30))
    cal.scale_low = np.maximum(np.quantile(predicted, .05, axis=0), min_std)
    cal.scale_high = np.maximum(np.quantile(predicted, .95, axis=0), cal.scale_low)
    del means, predicted, design
    location = np.zeros(d)
    for ix in slices(n, batch):
        z = cal.transform(q[ix], residual[ix]).reshape(-1, d)
        location += z.sum(0)
    count = n*length
    location /= count
    second = np.zeros((d, d)); fourth_sum = 0.
    for ix in slices(n, batch):
        z = cal.transform(q[ix], residual[ix]).reshape(-1, d)-location
        second += z.T @ z
        fourth_sum += float(np.square(np.square(z).sum(1)).sum())
    empirical = second/count
    mu = np.trace(empirical)/d
    delta_prime = np.square(empirical).sum()
    beta = (fourth_sum/count-delta_prime)/(d*count)
    delta = (delta_prime-2*mu*np.trace(empirical)+d*mu*mu)/d
    # Identical spherical/constant data need no shrinkage division by zero.
    shrinkage = 0. if delta <= 0 or d == 1 else min(max(beta, 0.), delta)/delta
    covariance = (1-shrinkage)*empirical + shrinkage*mu*np.eye(d)
    lite.covariance, lite.precision = covariance, pinvh(covariance, check_finite=False)
    lite.location, lite.shrinkage = location, float(shrinkage)
    mag, joint = np.empty(n), np.empty(n)
    for ix in slices(n, batch):
        mag[ix], joint[ix] = lite.components(cal.transform(q[ix], residual[ix]))
    lite.mag_cdf, lite.joint_cdf = EmpiricalCDF().fit(mag), EmpiricalCDF().fit(joint)
    return lite
