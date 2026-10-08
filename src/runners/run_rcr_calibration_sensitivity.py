"""Frozen-residual calibration mechanism diagnostic.

All calibration parameters and empirical CDFs are fitted from normal training
windows. Candidate ``min_std`` selection uses normal validation Gaussian NLL;
test labels are read only after that selection has been recorded.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_rcr import metadata, metrics
from src.models.rcr import EmpiricalCDF, LinearResidualCalibration
from src.models.rcr_fitting import slices
from src.runners.run_rcr import atomic_json


MIN_STDS = (0.005, 0.01, 0.02, 0.05, 0.10)


def cached(root, partition):
    directory = root/'residuals'/partition
    residual = np.load(directory/'residual.npy', mmap_mode='r')
    condition = np.load(directory/'condition.npy', mmap_mode='r')
    references = np.load(directory/'references.npy', mmap_mode='r')
    record = json.loads((directory/'complete.json').read_text())
    assert len(residual) == len(condition) == len(references) == record['windows']
    return residual, condition, references


def fit_mean_and_rms(q, residual, batch=4096):
    q = np.asarray(q, dtype=np.float64)
    n, _, d = residual.shape
    q_mean, q_std = q.mean(0), np.maximum(q.std(0), 1e-8)
    design = np.column_stack(((q-q_mean)/q_std, np.ones(n)))
    window_mean = np.empty((n, d), dtype=np.float64)
    for ix in slices(n, batch):
        window_mean[ix] = np.asarray(residual[ix], dtype=np.float64).mean(1)
    mean_coef = np.linalg.lstsq(design, window_mean, rcond=None)[0]
    rms = np.empty_like(window_mean)
    for ix in slices(n, batch):
        centered = np.asarray(residual[ix], dtype=np.float64)-(design[ix]@mean_coef)[:, None, :]
        rms[ix] = np.sqrt(np.square(centered).mean(1))
    return q_mean, q_std, mean_coef, rms


def make_calibration(q, q_mean, q_std, mean_coef, rms, min_std):
    q = np.asarray(q, dtype=np.float64)
    design = np.column_stack(((q-q_mean)/q_std, np.ones(len(q))))
    calibration = LinearResidualCalibration(min_std)
    calibration.q_mean, calibration.q_std = q_mean.copy(), q_std.copy()
    calibration.mean_coef = mean_coef.copy()
    target = np.log(np.maximum(rms, min_std))
    calibration.scale_coef = np.linalg.lstsq(design, target, rcond=None)[0]
    uncut = np.exp(np.clip(design@calibration.scale_coef, -30, 30))
    calibration.scale_low = np.maximum(np.quantile(uncut, .05, axis=0), min_std)
    calibration.scale_high = np.maximum(np.quantile(uncut, .95, axis=0), calibration.scale_low)
    return calibration, uncut


def window_scores(residual, q, mean_calibration, calibrations, names, batch=4096):
    result = np.empty((len(residual), len(names)), dtype=np.float64)
    for ix in slices(len(residual), batch):
        block = np.asarray(residual[ix], dtype=np.float64)
        mu, _ = mean_calibration.parameters(q[ix])
        centered = block-mu[:, None, :]
        result[ix, 0] = np.square(block).mean((1, 2))
        result[ix, 1] = np.square(centered).mean((1, 2))
        for column, calibration in enumerate(calibrations, 2):
            z = calibration.transform(q[ix], block)
            result[ix, column] = np.square(z).mean((1, 2))
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite calibration sensitivity scores')
    return result


def validation_nll(residual, q, calibration, batch=4096):
    total, count = 0., 0
    for ix in slices(len(residual), batch):
        block = np.asarray(residual[ix], dtype=np.float64)
        mu, sigma = calibration.parameters(q[ix])
        centered = block-mu[:, None, :]
        total += float((np.log(sigma)[:, None, :]+np.square(centered)/(2*np.square(sigma)[:, None, :])).sum())
        count += block.size
    return total/count


def aggregate(window, references, ids, names):
    sequence = references[:, 0]
    assert (np.diff(sequence) >= 0).all()
    boundaries = np.r_[0, np.flatnonzero(np.diff(sequence))+1, len(sequence)]
    assert len(boundaries)-1 == len(ids)
    rows = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        values = window[lo:hi]
        k = max(1, int(np.ceil(.05*len(values))))
        top = np.partition(values, -k, axis=0)[-k:].mean(0)
        rows.append(dict(sample_id=ids[int(sequence[lo])], windows=hi-lo,
                         **dict(zip(names, top))))
    return pd.DataFrame(rows)


def subgroup_metrics(frame, names, thresholds):
    rows = []
    for name in names:
        for q, threshold in thresholds[name].items():
            common = dict(score=name, q=q, threshold=threshold)
            for cycle, cycle_df in frame.groupby('cycle_kind'):
                normal, fault = cycle_df[cycle_df.sample_label == 0], cycle_df[cycle_df.sample_label == 1]
                rows.append(dict(**common, grouping='cycle', cycle=cycle, fault_type='all',
                                 dynamic='all', **metrics(cycle_df.sample_label, cycle_df[name], threshold)))
                for fault_type, subset in fault.groupby('fault_type'):
                    comparison = pd.concat([normal, subset])
                    rows.append(dict(**common, grouping='type', cycle=cycle, fault_type=fault_type,
                                     dynamic='all', **metrics(comparison.sample_label, comparison[name], threshold)))
                    for dynamic, dynamic_subset in subset.groupby('dynamic'):
                        comparison = pd.concat([normal, dynamic_subset])
                        rows.append(dict(**common, grouping='type_dynamic', cycle=cycle,
                                         fault_type=fault_type, dynamic=dynamic,
                                         **metrics(comparison.sample_label, comparison[name], threshold)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'complete.json').exists():
        print('CALIBRATION SENSITIVITY ALREADY COMPLETE', output, flush=True)
        return
    protocol = dict(seed=3407, frozen_root=str(root.resolve()), min_stds=list(MIN_STDS),
                    selection='minimum normal-validation Gaussian NLL',
                    aggregation='window top 5 percent per sample', thresholds=[.95, .99],
                    no_test_selection=True)
    atomic_json(output/'protocol.json', protocol)
    train_residual, train_q, train_ref = cached(root, 'train')
    q_mean, q_std, mean_coef, rms = fit_mean_and_rms(train_q, train_residual)
    mean_calibration = LinearResidualCalibration(1.)
    mean_calibration.q_mean, mean_calibration.q_std = q_mean, q_std
    mean_calibration.mean_coef = mean_coef
    mean_calibration.scale_coef = np.zeros_like(mean_coef)
    mean_calibration.scale_low = mean_calibration.scale_high = np.ones(mean_coef.shape[1])
    calibrations, diagnostics = [], []
    columns = json.loads((root/'mapping.json').read_text())['feature_columns']
    for min_std in MIN_STDS:
        calibration, uncut = make_calibration(train_q, q_mean, q_std, mean_coef, rms, min_std)
        calibrations.append(calibration)
        calibration.save(output/f'calibration_min_std_{str(min_std).replace(".", "p")}.npz')
        clipped = np.clip(uncut, calibration.scale_low, calibration.scale_high)
        for variable, feature in enumerate(columns):
            quantiles = np.quantile(uncut[:, variable], [.05, .25, .5, .75, .95])
            diagnostics.append(dict(min_std=min_std, variable=feature, p05=quantiles[0],
                p25=quantiles[1], median=quantiles[2], p75=quantiles[3], p95=quantiles[4],
                below_0p005=float(np.mean(uncut[:, variable] < .005)),
                below_0p01=float(np.mean(uncut[:, variable] < .01)),
                below_0p02=float(np.mean(uncut[:, variable] < .02)),
                below_0p05=float(np.mean(uncut[:, variable] < .05)),
                low_clip_fraction=float(np.mean(uncut[:, variable] < calibration.scale_low[variable])),
                high_clip_fraction=float(np.mean(uncut[:, variable] > calibration.scale_high[variable])),
                changed_fraction=float(np.mean(clipped[:, variable] != uncut[:, variable])),
                scale_low=calibration.scale_low[variable], scale_high=calibration.scale_high[variable]))
    pd.DataFrame(diagnostics).to_csv(output/'sigma_diagnostics.csv', index=False)
    names = ['raw', 'mean_only']+[f'scale_min_{str(v).replace(".", "p")}' for v in MIN_STDS]
    train_window = window_scores(train_residual, train_q, mean_calibration, calibrations, names)
    references = [EmpiricalCDF().fit(train_window[:, column]) for column in range(len(names))]
    np.save(output/'train_cdf_references.npy', np.column_stack([x.reference for x in references]))
    mapping = json.loads((root/'mapping.json').read_text())['sample_ids']
    sample_frames, raw_windows = {}, {}
    for partition in ['validation', 'test']:
        residual, q, ref = cached(root, partition)
        window = window_scores(residual, q, mean_calibration, calibrations, names)
        percentile = np.column_stack([cdf.transform(window[:, i]) for i, cdf in enumerate(references)])
        all_names = names+[f'{name}_cdf' for name in names]
        raw_windows[partition] = np.column_stack((window, percentile))
        sample_frames[partition] = aggregate(raw_windows[partition], ref, mapping[partition], all_names)
        sample_frames[partition].to_csv(output/f'{partition}_sample_scores.csv', index=False)
    val_residual, val_q, _ = cached(root, 'validation')
    nll = {name: validation_nll(val_residual, val_q, calibration)
           for name, calibration in zip(names[2:], calibrations)}
    selected = min(nll, key=nll.get)
    atomic_json(output/'selection.json', dict(validation_normal_nll=nll, selected=selected,
                selected_min_std=float(selected.removeprefix('scale_min_').replace('p', '.')),
                test_labels_used=False))
    test_meta = metadata(root)
    test = test_meta.merge(sample_frames['test'], on='sample_id', validate='one_to_one')
    validation = sample_frames['validation']
    score_names = names+[f'{name}_cdf' for name in names]
    thresholds = {name: {q: float(validation[name].quantile(q)) for q in [.95, .99]}
                  for name in score_names}
    rows = []
    for name in score_names:
        for q, threshold in thresholds[name].items():
            rows.append(dict(score=name, q=q, threshold=threshold,
                             **metrics(test.sample_label, test[name], threshold)))
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    subgroup_metrics(test, score_names, thresholds).to_csv(output/'strata.csv', index=False)
    test.to_csv(output/'test_sample_details.csv', index=False)
    atomic_json(output/'complete.json', dict(selected=selected, candidates=names,
                normal_validation_selection=True, no_test_tuning=True, samples=len(test)))
    print('CALIBRATION SENSITIVITY COMPLETE', output, selected, nll[selected], flush=True)


if __name__ == '__main__':
    main()
