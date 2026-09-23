"""Fit frozen RCR calibration and Lite using all cached normal train windows."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.rcr import RCRLite
from src.models.rcr_fitting import fit_lite_streaming, slices
from src.runners.run_rcr import atomic_json


SCORE_NAMES = ['raw', 'magnitude', 'relation', 'magnitude_percentile', 'relation_percentile', 'combined']


def cached(root, partition):
    directory = root/'residuals'/partition
    record = json.loads((directory/'complete.json').read_text())
    complete = json.loads((root/'backbone_complete.json').read_text())
    assert record['backbone_hash'] == complete['checkpoint_hash']
    residual = np.load(directory/'residual.npy', mmap_mode='r')
    q = np.load(directory/'condition.npy', mmap_mode='r')
    references = np.load(directory/'references.npy', mmap_mode='r')
    assert len(residual) == len(q) == len(references) == record['windows']
    return residual, q, references, record


def conditions(root, partition, mode, q, references):
    if mode == 'continuous':
        return q
    if mode == 'none':
        return np.zeros((len(q), 8), dtype=np.float32)
    if mode != 'cycle':
        raise ValueError(mode)
    mapping = json.loads((root/'mapping.json').read_text())['sample_ids'][partition]
    metadata = pd.read_csv(root/'metadata.csv').set_index('sample_id')
    cycle = metadata.loc[mapping, 'cycle_kind']
    assert cycle.isin(['charge', 'discharge']).all()
    code = cycle.eq('discharge').to_numpy(dtype=int)
    return np.eye(2, dtype=np.float32)[code[references[:, 0]]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--condition', choices=['continuous', 'none', 'cycle'], default='continuous')
    args = parser.parse_args()
    root = Path(args.root)
    output = root/('lite_'+args.condition)
    output.mkdir(exist_ok=True)
    backbone = json.loads((root/'backbone_complete.json').read_text())
    if (output/'complete.json').exists():
        saved = json.loads((output/'complete.json').read_text())
        assert saved['backbone_hash'] == backbone['checkpoint_hash']
        for p in ['train', 'validation', 'test']:
            assert (output/p/'complete.json').exists()
        print('LITE ALREADY COMPLETE', output, flush=True)
        return
    residual, q, references, _ = cached(root, 'train')
    q = conditions(root, 'train', args.condition, q, references)
    print('FITTING FULL NORMAL CALIBRATION', len(q), args.condition, flush=True)
    lite = fit_lite_streaming(q, residual)
    lite.save(output)
    # Serialization audit on fixed beginning/end training windows, not label-selected.
    check = np.unique(np.r_[np.arange(min(64, len(q))), np.arange(max(0, len(q)-64), len(q))])
    restored = RCRLite.load(output)
    for name, score in lite.score(q[check], residual[check]).items():
        np.testing.assert_array_equal(restored.score(q[check], residual[check])[name], score)
    atomic_json(output/'fit_audit.json', dict(normal_train_only=True, condition=args.condition,
                windows=len(q), points=int(len(q)*residual.shape[1]), shrinkage=lite.shrinkage,
                precision_min_eigenvalue=float(np.linalg.eigvalsh(lite.precision).min()),
                score_reproduction='exact', alpha=.5, no_test_selection=True))
    for partition in ['train', 'validation', 'test']:
        residual, q, refs, record = cached(root, partition)
        q = conditions(root, partition, args.condition, q, refs)
        n, length, d = residual.shape
        directory = output/partition
        directory.mkdir(exist_ok=True)
        score = np.lib.format.open_memmap(directory/'scores.npy', mode='w+', dtype='float64', shape=(n, len(SCORE_NAMES)))
        z = np.lib.format.open_memmap(directory/'standardized_residual.npy', mode='w+', dtype='float32', shape=residual.shape)
        cq = np.lib.format.open_memmap(directory/'condition_standardized.npy', mode='w+', dtype='float32', shape=q.shape)
        for ix in slices(n):
            values = lite.score(q[ix], residual[ix], return_residual=True)
            score[ix] = np.column_stack([values[name] for name in SCORE_NAMES])
            z[ix] = values['standardized_residual']
            cq[ix] = (q[ix]-lite.calibration.q_mean)/lite.calibration.q_std
            if not np.isfinite(score[ix]).all() or not np.isfinite(z[ix]).all() or not np.isfinite(cq[ix]).all():
                raise FloatingPointError('Nonfinite calibration output')
        score.flush(); z.flush(); cq.flush()
        atomic_json(directory/'complete.json', dict(windows=n, score_names=SCORE_NAMES,
                    residual_source=str(root/'residuals'/partition/'residual.npy'),
                    references_source=str(root/'residuals'/partition/'references.npy'),
                    backbone_hash=backbone['checkpoint_hash'], condition=args.condition,
                    standardized_residual_dtype='float32', scoring_dtype='float64',
                    magnitude_saturation_fraction=float(np.mean(score[:, 3] == 1)),
                    relation_saturation_fraction=float(np.mean(score[:, 4] == 1))))
        print('LITE SCORED', partition, n, flush=True)
    atomic_json(output/'complete.json', dict(backbone_hash=backbone['checkpoint_hash'],
                condition=args.condition, stage='lite_calibration_and_scores', normal_fit_only=True))
    print('RCR LITE COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
