"""Normal-calibrated RCR evaluation; all predefined scores, never best-test selection."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.runners.run_rcr import atomic_json


def metadata(root):
    target = root/'evaluation_metadata.csv'
    source = pd.read_csv(root/'metadata.csv')
    source = source[source.partition == 'test'].sort_values('sample_id').copy()
    if target.exists():
        result = pd.read_csv(target)
        np.testing.assert_array_equal(result.sample_id, source.sample_id)
        return result
    di, voltage = [], []
    for row in source.itertuples():
        raw = pd.read_csv(row.file_path, usecols=['SUM_CURRENT', 'SUM_VOLTAGE'])
        di.append(raw.SUM_CURRENT.diff().abs().mean())
        voltage.append(raw.SUM_VOLTAGE.mean())
    source['di'] = di; source['pack_voltage'] = voltage
    source['dynamic'] = 'unassigned'
    for _, group in source.groupby(['vin', 'cycle_kind', 'fault_type']):
        # Same deterministic within-VIN thirds as earlier HR diagnostic.
        if len(group) >= 3:
            categories = pd.qcut(group.di.rank(method='first'), 3, labels=['low', 'medium', 'high'])
            source.loc[group.index, 'dynamic'] = categories.astype(str)
    source.to_csv(target, index=False)
    return source


def aggregate(root, unit, partition):
    record = json.loads((unit/partition/'complete.json').read_text())
    score = np.load(unit/partition/'scores.npy', mmap_mode='r')
    ref = np.load(root/'residuals'/partition/'references.npy', mmap_mode='r')
    ids = json.loads((root/'mapping.json').read_text())['sample_ids'][partition]
    assert len(score) == len(ref) == record['windows']
    names = record['score_names']
    assert names == ['raw', 'magnitude', 'relation', 'magnitude_percentile', 'relation_percentile', 'combined']
    assert np.isfinite(score).all()
    seq = ref[:, 0]
    assert (np.diff(seq) >= 0).all()
    boundaries = np.r_[0, np.flatnonzero(np.diff(seq))+1, len(seq)]
    assert len(boundaries)-1 == len(ids), 'Missing sample windows; smoke scores cannot be formal metrics'
    rows = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        sample = int(seq[lo]); window = np.asarray(score[lo:hi])
        # Alpha fusion occurs per window BEFORE sample aggregation.
        alpha_scores = np.column_stack([a*window[:, 3]+(1-a)*window[:, 4] for a in [1., .75, .5, .25, 0.]])
        values = np.column_stack((window, alpha_scores))
        k = max(1, int(np.ceil(.05*len(values))))
        summary = np.partition(values, -k, axis=0)[-k:].mean(0)
        keys = names+['alpha_1', 'alpha_075', 'alpha_05', 'alpha_025', 'alpha_0']
        rows.append(dict(sample_id=ids[sample], windows=hi-lo, **dict(zip(keys, summary))))
    frame = pd.DataFrame(rows)
    np.testing.assert_allclose(frame.combined, frame.alpha_05)
    frame.to_csv(unit/f'{partition}_sample_scores.csv', index=False)
    return frame


def metrics(y, score, threshold):
    y = np.asarray(y, dtype=int); score = np.asarray(score); p = score >= threshold
    assert set(np.unique(y)) <= {0, 1}
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    return dict(n=len(y), tp=tp, fp=fp, fn=fn, tn=tn,
                auroc=roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                ap=average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else np.nan,
                fpr=fp/(fp+tn) if fp+tn else np.nan, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def evaluate(root, unit, meta):
    assert (unit/'complete.json').exists()
    vf, tf = aggregate(root, unit, 'validation'), aggregate(root, unit, 'test')
    original = pd.read_csv(root/'metadata.csv').set_index('sample_id')
    assert original.loc[vf.sample_id].sample_label.eq(0).all()
    df = meta.merge(tf, on='sample_id', validate='one_to_one')
    assert len(df) == len(meta) == len(tf)
    rows, groups = [], []
    names = [c for c in vf if c not in ['sample_id', 'windows']]
    for name in names:
        for quantile in [.95, .99]:
            threshold = float(vf[name].quantile(quantile))
            common = dict(unit=unit.name, score=name, q=quantile, threshold=threshold)
            rows.append(dict(**common, **metrics(df.sample_label, df[name], threshold)))
            for cycle, cycle_df in df.groupby('cycle_kind'):
                groups.append(dict(**common, grouping='cycle', cycle=cycle, fault_type='all', severity='all',
                                   dynamic='all', **metrics(cycle_df.sample_label, cycle_df[name], threshold)))
                normal = cycle_df[cycle_df.sample_label == 0]
                fault = cycle_df[cycle_df.sample_label == 1]
                for (ft, sev), subset in fault.groupby(['fault_type', 'severity'], dropna=False):
                    comparison = pd.concat([normal, subset])
                    groups.append(dict(**common, grouping='type_severity', cycle=cycle, fault_type=ft,
                                       severity=sev, dynamic='all', **metrics(comparison.sample_label, comparison[name], threshold)))
                for ft, subset in fault.groupby('fault_type'):
                    comparison = pd.concat([normal, subset])
                    groups.append(dict(**common, grouping='type', cycle=cycle, fault_type=ft,
                                       severity='all', dynamic='all', **metrics(comparison.sample_label, comparison[name], threshold)))
                    for dynamic, dyn in subset.groupby('dynamic'):
                        # Explicit comparator: all same-cycle normals, NOT severity-matched normals.
                        comparison = pd.concat([normal, dyn])
                        groups.append(dict(**common, grouping='type_dynamic', cycle=cycle, fault_type=ft,
                                           severity='all', dynamic=dynamic, **metrics(comparison.sample_label, comparison[name], threshold)))
    pd.DataFrame(rows).to_csv(unit/'metrics.csv', index=False)
    pd.DataFrame(groups).to_csv(unit/'strata.csv', index=False)
    df.to_csv(unit/'test_sample_details.csv', index=False)
    atomic_json(unit/'evaluation_complete.json', dict(normal_validation_thresholds=[.95, .99],
                scores=names, alpha_primary=.5, samples=len(df),
                dynamic='within VIN/cycle/fault-type rank thirds of raw mean abs adjacent current diff',
                subgroup_normal_comparator='all same-cycle normals', no_test_tuning=True))
    print(pd.DataFrame(rows).query('q == .95 and score in ["raw", "magnitude", "combined"]').round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--unit', action='append', default=[])
    parser.add_argument('--prepare-metadata', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    assert not json.loads((root/'protocol.json').read_text())['smoke'], 'No formal evaluation of smoke outputs'
    meta = metadata(root)
    if args.prepare_metadata:
        print('METADATA PREPARED', len(meta), flush=True)
        return
    if not args.unit:
        raise ValueError('Specify completed units explicitly')
    for unit in args.unit:
        evaluate(root, root/unit, meta)


if __name__ == '__main__':
    main()
