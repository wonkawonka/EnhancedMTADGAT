"""A/B memory calibration on five shared, frozen Brand3 backbones.

One native 127-point window per charging snippet. Vehicle Top-5% scores and
normal-validation P95/P99 thresholds match the existing Brand3 RCR protocol.
Temporal prototype switching is unavailable in these unordered snippets.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from scripts.evaluate_rcr import metrics
from src.models.rcr import CONDITION_FEATURES, LinearResidualCalibration
from src.models.rcr_fitting import slices
from src.models.rcr_memory import PrototypeResidualCalibration, residual_moments
from src.runners.run_rcr import atomic_json
from src.runners.run_rcr_lite import cached
from src.runners.run_rcr_memory_probe import digest, normal_nll, score_unit, stability, SCORES, PARTITIONS


KS = [1, 2, 4, 6, 8]
UNITS = ['A_current_rcr']+[f'B_k{k}' for k in KS]


def verify_source(root):
    marker = json.loads((root/'memory_source_complete.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    mapping = json.loads((root/'mapping.json').read_text())
    meta = pd.read_csv(root/'metadata.csv', dtype={'car': str}).set_index('sample_id', verify_integrity=True)
    if protocol['dataset'] != 'Brand3' or protocol['smoke'] or not marker['full_data']:
        raise ValueError('Require completed full Brand3 memory source')
    if protocol['window'] != 127 or protocol['stride'] != 127 or mapping['control_indices'] != [1, 2]:
        raise ValueError('Unexpected window or query protocol')
    if len(mapping['feature_columns']) != 7 or mapping['feature_columns'][1:3] != ['current', 'soc']:
        raise ValueError('Unexpected current/SOC schema')
    vehicles = {p: set(mapping['cars'][p]) for p in PARTITIONS}
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if vehicles[left] & vehicles[right] or set(mapping['sample_ids'][left]) & set(mapping['sample_ids'][right]):
            raise ValueError('Vehicle or snippet leakage')
    signatures = {}
    for partition in PARTITIONS:
        r, q, refs, record = cached(root, partition)
        ids, cars, labels = (mapping[name][partition] for name in ['sample_ids', 'cars', 'labels'])
        n = len(ids)
        if r.shape != (n, 127, 7) or q.shape != (n, 8) or len(cars) != n or len(labels) != n:
            raise ValueError('Misaligned full-data residual cache')
        np.testing.assert_array_equal(refs, np.column_stack((np.arange(n), np.full(n, 126))))
        if record['sample_ids'] != ids or mapping['windows'][partition] != n:
            raise ValueError('Incomplete sample mapping')
        selected = meta.loc[ids]
        np.testing.assert_array_equal(selected.car.astype(str), np.asarray(cars, dtype=str))
        np.testing.assert_array_equal(selected.sample_label, labels)
        if not selected.partition.eq(partition).all():
            raise ValueError('Metadata partition mismatch')
        if partition != 'test' and set(labels) != {0}:
            raise ValueError('Non-normal fitting or threshold data')
        signatures[partition] = dict(completion=digest(root/'residuals'/partition/'complete.json'),
            files={name: dict(size=(root/'residuals'/partition/name).stat().st_size,
                             mtime_ns=(root/'residuals'/partition/name).stat().st_mtime_ns)
                   for name in ['residual.npy', 'condition.npy', 'references.npy']})
    return protocol, mapping, meta, signatures


def vehicle_scores(snippets, mapping, partition):
    """Join by immutable snippet ID before grouping; never rely on row order."""
    expected = pd.DataFrame(dict(sample_id=mapping['sample_ids'][partition],
        car=list(map(str, mapping['cars'][partition])), label=mapping['labels'][partition]))
    frame = snippets.merge(expected, on='sample_id', validate='one_to_one')
    if len(frame) != len(expected) or set(frame.sample_id) != set(expected.sample_id):
        raise ValueError('Incomplete snippet scores')
    rows = []
    for car, group in frame.groupby('car', sort=True):
        if group.label.nunique() != 1:
            raise ValueError('Inconsistent vehicle labels')
        values = group[SCORES].to_numpy()
        count = max(1, int(np.ceil(.05*len(values))))
        top = np.partition(values, -count, axis=0)[-count:].mean(0)
        rows.append(dict(car=str(car), label=int(group.label.iloc[0]), snippets=len(group), **dict(zip(SCORES, top))))
    return frame, pd.DataFrame(rows)


def evaluate_frames(frames, mapping, output, unit, fold):
    vehicles = {}
    for partition in ('validation', 'test'):
        snippet, vehicles[partition] = vehicle_scores(frames[partition], mapping, partition)
        snippet.to_csv(output/f'{partition}_snippet_scores.csv', index=False)
        vehicles[partition].to_csv(output/f'{partition}_vehicle_scores.csv', index=False)
    if not vehicles['validation'].label.eq(0).all():
        raise ValueError('Non-normal threshold data')
    thresholds, rows = {}, []
    for score in SCORES:
        thresholds[score] = {}
        for quantile in (.95, .99):
            threshold = float(vehicles['validation'][score].quantile(quantile))
            thresholds[score][str(quantile)] = threshold
            rows.append(dict(dataset='Brand3', fold=fold, unit=unit, score=score, q=quantile,
                threshold=threshold, **metrics(vehicles['test'].label, vehicles['test'][score], threshold)))
    atomic_json(output/'thresholds.json', thresholds)
    atomic_json(output/'metrics.json', rows)
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    atomic_json(output/'evaluation_complete.json', dict(ranking_level='vehicle',
        threshold_source='normal_validation_vehicle', thresholds=[.95, .99],
        aggregation='snippet_top5pct_then_vehicle_top5pct', no_test_tuning=True))
    return rows


def prototype_diagnostics(calibration, root, output, mapping, args):
    diagnostics, train_distance_thresholds = {}, None
    for partition in PARTITIONS:
        _, q, refs, _ = cached(root, partition)
        n, k = len(q), calibration.k
        labels = np.empty(n, dtype=np.int16)
        detail = np.empty((n, 3), dtype=np.float32)
        rng = np.random.default_rng(args.seed+PARTITIONS.index(partition))
        for ix in slices(n, args.batch):
            z = calibration.standardize(q[ix])
            assigned, distance, margin = calibration.query_standardized(z)
            perturbed = calibration.query_standardized(z+rng.normal(0, args.perturb_std, z.shape))[0]
            labels[ix] = assigned
            detail[ix] = np.column_stack((distance, margin, assigned != perturbed))
        np.save(output/f'{partition}_assignments.npy', labels)
        np.save(output/f'{partition}_query_diagnostics.npy', detail)
        if partition == 'train':
            train_distance_thresholds = [float(np.quantile(detail[labels == j, 0], .95)) for j in range(k)]
        far = detail[:, 0] > np.asarray(train_distance_thresholds)[labels]
        snippets = pd.DataFrame(dict(sample_id=mapping['sample_ids'][partition],
            car=list(map(str, mapping['cars'][partition])), label=mapping['labels'][partition],
            prototype=labels, nearest_distance=detail[:, 0], relative_margin=detail[:, 1],
            perturb_changed=detail[:, 2].astype(bool), above_train_prototype_p95_distance=far))
        snippets.to_csv(output/f'{partition}_snippet_prototypes.csv', index=False)
        vehicle_rows = []
        for car, frame in snippets.groupby('car', sort=True):
            counts = np.bincount(frame.prototype, minlength=k)
            p = counts[counts > 0]/len(frame)
            vehicle_rows.append(dict(car=str(car), label=int(frame.label.iloc[0]), snippets=len(frame),
                dominant_prototype=int(counts.argmax()), prototypes_used=int((counts > 0).sum()),
                dominant_fraction=float(counts.max()/len(frame)), entropy=float(-(p*np.log(p)).sum()),
                mean_distance=float(frame.nearest_distance.mean()),
                perturb_flip_rate=float(frame.perturb_changed.mean()),
                far_fraction=float(frame.above_train_prototype_p95_distance.mean()),
                temporal_switch_rate=None,
                **{f'prototype_{j}_snippets': int(counts[j]) for j in range(k)}))
        pd.DataFrame(vehicle_rows).to_csv(output/f'{partition}_vehicle_prototypes.csv', index=False)
        distribution = [dict(label=int(label), prototype=j, snippets=int((frame.prototype == j).sum()),
            vehicles=int(frame.loc[frame.prototype == j, 'car'].nunique()))
            for label, frame in snippets.groupby('label') for j in range(k)]
        pd.DataFrame(distribution).to_csv(output/f'{partition}_prototype_distribution.csv', index=False)
        by_class = {str(label): dict(snippets=len(frame), vehicles=int(frame.car.nunique()),
            mean_distance=float(frame.nearest_distance.mean()),
            perturb_flip_rate=float(frame.perturb_changed.mean()),
            far_fraction=float(frame.above_train_prototype_p95_distance.mean()))
            for label, frame in snippets.groupby('label')}
        diagnostics[partition] = dict(windows=n, prototype_windows=np.bincount(labels, minlength=k).tolist(),
            perturb_flip_rate=float(detail[:, 2].mean()), boundary_fraction=float(np.mean(detail[:, 1] < .05)),
            class_summary=by_class, temporal_switch_rate=None)
    atomic_json(output/'diagnostics.json', dict(partitions=diagnostics,
        query_columns=['nearest_distance', 'relative_margin', 'perturb_assignment_changed'],
        perturb_std_training_z=args.perturb_std, k1_margin_convention=1,
        normal_train_p95_distance_by_prototype=train_distance_thresholds,
        temporal_switching_available=False,
        reason='one window per snippet; inter-snippet temporal order and fault onset not established'))


def audit_fold(source, output, mapping):
    """Recompute each saved window score and vehicle metric, including mappings."""
    checked_windows = 0
    for unit in UNITS:
        p = output/unit
        if not (p/'complete.json').exists():
            raise ValueError('Incomplete arm')
        calibration = (LinearResidualCalibration.load(p/'calibration.npz') if unit.startswith('A_')
                       else PrototypeResidualCalibration.load(p/'calibration.npz'))
        for partition in PARTITIONS:
            r, q, refs, _ = cached(source, partition)
            score = np.load(p/f'{partition}_window_scores.npy', mmap_mode='r')
            baseline = np.load(output/'A_current_rcr'/f'{partition}_window_scores.npy', mmap_mode='r')
            if score.shape != (len(q), len(SCORES)):
                raise ValueError('Missing window scores')
            for ix in slices(len(q)):
                if not np.isfinite(score[ix]).all():
                    raise ValueError('Nonfinite score')
                np.testing.assert_array_equal(score[ix, 0], baseline[ix, 0])
                np.testing.assert_allclose(score[ix, 1], np.square(calibration.transform(q[ix], r[ix])).mean((1, 2)),
                                           rtol=1e-12, atol=1e-15)
            checked_windows += len(q)
            if unit.startswith('B_'):
                assigned = np.load(p/f'{partition}_assignments.npy')
                np.testing.assert_array_equal(assigned, calibration.query(q)[0])
                snippets = pd.read_csv(p/f'{partition}_snippet_prototypes.csv', dtype={'car': str})
                np.testing.assert_array_equal(snippets.sample_id, mapping['sample_ids'][partition])
                np.testing.assert_array_equal(snippets.prototype, assigned)
        for row in json.loads((p/'metrics.json').read_text()):
            val = pd.read_csv(p/'validation_vehicle_scores.csv', dtype={'car': str})
            test = pd.read_csv(p/'test_vehicle_scores.csv', dtype={'car': str})
            threshold = float(val[row['score']].quantile(row['q']))
            np.testing.assert_allclose(threshold, row['threshold'], rtol=1e-12, atol=1e-15)
            for name, value in metrics(test.label, test[row['score']], threshold).items():
                np.testing.assert_allclose(value, row[name], rtol=1e-12, atol=1e-15)
    result = dict(complete=True, arms=UNITS, score_windows_recomputed=checked_windows,
                  prototype_assignments_recomputed=True, vehicle_metrics_recomputed=True,
                  backbone_hash=json.loads((source/'backbone_complete.json').read_text())['checkpoint_hash'])
    atomic_json(output/'artifact_audit.json', result)
    return result


def run_fold(base, fold, args):
    source = base/'sources'/f'fold{fold}'/'full'
    protocol, mapping, meta, signatures = verify_source(source)
    output = base/'experiments'/f'fold{fold}'
    output.mkdir(parents=True, exist_ok=True)
    baseline = source/'lite_continuous/calibration.npz'
    a = LinearResidualCalibration.load(baseline)
    settings = dict(dataset='Brand3', fold=fold, source=str(source.resolve()), source_protocol=protocol,
        source_mapping_sha256=digest(source/'mapping.json'), source_metadata_sha256=digest(source/'metadata.csv'),
        cache_signatures=signatures, baseline_calibration_sha256=digest(baseline),
        backbone=json.loads((source/'backbone_complete.json').read_text()),
        query_features=list(CONDITION_FEATURES), query_uses_residuals=False,
        implementation_sha256={Path(f).name: digest(f) for f in [__file__,
            Path(__file__).with_name('run_rcr_memory_probe.py'), Path(__file__).parents[1]/'models/rcr_memory.py']},
        seed=args.seed, n_init=args.n_init, max_iter=args.max_iter, ks=KS,
        stability_seeds=args.stability_seeds, perturb_std=args.perturb_std, batch=args.batch,
        threads=args.threads, min_std=a.min_std, device='cpu', score_names=SCORES,
        primary_score='magnitude', normal_validation_threshold_quantiles=[.95, .99],
        aggregation='snippet_top5pct_then_vehicle_top5pct', temporal_switching_available=False,
        no_test_tuning=True, phase_c_started=False)
    manifest = output/'protocol_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError('Source/implementation/settings changed; use new output')
    atomic_json(manifest, settings)
    if (output/'complete.json').exists():
        audit_fold(source, output, mapping)
        print('BRAND3 MEMORY FOLD ALREADY COMPLETE', fold, flush=True)
        return
    for filename in ['mapping.json', 'metadata.csv']:
        shutil.copyfile(source/filename, output/filename)
    for partition in PARTITIONS:
        shutil.copyfile(source/'residuals'/partition/'references.npy', output/f'{partition}_references.npy')
    tr, tq, _, _ = cached(source, 'train')
    vr, vq, _, _ = cached(source, 'validation')
    moments = residual_moments(tr, args.batch)
    units, nlls = {'A_current_rcr': a}, {'A_current_rcr': normal_nll(a, vr, vq, args.batch)}
    for k in KS:
        name, target = f'B_k{k}', output/f'B_k{k}'
        target.mkdir(exist_ok=True)
        model = PrototypeResidualCalibration(k, a.min_std, args.seed, args.n_init, args.max_iter).fit(tq, moments=moments)
        model.save(target/'calibration.npz')
        stability(model, tq, vq, args.stability_seeds, target)
        pd.DataFrame(cdist(model.centers, model.centers)).to_csv(target/'center_distances.csv', index_label='prototype')
        pd.DataFrame(model.centers, columns=CONDITION_FEATURES).to_csv(target/'centers_standardized.csv', index_label='prototype')
        pd.DataFrame(model.centers*model.q_std+model.q_mean, columns=CONDITION_FEATURES).to_csv(
            target/'centers_control_units.csv', index_label='prototype')
        atomic_json(target/'fit_audit.json', dict(normal_train_only=True, windows=len(tq),
            prototype_windows=model.counts.tolist(), min_std=model.min_std, scale=model.scale.tolist(),
            mean=model.mean.tolist(), inertia=model.inertia, iterations=model.n_iter,
            reached_max_iter=model.n_iter >= args.max_iter,
            floor_fraction_by_prototype_variable=(model.floor_counts/model.counts[:, None]).tolist()))
        nlls[name] = normal_nll(model, vr, vq, args.batch)
        units[name] = model
        print('BRAND3 MEMORY FIT', fold, name, nlls[name], flush=True)
    selected = min((n for n in nlls if n.startswith('B_')), key=nlls.get)
    atomic_json(output/'normal_validation_selection.json', dict(nll=nlls, selected=selected, test_labels_used=False))
    results = []
    for name, model in units.items():
        target = output/name
        target.mkdir(exist_ok=True)
        if name.startswith('A_'):
            shutil.copyfile(baseline, target/'calibration.npz')
        frames = score_unit(source, target, model, mapping, args)
        if name.startswith('B_'):
            prototype_diagnostics(model, source, target, mapping, args)
        rows = evaluate_frames(frames, mapping, target, name, fold)
        results.extend(rows)
        atomic_json(target/'complete.json', dict(unit=name, fold=fold, normal_fit_only=True,
            score_names=SCORES, backbone_hash=settings['backbone']['checkpoint_hash'], no_test_tuning=True))
        primary = next(r for r in rows if r['score'] == 'magnitude' and r['q'] == .95)
        print('BRAND3 MEMORY ARM COMPLETE', json.dumps(primary), flush=True)
    audit_fold(source, output, mapping)
    pd.DataFrame(results).to_csv(output/'summary_metrics.csv', index=False)
    atomic_json(output/'complete.json', dict(units=UNITS, fold=fold, backbone_frozen=True,
        selected_by_normal_nll=selected, phase_c_started=False, no_test_tuning=True))
    print('BRAND3 MEMORY FOLD COMPLETE', fold, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs/internal/rcr_memory_brand3_20261025_v1')
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--n-init', type=int, default=5)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--stability-seeds', nargs='+', type=int, default=[3408, 3409])
    parser.add_argument('--perturb-std', type=float, default=.01)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or not all(f in range(5) for f in args.folds):
        parser.error('Specify unique folds 0..4')
    if args.batch < 1 or args.threads < 1 or args.n_init < 1 or args.max_iter < 1 or args.perturb_std <= 0:
        parser.error('Positive fitting and query settings required')
    with threadpool_limits(limits=args.threads):
        for fold in args.folds:
            run_fold(Path(args.root).resolve(), fold, args)


if __name__ == '__main__':
    main()
