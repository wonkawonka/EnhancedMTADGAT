"""A/B frozen-backbone memory probe, with normal-only fitting and diagnostics.

Uses existing full RCR residual caches; never runs a backbone or uses CUDA.
All K values are reported. C (learned memory) is a subsequent research stage.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from scripts.evaluate_rcr import metrics
from src.models.rcr import CONDITION_FEATURES, EmpiricalCDF, LinearResidualCalibration
from src.models.rcr_fitting import slices
from src.models.rcr_memory import PrototypeResidualCalibration, residual_moments
from src.runners.run_rcr import atomic_json
from src.runners.run_rcr_calibration_sensitivity import aggregate
from src.runners.run_rcr_lite import cached


PARTITIONS = ('train', 'validation', 'test')
SCORES = ['raw', 'magnitude', 'magnitude_cdf']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(root):
    protocol = json.loads((root/'protocol.json').read_text())
    if protocol.get('smoke', True):
        raise ValueError('A full-data frozen residual cache is required')
    mapping = json.loads((root/'mapping.json').read_text())
    if mapping.get('control_indices') != [1, 2] or len(mapping['feature_columns']) != 7:
        raise ValueError('This first probe requires the CH current/SOC control schema')
    meta = pd.read_csv(root/'metadata.csv').set_index('sample_id', verify_integrity=True)
    signatures = {}
    for partition in PARTITIONS:
        residual, q, refs, record = cached(root, partition)
        ids = mapping['sample_ids'][partition]
        rows = meta.loc[ids]
        if not rows.partition.eq(partition).all():
            raise ValueError('Metadata partition mismatch')
        if partition != 'test' and not rows.sample_label.eq(0).all():
            raise ValueError('Fitting and threshold partitions must contain normals only')
        if residual.shape != (len(q), protocol['window'], 7) or q.shape != (len(refs), 8):
            raise ValueError('Unexpected cache dimensions')
        if len(q) != mapping['windows'][partition] or not len(refs):
            raise ValueError('Incomplete cache')
        sequence = refs[:, 0]
        if not np.array_equal(np.unique(sequence), np.arange(len(ids))) or (np.diff(sequence) < 0).any():
            raise ValueError('Incomplete or unordered sample references')
        same = np.diff(sequence) == 0
        if not (np.diff(refs[:, 1])[same] == protocol['stride']).all():
            raise ValueError('Noncontiguous windows: cannot measure temporal switching')
        if record.get('sample_ids', ids) != ids:
            raise ValueError('Cache/sample mapping mismatch')
        signatures[partition] = dict(completion=digest(root/'residuals'/partition/'complete.json'),
            files={name: dict(size=(root/'residuals'/partition/name).stat().st_size,
                             mtime_ns=(root/'residuals'/partition/name).stat().st_mtime_ns)
                   for name in ('residual.npy', 'condition.npy', 'references.npy')})
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if set(mapping['sample_ids'][left]) & set(mapping['sample_ids'][right]):
            raise ValueError('Sample leakage across partitions')
    # Fault samples can share source VINs with normals in this synthetic-fault
    # protocol. Enforce the existing normal-vehicle split, not a new split.
    for left, right in (('train_vins', 'validation_vins'), ('train_vins', 'test_normal_vins'),
                        ('validation_vins', 'test_normal_vins')):
        if set(mapping[left]) & set(mapping[right]):
            raise ValueError('Normal VIN leakage across partitions')
    return protocol, mapping, meta, signatures


def normal_nll(calibration, residual, q, batch):
    total = 0.
    for ix in slices(len(q), batch):
        mean, scale = calibration.parameters(q[ix])
        z = (np.asarray(residual[ix], dtype=np.float64)-mean[:, None, :])/scale[:, None, :]
        total += float((np.log(scale)[:, None, :]+.5*np.square(z)).sum())
    value = total/residual.size
    if not np.isfinite(value):
        raise FloatingPointError('Nonfinite normal validation NLL')
    return value


def stability(calibration, q, validation_q, seeds, output):
    """All normal windows; Hungarian alignment removes arbitrary cluster IDs."""
    zq, zv = calibration.standardize(q), calibration.standardize(validation_q)
    labels = calibration.query(q)[0]
    val_labels = calibration.query(validation_q)[0]
    rows = []
    for seed in seeds:
        alternative = KMeans(n_clusters=calibration.k, random_state=seed,
                             n_init=calibration.n_init, max_iter=calibration.max_iter,
                             algorithm='lloyd').fit(zq)
        cost = cdist(calibration.centers, alternative.cluster_centers_)
        primary, secondary = linear_sum_assignment(cost)
        mapping = np.empty(calibration.k, dtype=int)
        mapping[secondary] = primary
        for partition, values, base in (('train', zq, labels), ('validation', zv, val_labels)):
            predicted = alternative.predict(values)
            rows.append(dict(seed=seed, partition=partition,
                ari=float(adjusted_rand_score(base, predicted)),
                aligned_agreement=float(np.mean(mapping[predicted] == base)),
                aligned_center_distance_mean=float(cost[primary, secondary].mean()),
                inertia=float(alternative.inertia_), iterations=int(alternative.n_iter_)))
        np.savez(output/f'stability_seed{seed}.npz', centers=alternative.cluster_centers_,
                 alternative_to_primary=mapping)
    pd.DataFrame(rows).to_csv(output/'matching_stability.csv', index=False)


def prototype_diagnostics(calibration, q, refs, ids, meta, partition, output, args):
    n, k = len(q), calibration.k
    assignments = np.lib.format.open_memmap(output/f'{partition}_assignments.npy',
                                          mode='w+', dtype='int16', shape=(n,))
    # Columns are documented in diagnostics.json; references are saved once.
    details = np.lib.format.open_memmap(output/f'{partition}_query_diagnostics.npy',
                                       mode='w+', dtype='float32', shape=(n, 3))
    rng = np.random.default_rng(args.seed+PARTITIONS.index(partition))
    for ix in slices(n, args.batch):
        zq = calibration.standardize(q[ix])
        labels, distance, margin = calibration.query_standardized(zq)
        perturbed = calibration.query_standardized(zq+rng.normal(0, args.perturb_std, zq.shape))[0]
        assignments[ix] = labels
        details[ix] = np.column_stack((distance, margin, labels != perturbed))
    assignments.flush(); details.flush()
    seq = refs[:, 0]
    boundaries = np.r_[0, np.flatnonzero(np.diff(seq))+1, n]
    sample_rows = []
    transitions = np.zeros((k, k), dtype=np.int64)
    label_transitions = {0: np.zeros_like(transitions), 1: np.zeros_like(transitions)}
    grouped_transitions = {}
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        sample_id = ids[int(seq[lo])]
        row = meta.loc[sample_id]
        label = int(row.sample_label)
        a = np.asarray(assignments[lo:hi], dtype=int)
        counts = np.bincount(a, minlength=k)
        probability = counts[counts > 0]/len(a)
        pairs = max(0, len(a)-1)
        switch_count = int(np.count_nonzero(np.diff(a)))
        common = dict(partition=partition, sample_id=sample_id, sample_label=label,
                      vin=str(row.vin), cycle_kind=str(row.cycle_kind), fault_type=str(row.fault_type))
        sample_rows.append(dict(**common, windows=len(a), dominant_prototype=int(counts.argmax()),
            prototypes_used=int((counts > 0).sum()), dominant_fraction=float(counts.max()/len(a)),
            entropy=float(-(probability*np.log(probability)).sum()), switches=switch_count,
            adjacent_pairs=pairs, switch_rate=switch_count/pairs if pairs else np.nan,
            mean_distance=float(details[lo:hi, 0].mean()),
            boundary_fraction=float(np.mean(details[lo:hi, 1] < .05)),
            perturb_flip_rate=float(details[lo:hi, 2].mean()),
            **{f'prototype_{j}_windows': int(counts[j]) for j in range(k)}))
        transition = np.bincount(a[:-1]*k+a[1:], minlength=k*k).reshape(k, k)
        transitions += transition
        label_transitions[label] += transition
        key = (label, str(row.cycle_kind), str(row.fault_type))
        if key not in grouped_transitions:
            grouped_transitions[key] = np.zeros_like(transition)
        grouped_transitions[key] += transition
    sample_frame = pd.DataFrame(sample_rows)
    sample_frame.to_csv(output/f'{partition}_sample_prototypes.csv', index=False)
    np.savez(output/f'{partition}_transitions.npz', all=transitions,
             normal=label_transitions[0], fault=label_transitions[1])
    transition_rows = [dict(sample_label=label, cycle_kind=cycle, fault_type=fault,
                            source=i, target=j, count=int(matrix[i, j]))
                       for (label, cycle, fault), matrix in grouped_transitions.items()
                       for i in range(k) for j in range(k)]
    pd.DataFrame(transition_rows).to_csv(output/f'{partition}_transitions_by_group.csv', index=False)
    distributions = []
    for (label, cycle, fault), frame in sample_frame.groupby(['sample_label', 'cycle_kind', 'fault_type']):
        for j in range(k):
            occupied = frame[frame[f'prototype_{j}_windows'] > 0]
            distributions.append(dict(partition=partition, sample_label=int(label), cycle_kind=cycle,
                fault_type=fault, prototype=j, windows=int(frame[f'prototype_{j}_windows'].sum()),
                samples=len(occupied), vins=int(occupied.vin.nunique())))
    pd.DataFrame(distributions).to_csv(output/f'{partition}_prototype_distribution.csv', index=False)
    return dict(windows=n, perturb_flip_rate=float(details[:, 2].mean()),
                boundary_fraction=float(np.mean(details[:, 1] < .05)),
                switch_rate=float((transitions.sum()-np.trace(transitions))/max(1, transitions.sum())),
                prototype_windows=np.bincount(assignments, minlength=k).tolist())


def score_unit(root, output, calibration, mapping, args):
    frames, cdf = {}, None
    for partition in PARTITIONS:
        residual, q, refs, _ = cached(root, partition)
        values = np.lib.format.open_memmap(output/f'{partition}_window_scores.npy', mode='w+',
                                           dtype='float64', shape=(len(q), len(SCORES)))
        for ix in slices(len(q), args.batch):
            block = np.asarray(residual[ix], dtype=np.float64)
            values[ix, 0] = np.square(block).mean((1, 2))
            values[ix, 1] = np.square(calibration.transform(q[ix], block)).mean((1, 2))
        if partition == 'train':
            cdf = EmpiricalCDF().fit(values[:, 1])
            np.save(output/'magnitude_cdf_reference.npy', cdf.reference)
        for ix in slices(len(q), args.batch):
            values[ix, 2] = cdf.transform(values[ix, 1])
            if not np.isfinite(values[ix]).all():
                raise FloatingPointError('Nonfinite memory probe scores')
        values.flush()
        if partition != 'train':
            frames[partition] = aggregate(values, refs, mapping['sample_ids'][partition], SCORES)
            frames[partition].to_csv(output/f'{partition}_sample_scores.csv', index=False)
    return frames


def evaluate_frames(frames, meta, output, unit):
    validation = frames['validation']
    test = frames['test'].merge(meta.reset_index(), on='sample_id', validate='one_to_one')
    if len(test) != len(frames['test']):
        raise ValueError('Missing test metadata')
    rows, strata, thresholds = [], [], {}
    for name in SCORES:
        thresholds[name] = {}
        for quantile in (.95, .99):
            threshold = float(validation[name].quantile(quantile))
            thresholds[name][str(quantile)] = threshold
            common = dict(unit=unit, score=name, q=quantile, threshold=threshold)
            rows.append(dict(**common, **metrics(test.sample_label, test[name], threshold)))
            for cycle, frame in test.groupby('cycle_kind'):
                strata.append(dict(**common, cycle=cycle, fault_type='all',
                                   **metrics(frame.sample_label, frame[name], threshold)))
                normal = frame[frame.sample_label == 0]
                for fault, faults in frame[frame.sample_label == 1].groupby('fault_type'):
                    comparison = pd.concat((normal, faults))
                    strata.append(dict(**common, cycle=cycle, fault_type=fault,
                        **metrics(comparison.sample_label, comparison[name], threshold)))
    atomic_json(output/'thresholds.json', thresholds)
    # Convert undefined (single-class) metrics to JSON null, not NaN.
    atomic_json(output/'metrics.json', json.loads(pd.DataFrame(rows).to_json(orient='records', double_precision=15)))
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(strata).to_csv(output/'strata.csv', index=False)
    test.to_csv(output/'test_sample_details.csv', index=False)
    return rows


def run(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    source_protocol, mapping, meta, cache_signatures = verify_source(root)
    baseline = root/'lite_continuous/calibration.npz'
    a = LinearResidualCalibration.load(baseline)
    implementation = {str(Path(p).name): digest(p) for p in (__file__,
        Path(__file__).parents[1]/'models/rcr_memory.py')}
    protocol = dict(implementation='rcr_memory_probe_v1', implementation_sha256=implementation,
        frozen_root=str(root), backbone=json.loads((root/'backbone_complete.json').read_text()),
        source_protocol=source_protocol, source_mapping_sha256=digest(root/'mapping.json'),
        source_metadata_sha256=digest(root/'metadata.csv'), cache_signatures=cache_signatures,
        baseline_calibration_sha256=digest(baseline), ks=args.ks, seed=args.seed, n_init=args.n_init,
        max_iter=args.max_iter, stability_seeds=args.stability_seeds, batch=args.batch,
        perturb_std=args.perturb_std, min_std=a.min_std, device='cpu', gpu='unused',
        query_features=list(CONDITION_FEATURES), query_uses_residuals=False,
        train_window_weighting='all native windows, same as existing RCR',
        score_names=SCORES, primary_score='magnitude', cdf='normal training windows only',
        aggregation='per-sample top 5 percent window mean', threshold_quantiles=[.95, .99],
        threshold_source='normal validation samples only', primary_threshold=.95,
        k_selection='minimum normal validation Gaussian NLL; report every K',
        phase_c='deferred until multi-state benefit is reviewed; no test-time memory updates',
        no_test_tuning=True, threads=args.threads)
    if output == root or root in output.parents:
        raise ValueError('Use a separate output tree to preserve the frozen source')
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'protocol_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Source or protocol changed; use a new output directory')
    if (output/'complete.json').exists():
        print('MEMORY PROBE ALREADY COMPLETE', output, flush=True)
        return
    atomic_json(manifest, protocol)
    shutil.copyfile(root/'mapping.json', output/'mapping.json')
    shutil.copyfile(root/'metadata.csv', output/'metadata.csv')
    for partition in PARTITIONS:
        # Small mappings only; never duplicate the multi-GB residual cache.
        shutil.copyfile(root/'residuals'/partition/'references.npy', output/f'{partition}_references.npy')
    tr, tq, _, _ = cached(root, 'train')
    vr, vq, _, _ = cached(root, 'validation')
    print('MEMORY PROBE START', json.dumps(dict(root=str(root), output=str(output), device='cpu',
          threads=args.threads, windows=mapping['windows'], ks=args.ks, min_std=a.min_std)), flush=True)
    moments = residual_moments(tr, args.batch)
    units = {'A_current_rcr': a}
    normal_nlls = {}
    for k in args.ks:
        name = f'B_k{k}'
        target = output/name
        target.mkdir(exist_ok=True)
        model = PrototypeResidualCalibration(k, a.min_std, args.seed, args.n_init, args.max_iter)
        model.fit(tq, moments=moments)
        model.save(target/'calibration.npz')
        restored = PrototypeResidualCalibration.load(target/'calibration.npz')
        audit = np.unique(np.r_[np.arange(min(16, len(tq))), np.arange(max(0, len(tq)-16), len(tq))])
        np.testing.assert_array_equal(model.transform(tq[audit], tr[audit]), restored.transform(tq[audit], tr[audit]))
        stability(model, tq, vq, args.stability_seeds, target)
        distances = cdist(model.centers, model.centers)
        pd.DataFrame(distances).to_csv(target/'center_distances.csv', index_label='prototype')
        pd.DataFrame(model.centers, columns=CONDITION_FEATURES).to_csv(target/'centers_standardized.csv', index_label='prototype')
        pd.DataFrame(model.centers*model.q_std+model.q_mean, columns=CONDITION_FEATURES).to_csv(
            target/'centers_control_units.csv', index_label='prototype')
        normal_nlls[name] = normal_nll(model, vr, vq, args.batch)
        atomic_json(target/'fit_audit.json', dict(normal_train_only=True, windows=len(tq),
            prototype_windows=model.counts.tolist(), min_std=model.min_std, inertia=model.inertia,
            iterations=model.n_iter, reached_max_iter=model.n_iter >= args.max_iter,
            floor_fraction_by_prototype_variable=(model.floor_counts/model.counts[:, None]).tolist(),
            scale=model.scale.tolist(), mean=model.mean.tolist(), exact_serialization=True))
        units[name] = model
        print('MEMORY FIT COMPLETE', name, 'normal_validation_nll', normal_nlls[name], flush=True)
    del moments
    normal_nlls['A_current_rcr'] = normal_nll(a, vr, vq, args.batch)
    # Record a normal-only selection BEFORE scoring test data. This does not
    # authorize C automatically: A and K=1 are required empirical comparators.
    selected = min((name for name in normal_nlls if name.startswith('B_')), key=normal_nlls.get)
    atomic_json(output/'normal_validation_selection.json', dict(nll=normal_nlls, selected=selected,
        test_labels_used=False, phase_c_started=False))
    results = []
    for name, calibration in units.items():
        target = output/name
        target.mkdir(exist_ok=True)
        if name.startswith('A_'):
            shutil.copyfile(baseline, target/'calibration.npz')
        frames = score_unit(root, target, calibration, mapping, args)
        if isinstance(calibration, PrototypeResidualCalibration):
            diagnostics = {}
            for partition in PARTITIONS:
                _, q, refs, _ = cached(root, partition)
                diagnostics[partition] = prototype_diagnostics(calibration, q, refs,
                    mapping['sample_ids'][partition], meta, partition, target, args)
            atomic_json(target/'diagnostics.json', dict(partitions=diagnostics,
                query_columns=['nearest_distance', 'relative_margin', 'perturb_assignment_changed'],
                perturb_std_training_z=args.perturb_std, perturb_std_seed=args.seed,
                boundary_relative_margin=.05, k1_margin_convention=1,
                temporal_pairs='only adjacent native-stride windows within same sample',
                fault_switches='within fault-labelled samples; no annotated fault onset is assumed'))
        rows = evaluate_frames(frames, meta, target, name)
        results.extend(rows)
        atomic_json(target/'complete.json', dict(unit=name, normal_fit_only=True,
            windows=mapping['windows'], backbone_hash=protocol['backbone']['checkpoint_hash'],
            score_names=SCORES, no_test_tuning=True))
        primary = next(r for r in rows if r['score'] == 'magnitude' and r['q'] == .95)
        print('MEMORY EVALUATION COMPLETE', json.dumps(primary), flush=True)
    pd.DataFrame(results).to_csv(output/'summary_metrics.csv', index=False)
    atomic_json(output/'complete.json', dict(units=list(units), selected_by_normal_nll=selected,
        backbone_frozen=True, phase_c_started=False, no_test_tuning=True))
    print('MEMORY PROBE COMPLETE', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ks', nargs='+', type=int, default=[1, 2, 4, 6, 8])
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--n-init', type=int, default=5)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--stability-seeds', nargs='+', type=int, default=[3408, 3409])
    parser.add_argument('--perturb-std', type=float, default=.01)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if len(set(args.ks)) != len(args.ks) or not all(1 <= k <= 32767 for k in args.ks) or 1 not in args.ks:
        parser.error('K must be unique positive int16 values and include K=1')
    if args.batch < 1 or args.threads < 1 or args.perturb_std <= 0:
        parser.error('Batch, threads and perturbation scale must be positive')
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == '__main__':
    main()
