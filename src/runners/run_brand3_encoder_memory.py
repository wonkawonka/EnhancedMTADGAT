"""Frozen Encoder H memory versus unchanged RCR, with fixed CDF fusion."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from scripts.evaluate_rcr import metrics
from src.models.rcr import EmpiricalCDF
from src.models.rcr_feature_memory import EncoderFeatureMemory
from src.models.rcr_fitting import slices
from src.runners.run_rcr import atomic_json
from src.runners.run_rcr_memory_probe import digest, stability
from src.runners.run_brand3_memory_probe import KS, PARTITIONS


SCORE_NAMES = ['RCR', 'RCR_CDF']+[name for k in KS for name in [f'Memory_k{k}', f'Memory_CDF_k{k}', f'Fusion_k{k}']]


def fuse_percentiles(rcr, memory):
    if rcr.shape != memory.shape or not np.isfinite(rcr).all() or not np.isfinite(memory).all():
        raise ValueError('Expected aligned finite branch percentiles')
    if (rcr < 0).any() or (rcr > 1).any() or (memory < 0).any() or (memory > 1).any():
        raise ValueError('Fusion inputs must be training-CDF percentiles')
    return .5*rcr+.5*memory


def aggregate_vehicle(window_scores, mapping, partition):
    n = len(mapping['sample_ids'][partition])
    if window_scores.shape != (n, len(SCORE_NAMES)):
        raise ValueError('Expected one native window per snippet')
    snippet = pd.DataFrame(window_scores, columns=SCORE_NAMES)
    snippet.insert(0, 'label', mapping['labels'][partition])
    snippet.insert(0, 'car', list(map(str, mapping['cars'][partition])))
    snippet.insert(0, 'sample_id', mapping['sample_ids'][partition])
    vehicles = []
    for car, group in snippet.groupby('car', sort=True):
        if group.label.nunique() != 1:
            raise ValueError('Inconsistent vehicle labels')
        values = group[SCORE_NAMES].to_numpy()
        count = max(1, int(np.ceil(.05*len(values))))
        top = np.partition(values, -count, axis=0)[-count:].mean(0)
        vehicles.append(dict(car=str(car), label=int(group.label.iloc[0]), snippets=len(group),
                             **dict(zip(SCORE_NAMES, top))))
    return snippet, pd.DataFrame(vehicles)


def evaluate(validation, test, fold):
    if not validation.label.eq(0).all() or set(test.label) != {0, 1}:
        raise ValueError('Invalid threshold or test cohort')
    rows, thresholds = [], {}
    for name in SCORE_NAMES:
        thresholds[name] = {}
        for quantile in [.95, .99]:
            threshold = float(validation[name].quantile(quantile))
            thresholds[name][str(quantile)] = threshold
            rows.append(dict(fold=fold, method=name, q=quantile, threshold=threshold,
                             **metrics(test.label, test[name], threshold)))
    return rows, thresholds


def memory_diagnostics(memory, features, mapping, output, args):
    diagnostics, cutoff = {}, None
    for partition in PARTITIONS:
        h = features[partition]
        labels, distance, margin = memory.query(h)
        flipped = np.empty(len(h), dtype=bool)
        rng = np.random.default_rng(args.seed+PARTITIONS.index(partition))
        for ix in slices(len(h), args.batch):
            z = memory.standardize(h[ix])
            perturbed = memory.query_standardized(z+rng.normal(0, .01, z.shape))[0]
            flipped[ix] = labels[ix] != perturbed
        if partition == 'train':
            cutoff = [float(np.quantile(distance[labels == j], .95)) for j in range(memory.k)]
        np.save(output/f'{partition}_assignments.npy', labels)
        np.save(output/f'{partition}_query_diagnostics.npy', np.column_stack((distance, margin, flipped)))
        snippets = pd.DataFrame(dict(sample_id=mapping['sample_ids'][partition],
            car=list(map(str, mapping['cars'][partition])), label=mapping['labels'][partition],
            prototype=labels, nearest_distance=distance, relative_margin=margin, perturb_changed=flipped,
            far=distance > np.asarray(cutoff)[labels]))
        snippets.to_csv(output/f'{partition}_snippet_prototypes.csv', index=False)
        vehicle_rows = []
        for car, frame in snippets.groupby('car', sort=True):
            counts = np.bincount(frame.prototype, minlength=memory.k)
            p = counts[counts > 0]/len(frame)
            vehicle_rows.append(dict(car=str(car), label=int(frame.label.iloc[0]), snippets=len(frame),
                dominant_prototype=int(counts.argmax()), dominant_fraction=float(counts.max()/len(frame)),
                prototypes_used=int((counts > 0).sum()), entropy=float(-(p*np.log(p)).sum()),
                perturb_flip_rate=float(frame.perturb_changed.mean()), temporal_switch_rate=None,
                **{f'prototype_{j}_snippets': int(counts[j]) for j in range(memory.k)}))
        pd.DataFrame(vehicle_rows).to_csv(output/f'{partition}_vehicle_prototypes.csv', index=False)
        distribution = [dict(label=int(label), prototype=j, snippets=int((frame.prototype == j).sum()),
            vehicles=int(frame.loc[frame.prototype == j, 'car'].nunique()))
            for label, frame in snippets.groupby('label') for j in range(memory.k)]
        pd.DataFrame(distribution).to_csv(output/f'{partition}_prototype_distribution.csv', index=False)
        diagnostics[partition] = dict(prototype_windows=np.bincount(labels, minlength=memory.k).tolist(),
            perturb_flip_rate=float(flipped.mean()), boundary_fraction=float(np.mean(margin < .05)),
            class_summary={str(label): dict(snippets=len(frame), vehicles=int(frame.car.nunique()),
                mean_distance=float(frame.nearest_distance.mean()), far_fraction=float(frame.far.mean()))
                for label, frame in snippets.groupby('label')}, temporal_switch_rate=None)
    atomic_json(output/'diagnostics.json', dict(partitions=diagnostics,
        space='training-standardized mean-pooled Encoder H, not residuals or controls-only statistics',
        query_columns=['nearest_mean_squared_distance', 'relative_margin', 'perturb_assignment_changed'],
        train_distance_p95_by_prototype=cutoff, perturb_std_training_h=.01,
        temporal_switching_available=False, reason='one native window per unordered snippet'))


def run_fold(base, output, fold, args):
    source = base/'sources'/f'fold{fold}'/'full'
    baseline = base/'experiments'/f'fold{fold}'/'A_current_rcr'
    features_root = output/'features'/f'fold{fold}'
    feature_record = json.loads((features_root/'complete.json').read_text())
    feature_protocol = json.loads((features_root/'protocol_manifest.json').read_text())
    mapping = json.loads((source/'mapping.json').read_text())
    backbone = json.loads((source/'backbone_complete.json').read_text())
    if feature_record['backbone_hash'] != backbone['checkpoint_hash'] or not feature_record['backbone_unchanged']:
        raise ValueError('Hidden features do not share the RCR Backbone')
    if feature_protocol['uses_residuals'] or feature_protocol['uses_calibration']:
        raise ValueError('Memory requires original-X Encoder features only')
    target = output/'experiments'/f'fold{fold}'
    target.mkdir(parents=True, exist_ok=True)
    settings = dict(dataset='Brand3', fold=fold, backbone_hash=backbone['checkpoint_hash'],
        frozen_RCR_baseline=str(baseline.resolve()), source_mapping_sha256=digest(source/'mapping.json'),
        feature_protocol_sha256=digest(features_root/'protocol_manifest.json'),
        features={p: dict(sha256=digest(features_root/f'{p}_features.npy')) for p in PARTITIONS},
        baseline_scores={p: digest(baseline/f'{p}_window_scores.npy') for p in PARTITIONS},
        representation='temporal Encoder output H before decoder; mean pool time',
        memory_fit='only normal training H; train-standardized K-means', uses_residuals_for_memory=False,
        ks=KS, primary_k=6, seed=args.seed, n_init=args.n_init, max_iter=args.max_iter,
        stability_seeds=args.stability_seeds, perturb_std_training_h=.01,
        score_names=SCORE_NAMES, fusion='0.5 * training_RCR_CDF + 0.5 * training_Memory_CDF per window',
        RCR_unchanged=True, extra_control='RCR_CDF to isolate CDF/aggregation effects',
        aggregation='snippet_top5pct_then_vehicle_top5pct', thresholds=[.95, .99], primary_threshold=.95,
        test_selection=False, complete_MEMTO_reproduction=False, threads=args.threads, batch=args.batch,
        implementation_sha256={Path(p).name: digest(p) for p in [__file__,
            Path(__file__).parents[1]/'models/rcr_feature_memory.py']})
    manifest = target/'protocol_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError('Hidden-memory protocol changed; use new output')
    atomic_json(manifest, settings)
    if (target/'complete.json').exists():
        print('ENCODER MEMORY FOLD ALREADY COMPLETE', fold, flush=True)
        return
    shutil.copyfile(source/'mapping.json', target/'mapping.json')
    features = {p: np.load(features_root/f'{p}_features.npy', mmap_mode='r') for p in PARTITIONS}
    for partition, h in features.items():
        if h.ndim != 2 or len(h) != len(mapping['sample_ids'][partition]):
            raise ValueError('Hidden feature/sample mapping mismatch')
        shutil.copyfile(source/'residuals'/partition/'references.npy', target/f'{partition}_references.npy')
    memories, cdfs, val_distances = {}, {}, {}
    for k in KS:
        directory = target/f'memory_k{k}'
        directory.mkdir(exist_ok=True)
        memory = EncoderFeatureMemory(k, args.seed, args.n_init, args.max_iter).fit(features['train'])
        memory.save(directory/'memory.npz')
        restored = EncoderFeatureMemory.load(directory/'memory.npz')
        np.testing.assert_array_equal(memory.score(features['train']), restored.score(features['train']))
        stability(memory, features['train'], features['validation'], args.stability_seeds, directory)
        pd.DataFrame(cdist(memory.centers, memory.centers)).to_csv(directory/'center_distances.csv', index_label='prototype')
        pd.DataFrame(memory.centers).to_csv(directory/'centers_standardized_H.csv', index_label='prototype')
        atomic_json(directory/'fit_audit.json', dict(normal_train_only=True, representation='Encoder H',
            residuals_used=False, training_windows=len(features['train']), prototype_windows=memory.counts.tolist(),
            iterations=memory.n_iter, inertia=memory.inertia, feature_mean=memory.feature_mean.tolist(),
            feature_std=memory.feature_std.tolist(), exact_memory_serialization=True))
        cdf = EmpiricalCDF().fit(memory.score(features['train']))
        np.save(directory/'training_distance_cdf.npy', cdf.reference)
        memories[k], cdfs[k] = memory, cdf
        val_distances[k] = float(memory.score(features['validation']).mean())
    # No test labels, F1, or AP participate in K/weight selection. K=6 is primary.
    atomic_json(target/'normal_validation_diagnostic.json', dict(mean_distance=val_distances,
        primary_k=6, all_ks_reported=True, test_labels_used=False, alpha_fixed=.5))
    vehicles, saturation = {}, {}
    for partition in PARTITIONS:
        original = np.load(baseline/f'{partition}_window_scores.npy', mmap_mode='r')
        scores = np.lib.format.open_memmap(target/f'{partition}_window_scores.npy', mode='w+',
                                           dtype='float64', shape=(len(original), len(SCORE_NAMES)))
        scores[:, :2] = original[:, 1:3]
        for k in KS:
            start = SCORE_NAMES.index(f'Memory_k{k}')
            distance = memories[k].score(features[partition])
            percentile = cdfs[k].transform(distance)
            scores[:, start] = distance
            scores[:, start+1] = percentile
            scores[:, start+2] = fuse_percentiles(scores[:, 1], percentile)
        scores.flush()
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite hidden Memory/fusion scores')
        np.testing.assert_array_equal(scores[:, :2], original[:, 1:3])
        saturation[partition] = {name: float(np.mean(scores[:, j] == 1))
            for j, name in enumerate(SCORE_NAMES) if 'CDF' in name or name.startswith('Fusion')}
        if partition != 'train':
            snippets, vehicles[partition] = aggregate_vehicle(scores, mapping, partition)
            snippets.to_csv(target/f'{partition}_snippet_scores.csv', index=False)
            vehicles[partition].to_csv(target/f'{partition}_vehicle_scores.csv', index=False)
            old = pd.read_csv(baseline/f'{partition}_vehicle_scores.csv', dtype={'car': str}).set_index('car').sort_index()
            new = vehicles[partition].set_index('car').sort_index()
            np.testing.assert_array_equal(new.label, old.label)
            np.testing.assert_allclose(new.RCR, old.magnitude, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(new.RCR_CDF, old.magnitude_cdf, rtol=1e-12, atol=1e-15)
    for k in KS:
        memory_diagnostics(memories[k], features, mapping, target/f'memory_k{k}', args)
    rows, thresholds = evaluate(vehicles['validation'], vehicles['test'], fold)
    atomic_json(target/'thresholds.json', thresholds)
    atomic_json(target/'metrics.json', rows)
    pd.DataFrame(rows).to_csv(target/'metrics.csv', index=False)
    atomic_json(target/'cdf_saturation.json', saturation)
    atomic_json(target/'complete.json', dict(fold=fold, score_names=SCORE_NAMES, backbone_hash=backbone['checkpoint_hash'],
        original_RCR_window_scores_unchanged=True, original_RCR_vehicle_scores_unchanged=True,
        memory_fit_uses_residuals=False, fusion_weight=.5, no_test_tuning=True))
    print('ENCODER MEMORY FOLD COMPLETE', fold, flush=True)
    print(pd.DataFrame(rows).query("q == .95 and method in ['RCR','RCR_CDF','Memory_k6','Fusion_k6']")
          [['method', 'ap', 'auroc', 'f1', 'fpr']].round(4).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='runs/internal/rcr_memory_brand3_20261025_v1')
    parser.add_argument('--output', default='runs/internal/rcr_encoder_memory_brand3_20261025_v1')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--n-init', type=int, default=5)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--stability-seeds', nargs='+', type=int, default=[3408, 3409])
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    output, base = Path(args.output).resolve(), Path(args.base).resolve()
    if not (base/'complete.json').exists() or not (output/'features/complete.json').exists():
        raise ValueError('Finish original Brand3 experiment and all frozen H extraction first')
    with threadpool_limits(limits=args.threads):
        for fold in range(5):
            run_fold(base, output, fold, args)


if __name__ == '__main__':
    main()
