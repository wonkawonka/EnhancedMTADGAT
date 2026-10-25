"""Two minimal normal-only residual models on CH and frozen Brand3 folds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.evaluate_rcr import metrics
from src.models.rcr import EmpiricalCDF, LinearResidualCalibration
from src.models.rcr_memory import residual_moments
from src.models.rcr_fitting import slices
from src.models.local_residual_models import (FLOORS, ReliableConditionalCalibration,
    ConditionalLOORelation, fit_reliability, augmented_second, gram_from_moments)
from src.runners.run_rcr import atomic_json
from src.runners.run_rcr_lite import cached
from src.runners.run_rcr_memory_probe import digest
from src.runners.run_rcr_calibration_sensitivity import aggregate


PARTITIONS = ['train', 'validation', 'test']
SPECS = [str(v) for v in FLOORS]+['robust']
def spec_name(spec):
    return 'robust' if spec == 'robust' else 'floor_'+spec.replace('.', 'p')

CANDIDATES = [f'{family}_{spec_name(spec)}' for family in ['Full', 'Reliable'] for spec in SPECS]
NAMES = ['Raw', 'A', 'Full_mean', 'Reliable_mean']+CANDIDATES+[
    'Full_selected', 'M1', 'M1_CDF', 'Marginal_diag', 'Relation_unconditional',
    'Relation_conditional', 'Relation_CDF', 'M2', 'M1_Diag_CDF', 'M2_diag_control', 'M2_unconditional']
PRIMARY = ['Raw', 'A', 'Full_selected', 'M1', 'M1_CDF', 'Marginal_diag',
           'Relation_unconditional', 'Relation_conditional', 'M2', 'M2_diag_control', 'M2_unconditional']


def source_for(unit):
    if unit in ['LFP', 'NCM']:
        source = Path('runs/internal/rcr_v1')/unit/'full'
        reference = source/'lite_continuous'
    elif unit.startswith('Brand3_fold') and unit[-1] in '01234':
        fold = int(unit[-1]); base = Path('runs/internal/rcr_memory_brand3_20261025_v1')
        source, reference = base/'sources'/f'fold{fold}'/'full', base/'experiments'/f'fold{fold}'/'A_current_rcr'
    else:
        raise ValueError('Unknown unit')
    return source.resolve(), reference.resolve()


def load_context(source, output):
    mapping = json.loads((source/'mapping.json').read_text())
    metadata = pd.read_csv(source/'metadata.csv', dtype={'vin': str, 'car': str}).set_index('sample_id', verify_integrity=True)
    brand = mapping.get('dataset') == 'Brand3'
    if brand:
        metadata['vin'] = metadata.car
        metadata['cycle_kind'] = 'charge'
        metadata['fault_type'] = np.where(metadata.sample_label.eq(0), 'normal', 'unavailable')
    context = {}
    for partition in PARTITIONS:
        r, q, refs, _ = cached(source, partition)
        ids = mapping['sample_ids'][partition]
        frame = metadata.loc[ids].reset_index()
        if partition != 'test' and not frame.sample_label.eq(0).all():
            raise ValueError('Normal fitting/threshold partition contains faults')
        boundaries = np.r_[0, np.flatnonzero(np.diff(refs[:, 0]))+1, len(refs)]
        if len(boundaries) != len(ids)+1 or not np.array_equal(refs[boundaries[:-1], 0], np.arange(len(ids))):
            raise ValueError('Incomplete snippet mapping')
        qc = np.add.reduceat(np.asarray(q, dtype=np.float64), boundaries[:-1], axis=0)/np.diff(boundaries)[:, None]
        frame['current_mean'], frame['soc_mean'] = qc[:, 0], qc[:, 4]
        context[partition] = dict(residual=r, q=q, refs=refs, frame=frame)
    edges = {}
    cycles = sorted(context['train']['frame'].cycle_kind.unique())
    for cycle in cycles:
        f = context['train']['frame'].query('cycle_kind == @cycle')
        edges[cycle] = {name: np.unique(np.quantile(f[name], [1/3, 2/3])).tolist() for name in ['current_mean', 'soc_mean']}
    for partition, c in context.items():
        f = c['frame']; groups = np.empty(len(f), dtype=np.int16)
        f['current_bin'], f['soc_bin'] = 0, 0
        for cycle in cycles:
            mask = f.cycle_kind == cycle
            for name, key in [('current_mean', 'current_bin'), ('soc_mean', 'soc_bin')]:
                f.loc[mask, key] = np.searchsorted(edges[cycle][name], f.loc[mask, name], side='right')+1
            groups[mask] = cycles.index(cycle)*9+(f.loc[mask, 'current_bin']-1)*3+f.loc[mask, 'soc_bin']-1
        if (f.current_bin == 0).any():
            raise ValueError('Unseen cycle type')
        f['operating_regime'] = f.cycle_kind+'_I'+f.current_bin.astype(str)+'_S'+f.soc_bin.astype(str)
        # Diagnostics keep the historical snippet bins. The actual gate queries
        # current-window controls, so it can respond within a nonstationary clip.
        sequence = np.asarray(c['refs'][:, 0], dtype=int)
        window_cycles = f.cycle_kind.to_numpy()[sequence]
        window_groups = np.empty(len(sequence), dtype=np.int16)
        for cycle in cycles:
            mask = window_cycles == cycle
            current_bin = np.searchsorted(edges[cycle]['current_mean'], c['q'][mask, 0], side='right')
            soc_bin = np.searchsorted(edges[cycle]['soc_mean'], c['q'][mask, 4], side='right')
            window_groups[mask] = cycles.index(cycle)*9+current_bin*3+soc_bin
        c['groups'] = window_groups
        c['vehicles'] = pd.factorize(f.vin, sort=True)[0][np.asarray(c['refs'][:, 0], dtype=int)]
    atomic_json(output/'regime_edges.json', edges)
    return mapping, context, brand


def fit_gram(c, calibration, q_stats, batch):
    total = np.zeros((72, 72))
    for ix in slices(len(c['q']), batch):
        z = calibration.transform(c['q'][ix], c['residual'][ix], c['groups'][ix])
        second = augmented_second(z)
        q = ConditionalLOORelation.condition(c['q'][ix], *q_stats)
        total += gram_from_moments(q, second)
    return total/len(c['q'])


def rank_level(snippets, brand):
    if not brand:
        return snippets
    rows = []
    for vin, frame in snippets.groupby('vin', sort=True):
        if frame.sample_label.nunique() != 1:
            raise ValueError('Conflicting vehicle label')
        values = frame[NAMES].to_numpy()
        count = max(1, int(np.ceil(.05*len(values))))
        top = np.partition(values, -count, axis=0)[-count:].mean(0)
        rows.append(dict(vin=str(vin), sample_label=int(frame.sample_label.iloc[0]), snippets=len(frame), **dict(zip(NAMES, top))))
    return pd.DataFrame(rows)


def score_and_evaluate(source, reference, output, mapping, context, brand, base_cal, calibrations, selected, relations, args):
    frames, cdfs = {}, {}
    selected_full, selected_m1 = calibrations['Full_'+spec_name(selected)], calibrations['Reliable_'+spec_name(selected)]
    for partition in PARTITIONS:
        c = context[partition]; n = len(c['q'])
        score = np.lib.format.open_memmap(output/f'{partition}_window_scores.npy', mode='w+', dtype='float64', shape=(n, len(NAMES)))
        ref_path = reference/(partition+'_window_scores.npy') if brand else reference/partition/'scores.npy'
        original = np.load(ref_path, mmap_mode='r')
        for ix in slices(n, args.batch):
            q, g = c['q'][ix], c['groups'][ix]
            r = np.asarray(c['residual'][ix], dtype=np.float64)
            means, variance = r.mean(1), r.var(1)
            mu = base_cal.parameters(q)[0]
            values = dict(Raw=original[ix, 0], A=original[ix, 1], Full_mean=(variance+np.square(means-mu)).mean(1),
                Reliable_mean=(variance+np.square(means-selected_m1.location(q, g))).mean(1))
            for name, cal in calibrations.items():
                values[name] = cal.energy(q, means, variance, g)
            values['Full_selected'] = values['Full_'+spec_name(selected)]
            values['M1'] = values['Reliable_'+spec_name(selected)]
            second = augmented_second(selected_m1.transform(q, r, g))
            values['Marginal_diag'] = relations['conditional'].diagonal_score(second)
            values['Relation_unconditional'] = relations['unconditional'].score_moments(q, second)
            values['Relation_conditional'] = relations['conditional'].score_moments(q, second)
            for name, value in values.items():
                score[ix, NAMES.index(name)] = value
        if partition == 'train':
            for name in ['M1', 'Marginal_diag', 'Relation_unconditional', 'Relation_conditional']:
                cdfs[name] = EmpiricalCDF().fit(score[:, NAMES.index(name)])
            np.savez(output/'normal_train_CDFs.npz', **{k: v.reference for k, v in cdfs.items()})
        for ix in slices(n, args.batch):
            percentiles = {name: cdf.transform(score[ix, NAMES.index(name)]) for name, cdf in cdfs.items()}
            score[ix, NAMES.index('M1_CDF')] = percentiles['M1']
            score[ix, NAMES.index('M1_Diag_CDF')] = percentiles['Marginal_diag']
            score[ix, NAMES.index('Relation_CDF')] = percentiles['Relation_conditional']
            score[ix, NAMES.index('M2')] = .5*(percentiles['M1']+percentiles['Relation_conditional'])
            score[ix, NAMES.index('M2_diag_control')] = .5*(percentiles['Marginal_diag']+percentiles['Relation_conditional'])
            score[ix, NAMES.index('M2_unconditional')] = .5*(percentiles['M1']+percentiles['Relation_unconditional'])
            if not np.isfinite(score[ix]).all():
                raise FloatingPointError('Nonfinite model scores')
        score.flush()
        snippets = aggregate(score, c['refs'], mapping['sample_ids'][partition], NAMES)
        snippets = c['frame'].merge(snippets, on='sample_id', validate='one_to_one')
        snippets.to_csv(output/f'{partition}_snippet_scores.csv', index=False)
        frames[partition] = rank_level(snippets, brand)
        frames[partition].to_csv(output/f'{partition}_scores.csv', index=False)
        print('LOCAL MODELS SCORED', partition, n, flush=True)
    thresholds, rows = {}, []
    for name in NAMES:
        thresholds[name] = {}
        for quantile in [.95, .99]:
            threshold = float(frames['validation'][name].quantile(quantile))
            thresholds[name][str(quantile)] = threshold
            rows.append(dict(method=name, q=quantile, threshold=threshold,
                **metrics(frames['test'].sample_label, frames['test'][name], threshold)))
    atomic_json(output/'thresholds.json', thresholds)
    atomic_json(output/'metrics.json', json.loads(pd.DataFrame(rows).to_json(orient='records', double_precision=15)))
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    local_rows, compositions = [], []
    test = pd.read_csv(output/'test_snippet_scores.csv', dtype={'vin': str}, float_precision='round_trip')
    for regime, frame in test.groupby('operating_regime'):
        normal, fault = frame[frame.sample_label == 0], frame[frame.sample_label == 1]
        for fault_type, affected in fault.groupby('fault_type'):
            if len(normal) < 10 or len(affected) < 10:
                continue
            group = pd.concat([normal, affected])
            ranked = rank_level(group, brand)
            if ranked.groupby('sample_label').vin.nunique().min() < (3 if brand else 1):
                continue
            common = dict(group=regime+'/'+str(fault_type), regime=regime, fault_type=fault_type,
                normal_snippets=len(normal), fault_snippets=len(affected),
                normal_vins=normal.vin.nunique(), fault_vins=affected.vin.nunique())
            for name in PRIMARY:
                threshold = thresholds[name]['0.95']
                local_rows.append(dict(**common, method=name,
                    **metrics(ranked.sample_label, ranked[name], threshold)))
            for subtype, subgroup in affected.groupby('fault_type'):
                compositions.append(dict(group=common['group'], regime=regime, fault_type=subtype,
                    snippets=len(subgroup), vins=subgroup.vin.nunique(),
                    fault_share_within_regime=len(subgroup)/len(fault)))
    pd.DataFrame(local_rows).to_csv(output/'local_metrics.csv', index=False)
    pd.DataFrame(compositions).to_csv(output/'local_fault_composition.csv', index=False)
    return frames, rows


def run(unit, args):
    source, reference = source_for(unit)
    output = Path(args.output).resolve()/unit
    output.mkdir(parents=True, exist_ok=True)
    base_protocol = json.loads((source/'protocol.json').read_text())
    if base_protocol['smoke']:
        raise ValueError('Full data required')
    settings = dict(task='RCR-LOCAL-TWO-MODELS-20261025-v1', unit=unit, source=str(source),
        reference=str(reference), backbone=json.loads((source/'backbone_complete.json').read_text()),
        source_mapping_sha256=digest(source/'mapping.json'), source_calibration_sha256=digest(source/'lite_continuous/calibration.npz'),
        implementations={Path(p).name: digest(p) for p in [__file__, Path(__file__).parents[1]/'models/local_residual_models.py']},
        normal_only=True, folds_for_normal_reliability=5, seed=3407, gate_shrinkage_vehicle_count=10,
        gate_min_vehicles=5, extrapolation_attenuation='normal_train_q_radius_P99',
        scales=SPECS, robust='median normal window RMS by operating state, minimum 50 windows',
        scale_selection='minimum full-calibration normal-validation NLL; same scale spec for full and reliable',
        relation='Gaussian ridge LOO, unconditional versus q*z interactions; target excluded',
        ridge_grid=[.0001, .01, 1.], ridge_selection='normal validation NLL',
        relation_condition_clip='normal train q P1/P99', fusion_alpha=.5, test_tuning=False,
        ranking='vehicle' if unit.startswith('Brand3') else 'snippet', threshold_quantiles=[.95, .99],
        device='cpu', gpu='unused', threads=args.threads, batch=args.batch)
    manifest = output/'protocol_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError('Protocol changed; use new output')
    if (output/'complete.json').exists():
        print('LOCAL MODELS ALREADY COMPLETE', unit, flush=True)
        return
    atomic_json(manifest, settings)
    mapping, context, brand = load_context(source, output)
    atomic_json(output/'mapping.json', mapping)
    base = LinearResidualCalibration.load(source/'lite_continuous/calibration.npz')
    train, val = context['train'], context['validation']
    print('LOCAL MODEL FIT START', unit, mapping['windows'], flush=True)
    means, variances = residual_moments(train['residual'], args.batch)
    vm, vv = residual_moments(val['residual'], args.batch)
    reliability = fit_reliability(train['q'], means, train['vehicles'], train['groups'], base)
    pd.DataFrame(reliability['audit']).to_csv(output/'gate_audit.csv', index=False)
    np.save(output/'normal_training_crossfit_folds.npy', reliability['folds'])
    calibrations, nlls = {}, {}
    for family, kind in [('Full', 'full'), ('Reliable', 'reliable')]:
        for spec in SPECS:
            name = family+'_'+spec_name(spec)
            cal = ReliableConditionalCalibration(base, reliability, kind, spec).fit_scale(train['q'], means, variances, train['groups'])
            cal.save(output/(name+'.npz'))
            nlls[name] = cal.normal_nll(val['q'], vm, vv, val['groups'])
            calibrations[name] = cal
    selected_name = min((n for n in nlls if n.startswith('Full_')), key=nlls.get)
    selected = SPECS[[spec_name(s) for s in SPECS].index(selected_name.removeprefix('Full_'))]
    atomic_json(output/'scale_selection.json', dict(normal_validation_nll=nlls, selected_scale=selected,
        selected_by_full_calibration=True, matched_for_reliable=True, test_labels_used=False))
    chosen = calibrations['Reliable_'+spec_name(selected)]
    chosen.save(output/'M1_calibration.npz')
    del means, variances, vm, vv
    q = np.asarray(train['q'], dtype=np.float64)
    qm, qs = q.mean(0), np.maximum(q.std(0), 1e-8)
    low, high = np.quantile((q-qm)/qs, [.01, .99], axis=0)
    q_stats = qm, qs, low, high
    print('FITTING FULL NORMAL JOINT MOMENTS', unit, 'scale', selected, flush=True)
    train_gram = fit_gram(train, chosen, q_stats, args.batch)
    val_gram = fit_gram(val, chosen, q_stats, args.batch)
    np.savez(output/'normal_joint_sufficient_moments.npz', train=train_gram, validation=val_gram)
    relations = {}
    for kind, conditional in [('unconditional', False), ('conditional', True)]:
        model = ConditionalLOORelation.fit(train_gram, val_gram, 8, 7, q_stats, conditional)
        model.save(output/('relation_'+kind+'.npz'))
        atomic_json(output/('relation_'+kind+'_selection.json'), dict(candidates=model.selection,
            selected_ridge=model.ridge, test_labels_used=False, target_excluded_from_all_predictors=True))
        relations[kind] = model
    frames, rows = score_and_evaluate(source, reference, output, mapping, context, brand, base, calibrations, selected, relations, args)
    atomic_json(output/'complete.json', dict(unit=unit, primary_methods=PRIMARY, score_names=NAMES,
        selected_scale=selected, normal_training_only=True, frozen_backbone=True, no_test_tuning=True))
    print(pd.DataFrame(rows).query('q == .95 and method in @PRIMARY')[['method', 'ap', 'auroc', 'f1', 'fpr']].round(4).to_string(index=False), flush=True)
    print('LOCAL TWO MODELS COMPLETE', unit, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--units', nargs='+', default=['NCM', 'LFP']+[f'Brand3_fold{k}' for k in range(5)])
    parser.add_argument('--output', default='runs/internal/rcr_local_two_models_20261025_v1')
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        for unit in args.units:
            run(unit, args)


if __name__ == '__main__':
    main()
