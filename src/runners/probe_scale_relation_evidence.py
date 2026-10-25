"""Frozen-cache scale attribution and paired relation-error evidence (CPU)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.evaluate_rcr import metrics
from src.models.rcr import EmpiricalCDF
from src.models.rcr_memory import residual_moments
from src.models.local_residual_models import ReliableConditionalCalibration
from src.models.residual_scale_probe import (LAMBDAS, centered_energy, gaussian_nll,
    shrink_variance, scale_components, select_normal_only)
from src.runners.run_local_residual_models import (source_for, load_context, NAMES,
    spec_name, PARTITIONS)
from src.runners.run_rcr_calibration_sensitivity import aggregate
from src.runners.run_rcr_memory_probe import digest
from src.runners.run_rcr import atomic_json


TASK = 'RCR-SCALE-RELATION-EVIDENCE-20261025-v1'
UNITS = ['LFP', 'NCM']+[f'Brand3_fold{i}' for i in range(5)]
ROOT = Path('runs/internal/rcr_local_two_models_20261025_v1')
SCALE_PAIRS = [('A', 'Global_rms'), ('Global_rms', 'Log_conditional'),
    ('A', 'Robust_conditional'), ('A', 'Selected_NLL'), ('A', 'Selected_1SE'),
    ('Previous_selected', 'Selected_1SE')]
REL_PAIRS = [('A', 'Relation_conditional'), ('M1', 'Relation_conditional'),
    ('Marginal_diag', 'Relation_conditional'), ('Relation_unconditional', 'Relation_conditional'),
    ('A', 'A_Relation_fusion'), ('A_CDF', 'A_Relation_fusion'),
    ('A', 'A_Relation_max'), ('A', 'A_Diag_fusion'),
    ('A_Diag_fusion', 'A_Relation_fusion')]


def read(path):
    return pd.read_csv(path, dtype={'vin': str}, float_precision='round_trip')


def rank_frame(snippets, names, brand):
    if not brand:
        return snippets
    rows = []
    for vin, frame in snippets.groupby('vin', sort=True):
        assert frame.sample_label.nunique() == 1
        k = max(1, int(np.ceil(.05*len(frame))))
        values = np.partition(frame[names].to_numpy(), -k, axis=0)[-k:].mean(axis=0)
        rows.append(dict(vin=vin, sample_label=int(frame.sample_label.iloc[0]),
                         **dict(zip(names, values))))
    return pd.DataFrame(rows)


def evaluate(frames, names, unit, output):
    rows, thresholds = [], {}
    for name in names:
        thresholds[name] = {}
        for q in [.95, .99]:
            threshold = float(frames['validation'][name].quantile(q))
            thresholds[name][str(q)] = threshold
            rows.append(dict(unit=unit, method=name, q=q, threshold=threshold,
                validation_fpr=float(np.mean(frames['validation'][name] >= threshold)),
                **metrics(frames['test'].sample_label, frames['test'][name], threshold)))
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    atomic_json(output/'thresholds.json', thresholds)
    return pd.DataFrame(rows), thresholds


def scale_probe(unit, output, batch):
    source, _ = source_for(unit)
    previous = ROOT/unit
    output.mkdir(parents=True, exist_ok=True)
    mapping, context, brand = load_context(source, output)
    selection = json.loads((previous/'scale_selection.json').read_text())
    nlls = selection['normal_validation_nll']
    log_name = min((n for n in nlls if n.startswith('Full_floor_')), key=nlls.get)
    log_cal = ReliableConditionalCalibration.load(previous/(log_name+'.npz'))
    robust_cal = ReliableConditionalCalibration.load(previous/'Full_robust.npz')
    train = context['train']
    tm, tv = residual_moments(train['residual'], batch)
    energy = centered_energy(tm, tv, log_cal.location(train['q'], train['groups']))
    globals_ = dict(log=np.maximum(np.sqrt(energy.mean(0)), 1e-5),
                    robust=np.maximum(np.median(np.sqrt(energy), axis=0), 1e-5))
    np.savez(output/'global_scales.npz', **globals_)
    del tm, tv, energy
    candidates = {f'{family}_lambda{int(lam*100):03d}': (family, lam)
                  for family in ['log', 'robust'] for lam in LAMBDAS}
    names = ['A', 'Mean_only', 'Global_rms', 'Global_median', 'Log_conditional',
             'Robust_conditional', 'Log_shared_state', 'Log_variable_shape',
             'Previous_selected']+list(candidates)
    names += [f'Log_drop_{j}' for j in range(7)]
    frames, losses, diagnostics, contributions = {}, {}, [], []
    explicit_error = 0.
    for partition in PARTITIONS:
        c = context[partition]
        old = np.load(previous/f'{partition}_window_scores.npy', mmap_mode='r')
        score = np.lib.format.open_memmap(output/f'{partition}_window_scores.npy', mode='w+',
            dtype='float64', shape=(len(c['q']), len(names)))
        val_loss = {name: np.empty(len(c['q'])) for name in candidates} if partition == 'validation' else {}
        # Small Brand3 caches allow exact variable diagnostics for every window.
        diag_parts = []
        for lo in range(0, len(c['q']), batch):
            ix = slice(lo, min(lo+batch, len(c['q'])))
            q, g = c['q'][ix], c['groups'][ix]
            r = np.asarray(c['residual'][ix], dtype=np.float64)
            mu, sl = log_cal.parameters(q, g)
            sr = robust_cal.parameters(q, g)[1]
            e = centered_energy(r.mean(1), r.var(1), mu)
            shared, shape = scale_components(sl, globals_['log'])
            values = dict(A=old[ix, NAMES.index('A')], Mean_only=e.mean(1),
                Global_rms=(e/globals_['log']**2).mean(1),
                Global_median=(e/globals_['robust']**2).mean(1),
                Log_conditional=(e/sl**2).mean(1), Robust_conditional=(e/sr**2).mean(1),
                Log_shared_state=(e/shared**2).mean(1), Log_variable_shape=(e/shape**2).mean(1),
                Previous_selected=old[ix, NAMES.index('Full_selected')])
            for name, (family, lam) in candidates.items():
                scale = shrink_variance(sl if family == 'log' else sr, globals_[family], lam)
                values[name] = (e/scale**2).mean(1)
                if partition == 'validation':
                    val_loss[name][ix] = gaussian_nll(e, scale)
            normalized = e/sl**2
            for j in range(7):
                values[f'Log_drop_{j}'] = np.delete(normalized, j, axis=1).mean(1)
            for name, value in values.items():
                score[ix, names.index(name)] = value
            explicit_error = max(explicit_error, float(np.max(np.abs(
                np.square((r-mu[:, None, :])/sl[:, None, :]).mean((1, 2))-values['Log_conditional']))))
            # Diagnostic rows preserve original sample and VIN mappings.
            diag_parts.append((e, sl, sr))
        score.flush()
        snippets = c['frame'].merge(aggregate(score, c['refs'], mapping['sample_ids'][partition], names),
                                    on='sample_id', validate='one_to_one')
        frames[partition] = rank_frame(snippets, names, brand)
        if partition == 'validation':
            vins = c['frame'].vin.to_numpy()[np.asarray(c['refs'][:, 0], dtype=int)]
            best, conservative, records = select_normal_only(val_loss, vins,
                {n: lam for n, (_, lam) in candidates.items()})
            losses = dict(selected_nll=best, selected_one_se=conservative,
                log_family_source=log_name, selection_records=records,
                selection='VIN-equal normal validation NLL; one-SE prefers smallest lambda',
                test_labels_used=False)
            atomic_json(output/'selection.json', losses)
        e = np.concatenate([v[0] for v in diag_parts])
        sl = np.concatenate([v[1] for v in diag_parts])
        sr = np.concatenate([v[2] for v in diag_parts])
        sequence = np.asarray(c['refs'][:, 0], dtype=int)
        labels = c['frame'].sample_label.to_numpy()[sequence]
        for label in np.unique(labels):
            mask = labels == label
            for j, feature in enumerate(mapping['feature_columns']):
                for family, sc in [('log', sl), ('robust', sr)]:
                    v = e[mask]/sc[mask]**2
                    diagnostics.append(dict(unit=unit, partition=partition, label=int(label),
                        family=family, feature=feature, global_scale=float(globals_[family][j]),
                        centered_rms_p50=float(np.median(np.sqrt(e[mask, j]))),
                        sigma_p05=float(np.quantile(sc[mask, j], .05)),
                        sigma_p50=float(np.median(sc[mask, j])), sigma_p95=float(np.quantile(sc[mask, j], .95)),
                        inverse_variance_p50=float(np.median(1/sc[mask, j]**2)),
                        raw_energy_share=float(e[mask, j].sum()/e[mask].sum()),
                        scaled_energy_share=float(v[:, j].sum()/v.sum()),
                        mean_standardized_energy=float(v[:, j].mean())))
        # Train/val/test variance calibration and tail failure are diagnostic only.
        for family, sc in [('log', sl), ('robust', sr)]:
            z2 = e/sc**2
            for label in np.unique(labels):
                mask = labels == label
                for group in np.unique(c['groups'][mask]):
                    use = mask & (c['groups'] == group)
                    contributions.append(dict(unit=unit, partition=partition, family=family,
                        label=int(label), group=int(group), windows=int(use.sum()),
                        vins=int(c['frame'].iloc[np.unique(sequence[use])].vin.nunique()),
                        normal_or_fault_nll=float(gaussian_nll(e[use], sc[use]).mean()),
                        z2_mean=float(z2[use].mean()), z2_p95=float(np.quantile(z2[use].mean(1), .95))))
        snippets.to_csv(output/f'{partition}_snippet_scores.csv', index=False)
        print('SCALE SCORED', unit, partition, len(c['q']), flush=True)
    for partition, f in frames.items():
        f['Selected_NLL'] = f[losses['selected_nll']]
        f['Selected_1SE'] = f[losses['selected_one_se']]
        f.to_csv(output/f'{partition}_scores.csv', index=False)
    names += ['Selected_NLL', 'Selected_1SE']
    result, thresholds = evaluate(frames, names, unit, output)
    pd.DataFrame(diagnostics).to_csv(output/'variable_scale_diagnostics.csv', index=False)
    pd.DataFrame(contributions).to_csv(output/'state_tail_diagnostics.csv', index=False)
    # Exact ranking flips at the deployed vehicle aggregation level.
    f = frames['test']; normal, fault = f[f.sample_label == 0], f[f.sample_label == 1]
    pairs = []
    for base, contender in [('Mean_only', 'Global_rms'), ('Global_rms', 'Log_conditional'),
                             ('A', 'Previous_selected'), ('A', 'Selected_1SE')]:
        for _, a in fault.iterrows():
            for _, b in normal.iterrows():
                before = float(np.sign(a[base]-b[base]))
                after = float(np.sign(a[contender]-b[contender]))
                pairs.append(dict(unit=unit, base=base, contender=contender, fault_vin=a.vin,
                    normal_vin=b.vin, before=before, after=after, auc_pair_delta=(after-before)/2))
    pd.DataFrame(pairs).to_csv(output/'fault_normal_rank_pairs.csv', index=False)
    atomic_json(output/'audit.json', dict(explicit_energy_max_abs_error=explicit_error,
        full_mean_matches_previous_max_abs=float(np.max(np.abs(
            frames['test']['Mean_only'].to_numpy()-read(previous/'test_scores.csv').Full_mean.to_numpy()))),
        previous_source_manifest_sha256=digest(previous/'protocol_manifest.json')))
    atomic_json(output/'complete.json', dict(unit=unit, scale=True, test_tuning=False))
    return frames, result, thresholds


def error_overlap(frame, base, relation, bt, rt, unit, q, grouping='all', group='all'):
    y = frame.sample_label.to_numpy()
    a, b = frame[base].to_numpy() >= bt, frame[relation].to_numpy() >= rt
    row = dict(unit=unit, q=q, base=base, relation=relation, grouping=grouping, group=group,
               n=len(y), normal=int((y == 0).sum()), fault=int((y == 1).sum()))
    for label, prefix in [(0, 'normal'), (1, 'fault')]:
        m = y == label
        for state, mask in [('both', a & b), ('base_only', a & ~b), ('relation_only', ~a & b),
                            ('neither', ~a & ~b)]:
            row[prefix+'_'+state] = int((m & mask).sum())
    row['rescue_rate_among_base_FN'] = row['fault_relation_only']/max(1, row['fault_relation_only']+row['fault_neither'])
    row['added_FP_per_rescued_fault'] = row['normal_relation_only']/max(1, row['fault_relation_only'])
    row['fault_unique_vins_rescued'] = int(frame.loc[(y == 1) & ~a & b, 'vin'].nunique())
    row['normal_unique_vins_added_FP'] = int(frame.loc[(y == 0) & ~a & b, 'vin'].nunique())
    return row


def relation_probe(unit, output, batch):
    previous = ROOT/unit
    brand = unit.startswith('Brand3')
    mapping = json.loads((previous/'mapping.json').read_text())
    source, _ = source_for(unit)
    output.mkdir(parents=True, exist_ok=True)
    extras = ['A_CDF', 'A_Relation_fusion', 'A_Relation_max', 'A_Diag_fusion']
    keep = ['A', 'M1', 'Marginal_diag', 'Relation_unconditional', 'Relation_conditional', 'M2']
    names = keep+extras
    frames, cdfs, snippets_by_part = {}, {}, {}
    for partition in PARTITIONS:
        old = np.load(previous/f'{partition}_window_scores.npy', mmap_mode='r')
        if partition == 'train':
            for name in ['A', 'Relation_conditional', 'Marginal_diag']:
                cdfs[name] = EmpiricalCDF().fit(old[:, NAMES.index(name)])
            np.savez(output/'normal_train_CDFs.npz', **{n: c.reference for n, c in cdfs.items()})
        values = np.lib.format.open_memmap(output/f'{partition}_window_scores.npy', mode='w+',
            dtype='float64', shape=(len(old), len(names)))
        for lo in range(0, len(old), batch):
            ix = slice(lo, min(len(old), lo+batch))
            for name in keep:
                values[ix, names.index(name)] = old[ix, NAMES.index(name)]
            p = {n: c.transform(old[ix, NAMES.index(n)]) for n, c in cdfs.items()}
            values[ix, names.index('A_CDF')] = p['A']
            values[ix, names.index('A_Relation_fusion')] = .5*(p['A']+p['Relation_conditional'])
            values[ix, names.index('A_Relation_max')] = np.maximum(p['A'], p['Relation_conditional'])
            values[ix, names.index('A_Diag_fusion')] = .5*(p['A']+p['Marginal_diag'])
        values.flush()
        refs = np.load(source/'residuals'/partition/'references.npy', mmap_mode='r')
        meta = read(previous/f'{partition}_snippet_scores.csv').drop(columns=NAMES)
        snippets = meta.merge(aggregate(values, refs, mapping['sample_ids'][partition], names),
                              on='sample_id', validate='one_to_one')
        snippets.to_csv(output/f'{partition}_snippet_scores.csv', index=False)
        snippets_by_part[partition] = snippets
        frames[partition] = rank_frame(snippets, names, brand)
        frames[partition].to_csv(output/f'{partition}_scores.csv', index=False)
    result, thresholds = evaluate(frames, names, unit, output)
    test = frames['test']
    overlap, predictions = [], []
    compare = [('A', 'Relation_conditional'), ('M1', 'Relation_conditional'),
               ('Marginal_diag', 'Relation_conditional'), ('Relation_unconditional', 'Relation_conditional')]
    for q in [.95, .99]:
        for base, relation in compare:
            bt, rt = thresholds[base][str(q)], thresholds[relation][str(q)]
            overlap.append(error_overlap(test, base, relation, bt, rt, unit, q))
            rows = test[[c for c in ['sample_id', 'vin', 'sample_label', 'fault_type',
                                    'operating_regime', 'cycle_kind'] if c in test]].copy()
            rows['base'], rows['relation'], rows['q'] = base, relation, q
            rows['base_score'], rows['relation_score'] = test[base], test[relation]
            rows['base_threshold'], rows['relation_threshold'] = bt, rt
            rows['base_alarm'], rows['relation_alarm'] = test[base] >= bt, test[relation] >= rt
            rows['set'] = np.select([rows.base_alarm & rows.relation_alarm,
                rows.base_alarm & ~rows.relation_alarm, ~rows.base_alarm & rows.relation_alarm],
                ['both', 'base_only', 'relation_only'], default='neither')
            predictions.append(rows)
            if not brand:
                for grouping in ['fault_type', 'operating_regime']:
                    for group, sub in test.groupby(grouping):
                        overlap.append(error_overlap(sub, base, relation, bt, rt, unit, q, grouping, str(group)))
    pd.DataFrame(overlap).to_csv(output/'error_overlap.csv', index=False)
    pd.concat(predictions).to_csv(output/'prediction_sets.csv', index=False)
    # Raw OR is a coverage diagnostic and carries its own validation alarm cost.
    or_rows = []
    for q in [.95, .99]:
        for base, relation in compare:
            bt, rt = thresholds[base][str(q)], thresholds[relation][str(q)]
            alarm = (test[base] >= bt) | (test[relation] >= rt)
            val = frames['validation']
            or_rows.append(dict(unit=unit, q=q, base=base, relation=relation,
                validation_fpr=float(((val[base] >= bt) | (val[relation] >= rt)).mean()),
                **metrics(test.sample_label, alarm.astype(float), .5)))
    pd.DataFrame(or_rows).to_csv(output/'unbudgeted_OR_diagnostic.csv', index=False)
    for name in keep:
        np.testing.assert_array_equal(test[name].to_numpy(), read(previous/'test_scores.csv')[name].to_numpy())
    atomic_json(output/'audit.json', dict(original_ranking_scores_exact=True,
        fusion_weight=.5, fusion_before_top5=True, normal_train_cdfs=True,
        independent_normal_validation_thresholds=True))
    atomic_json(output/'complete.json', dict(unit=unit, relation=True, test_tuning=False))
    print('RELATION COMPLETE', unit, flush=True)
    return frames, result, thresholds


def weighted_metric(y, score, threshold, weights):
    """Weighted AP/AUROC/F1/recall/FPR with exact ties (no rank approximation)."""
    order = np.argsort(-score, kind='stable')
    s, target, w = score[order], y[order], weights[order]
    end = np.r_[np.flatnonzero(np.diff(s))+1, len(s)]-1
    tp, fp = np.cumsum(w*target)[end], np.cumsum(w*(1-target))[end]
    pos, neg = tp[-1], fp[-1]
    if pos <= 0 or neg <= 0:
        return None
    ap = float(np.sum(np.diff(np.r_[0., tp])/pos*tp/np.maximum(tp+fp, 1e-30)))
    auc = float(np.trapezoid(np.r_[0., tp]/pos, np.r_[0., fp]/neg))
    alarm = score >= threshold
    true = weights[alarm & (y == 1)].sum()
    false = weights[alarm & (y == 0)].sum()
    return np.array([ap, auc, 2*true/max(1e-30, pos+true+false), true/pos, false/neg])


def paired_bootstrap(entries, pairs, repeats, dataset, q=.95):
    """Unique VIN weights shared across methods, snippets and repeated folds."""
    identities = pd.concat([f['test'][['vin', 'sample_label']] for f, _ in entries]).drop_duplicates()
    assert identities.groupby('vin').sample_label.nunique().max() == 1
    identities = identities.groupby('vin').sample_label.first().sort_index()
    vins = identities.index.to_numpy()
    groups = [np.flatnonzero(identities.to_numpy() == y) for y in [0, 1]]
    prepared = []
    methods = sorted(set(sum(([a, b] for a, b in pairs), [])))
    for f, thresholds in entries:
        frame = f['test']; codes = pd.Index(vins).get_indexer(frame.vin)
        prepared.append((frame.sample_label.to_numpy(), codes,
            {n: (frame[n].to_numpy(), thresholds[n][str(q)]) for n in methods}))

    def calculate(counts):
        vals = {n: [] for n in methods}
        for y, codes, scores in prepared:
            w = counts[codes]
            for name, (score, threshold) in scores.items():
                metric = weighted_metric(y, score, threshold, w)
                if metric is None:
                    return None
                vals[name].append(metric)
        macro = {n: np.mean(v, axis=0) for n, v in vals.items()}
        return np.stack([macro[b]-macro[a] for a, b in pairs])

    observed = calculate(np.ones(len(vins)))
    draws, rng = [], np.random.default_rng(3407)
    for _ in range(repeats):
        counts = np.zeros(len(vins))
        for group in groups:
            counts += np.bincount(rng.choice(group, len(group), replace=True), minlength=len(vins))
        delta = calculate(counts)
        if delta is not None:
            draws.append(delta)
    low, high = np.quantile(draws, [.025, .975], axis=0)
    return [dict(dataset=dataset, q=q, base=a, contender=b, metric=metric,
        delta=float(observed[j, k]), ci_low=float(low[j, k]), ci_high=float(high[j, k]),
        valid_draws=len(draws), requested_draws=repeats, unique_vins=len(vins),
        unique_fault_vins=len(groups[1])) for j, (a, b) in enumerate(pairs)
        for k, metric in enumerate(['ap', 'auroc', 'f1', 'recall', 'fpr'])]


def collect(output, repeats):
    all_metrics, ci_rows = [], []
    for part, units, pairs in [('scale', UNITS[2:], SCALE_PAIRS), ('relation', UNITS, REL_PAIRS)]:
        for dataset in (['Brand3'] if part == 'scale' else ['LFP', 'NCM', 'Brand3']):
            selected_units = [u for u in units if (u.startswith('Brand3') if dataset == 'Brand3' else u == dataset)]
            entries = []
            for unit in selected_units:
                p = output/part/unit
                all_metrics.append(read(p/'metrics.csv').assign(part=part, dataset=dataset))
                frames = {s: read(p/f'{s}_scores.csv') for s in ['validation', 'test']}
                entries.append((frames, json.loads((p/'thresholds.json').read_text())))
            for q in [.95, .99]:
                ci_rows += [dict(part=part, **r) for r in paired_bootstrap(entries, pairs, repeats, dataset, q)]
            print('BOOTSTRAP COMPLETE', part, dataset, flush=True)
    all_metrics = pd.concat(all_metrics, ignore_index=True)
    all_metrics.to_csv(output/'all_metrics.csv', index=False)
    summary = all_metrics.groupby(['part', 'dataset', 'method', 'q'])[
        ['ap', 'auroc', 'precision', 'recall', 'f1', 'fpr', 'validation_fpr']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(output/'summary_metrics.csv', index=False)
    pd.DataFrame(ci_rows).to_csv(output/'paired_VIN_intervals.csv', index=False)
    for filename in ['error_overlap.csv', 'unbudgeted_OR_diagnostic.csv']:
        frames = [read(output/'relation'/u/filename) for u in UNITS]
        pd.concat(frames, ignore_index=True).to_csv(output/filename, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/internal/rcr_scale_relation_evidence_20261025_v1')
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(task=TASK, prior_root=str(ROOT), frozen_backbone=True, device='cpu', gpu='unused',
        lambdas=LAMBDAS, variance_mixture='(1-lambda)*global_variance+lambda*conditional_variance',
        selection='normal validation VIN-equal NLL and one-SE conservative rule',
        robust='median normal training window centered RMS; not pointwise MAD',
        global_rms='sqrt(mean normal training centered energy) per variable',
        condition_mean='unchanged prior Full conditional mean; no reliability gate',
        thresholds=[.95, .99], fusion_alpha=.5, bootstrap=args.bootstrap, seed=3407,
        test_selection=False, threads=args.threads,
        implementations={str(p): digest(p) for p in [Path(__file__), Path('src/models/residual_scale_probe.py')]},
        source_manifests={u: digest(ROOT/u/'protocol_manifest.json') for u in UNITS})
    mp = output/'protocol_manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError('Protocol changed; use a new output directory')
    atomic_json(mp, manifest)
    with threadpool_limits(limits=args.threads):
        if not args.collect_only:
            for unit in UNITS[2:]:
                p = output/'scale'/unit
                if not (p/'complete.json').exists():
                    scale_probe(unit, p, args.batch)
            for unit in UNITS:
                p = output/'relation'/unit
                if not (p/'complete.json').exists():
                    relation_probe(unit, p, args.batch)
        collect(output, args.bootstrap)
    atomic_json(output/'complete.json', dict(task=TASK, scale_units=5, relation_units=7,
                                           bootstrap=args.bootstrap, test_tuning=False))
    print('SCALE RELATION EVIDENCE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
