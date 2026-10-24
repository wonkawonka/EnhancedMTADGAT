"""Vehicle-isolated Brand2/3 reference models and seven coarse RCR settings.

The core model implementations are reused unchanged. Every unit saves its
mapping, window scores, snippet/vehicle scores, normal thresholds and counts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.external_protocol import load_external_protocol_data
from src.runners import run_unified_external_baselines as reference
from src.runners.run_rcr import atomic_checkpoint, backbone_train, cache_residuals

REFERENCES = ('mtad_gat', 'tranad', 'anomaly_transformer', 'dcdetector',
              'patchad', 'omnianomaly', 'isolation_forest', 'deep_svdd')
RCR_ARMS = ('raw', 'scale', 'lite', 'net', 'net_no_condition',
            'raw_no_feature_graph', 'raw_no_temporal')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


@contextmanager
def stage(label):
    started = time.monotonic()
    print('START', label, flush=True)
    try:
        yield
    except BaseException:
        print(f'FAILED {label} seconds={time.monotonic()-started:.1f}', flush=True)
        raise
    else:
        print(f'DONE {label} seconds={time.monotonic()-started:.1f}', flush=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def validate_plan(plan):
    assert plan['brand'] in (2, 3) and plan['model_seed'] == 3407 and plan['fold_seed'] == 0
    assert plan['split'] == 'strict_normal_validation' and plan['normalization'] == 'minmax'
    assert plan['threshold_quantiles'] == [.95, .99] and plan['vehicle_top_ratio'] == .05
    covered = []; units = []; owners = []
    for name, shard in plan['shards'].items():
        owners.append(shard['account'])
        assert shard['dataset_source'].split('/')[0] == shard['account']
        for job in shard['jobs']:
            assert job['fold'] in range(5)
            units.append(job['unit'])
            if job['kind'] == 'rcr':
                assert job['unit'] == f'rcr/fold{job["fold"]}'
                covered.extend((arm, job['fold']) for arm in RCR_ARMS)
            else:
                assert job['kind'] == 'reference' and job['method'] in REFERENCES
                assert job['unit'] == f'references/{job["method"]}/fold{job["fold"]}'
                covered.append((job['method'], job['fold']))
    assert len(plan['shards']) == 4 and owners.count('daisychen2') == owners.count('chenmjdaisy') == 2
    assert len(units) == len(set(units)) == 45
    assert len(covered) == len(set(covered)) == 75
    assert set(covered) == {(method, fold) for method in (*REFERENCES, *RCR_ARMS) for fold in range(5)}


def verify_mapping(mapping):
    cars = {p: set(mapping['vehicles'][p]) for p in ('train', 'validation', 'test')}
    assert not (cars['train'] & cars['validation'] or cars['train'] & cars['test']
                or cars['validation'] & cars['test']), 'Vehicle leakage'
    for p in ('train', 'validation', 'test'):
        ids = mapping['sample_ids'][p]
        assert len(ids) == len(set(ids)) == len(mapping['cars'][p]) == len(mapping['labels'][p])
        assert set(mapping['cars'][p]) == cars[p]
        if p != 'test':
            assert set(mapping['labels'][p]) == {0}, 'Fault in normal-only training/validation'
    assert set(mapping['labels']['test']) == {0, 1}
    assert len(mapping['feature_columns']) == 7


def load_fold(brand, fold, smoke):
    with stage(f'load Brand{brand} fold{fold} train-normal MinMax'):
        data = load_external_protocol_data(
            f'Brand{brand}', brand_fold=fold, brand_fold_seed=0, seed=3407,
            brand_split_protocol='strict_normal_validation', brand_normalization='minmax', brand_progress=True)
    sequences = dict(train=data.train_sequences, validation=data.validation_sequences,
                     test=data.test_sequences)
    ids = data.metadata['sample_ids']
    cars = dict(train=[value.split(':', 1)[0] for value in ids['train']],
                validation=data.validation_entity_ids, test=data.entity_ids)
    labels = dict(train=[0] * len(sequences['train']),
                  validation=[int(y[0]) for y in data.validation_labels],
                  test=[int(y[0]) for y in data.test_labels])
    assert all(len(x) == 128 and x.shape[1] == 7 for collection in sequences.values() for x in collection)
    if brand == 2:
        represented = {car: label for p in cars for car, label in zip(cars[p], labels[p])}
        assert sum(len(values) for values in sequences.values()) == 194245, 'Brand2 snippet cohort differs'
        assert sum(value == 0 for value in represented.values()) == 33
        assert sum(value == 1 for value in represented.values()) == 16
        # Official CSV lists one additional normal ID (230) with no snippets.
        # Split only the actually represented vehicles, as in existing runs.
        assert '230' not in represented
    if smoke:
        # Preserve the real train-only scaler and disjoint real vehicles. This
        # subset is explicitly marked smoke and can never be a formal result.
        for partition in sequences:
            if partition == 'test':
                selected = [i for label in (0, 1) for i in np.flatnonzero(np.array(labels[partition]) == label)[:8]]
            else:
                selected = list(range(min(128 if partition == 'train' else 16, len(sequences[partition]))))
            sequences[partition] = [sequences[partition][i] for i in selected]
            ids[partition] = [ids[partition][i] for i in selected]
            cars[partition] = [cars[partition][i] for i in selected]
            labels[partition] = [labels[partition][i] for i in selected]
    mapping = dict(brand=brand, fold=fold, model_seed=3407, fold_seed=0,
                   feature_columns=data.metadata['feature_columns'], control_indices=[1, 2],
                   sample_ids=ids, cars=cars, labels=labels,
                   vehicles={p: sorted(set(values)) for p, values in cars.items()},
                   validation_cars=cars['validation'], test_cars=cars['test'],
                   validation_labels=labels['validation'], test_labels=labels['test'],
                   windows={p: len(values) for p, values in sequences.items()}, smoke=smoke,
                   normalization=data.metadata['normalization'])
    verify_mapping(mapping)
    data.train_sequences = sequences['train']; data.validation_sequences = sequences['validation']
    data.test_sequences = sequences['test']
    data.validation_labels = [np.array([v]) for v in labels['validation']]
    data.test_labels = [np.array([v]) for v in labels['test']]
    data.validation_entity_ids = cars['validation']; data.entity_ids = cars['test']
    print('DATA_READY', json.dumps(dict(snippets=mapping['windows'],
          vehicles={p: len(v) for p, v in mapping['vehicles'].items()}, smoke=smoke)), flush=True)
    return data, sequences, mapping


def vehicle_aggregate(snippets):
    rows = []
    for car, group in snippets.groupby('car', sort=True):
        assert group.label.nunique() == 1
        values = group.score.to_numpy(dtype=np.float64)
        k = max(1, int(np.ceil(.05 * len(values))))
        rows.append(dict(car=str(car), label=int(group.label.iloc[0]), snippets=len(values),
                         score=float(np.partition(values, -k)[-k:].mean())))
    return pd.DataFrame(rows)


def threshold_metrics(validation, test):
    assert validation.label.eq(0).all() and test.label.nunique() == 2
    y = test.label.to_numpy(dtype=int); scores = test.score.to_numpy(dtype=np.float64)
    rank = dict(auroc=float(roc_auc_score(y, scores)), ap=float(average_precision_score(y, scores)))
    rows = []
    for q in (.95, .99):
        threshold = float(validation.score.quantile(q)); pred = scores >= threshold
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        rows.append(dict(q=q, threshold=threshold, n=len(y), tp=tp, fp=fp, fn=fn, tn=tn,
                         precision=precision, recall=recall, fpr=fp / max(fp + tn, 1),
                         f1=2 * tp / max(2 * tp + fp + fn, 1), **rank))
    return rows


def evaluate(output, mapping, scores, window, method, source):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'mapping.json', mapping)
    frames = {}
    for partition in ('validation', 'test'):
        values = np.asarray(scores[partition], dtype=np.float64)
        ids = mapping['sample_ids'][partition]
        assert values.shape == (len(ids),) and np.isfinite(values).all()
        # Brand snippets are exactly 128 points; all native protocols here
        # produce one window per snippet. Store this mapping explicitly.
        np.savez_compressed(output / f'{partition}_windows.npz', score=values,
                            sequence_ids=np.arange(len(ids), dtype=np.int64),
                            end_indices=np.full(len(ids), window-1, dtype=np.int64),
                            sample_ids=np.asarray(ids))
        snippets = pd.DataFrame(dict(sample_id=ids, car=mapping['cars'][partition],
                                     label=mapping['labels'][partition], score=values))
        vehicles = vehicle_aggregate(snippets)
        snippets.to_csv(output / f'{partition}_snippet_scores.csv', index=False)
        vehicles.to_csv(output / f'{partition}_vehicle_scores.csv', index=False)
        frames[partition] = vehicles
    rows = threshold_metrics(frames['validation'], frames['test'])
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
    write_json(output / 'metrics.json', dict(method=method, rows=rows,
        brand=mapping['brand'], fold=mapping['fold'], smoke=mapping['smoke'],
        threshold_source='normal_validation_vehicle_only', source=source))
    write_json(output / 'complete.json', dict(method=method, fold=mapping['fold'], brand=mapping['brand'],
        model_seed=3407, fold_seed=0, smoke=mapping['smoke'],
        ranking_level='vehicle', normal_only_training=True, no_test_selection=True,
        mapping_sha256=file_hash(output / 'mapping.json'), source=source))
    print('METRICS', method, json.dumps(rows, ensure_ascii=False), flush=True)


def baseline(job, output, data, mapping, smoke):
    method = job['method']; output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').is_file():
        assert json.loads((output / 'complete.json').read_text())['smoke'] == smoke
        print('SKIP_COMPLETE', job['unit'], flush=True); return
    # Exactly the native window/stride conventions in the current Brand3 RCR
    # reference table, including the 10-point TranAD and 128-point patch models.
    window, stride = (128, 128) if method in {'patchad', 'dcdetector', 'isolation_forest', 'deep_svdd'} else (127, 127)
    saved = sys.argv
    try:
        sys.argv = ['brand_reference', '--method', method, '--dataset', f'Brand{mapping["brand"]}',
                    '--output_dir', str(output), '--lookback', str(window), '--window_stride', str(stride),
                    '--epochs', '1' if smoke else '10', '--batch_size', '64',
                    '--train_sample_limit', '128' if smoke else '100000', '--evaluation_sample_limit', '0',
                    '--svdd_pretrain_epochs', '1' if smoke else '5',
                    '--brand_split_protocol', 'strict_normal_validation', '--brand_normalization', 'minmax',
                    '--brand_fold', str(mapping['fold']), '--brand_fold_seed', '0',
                    '--brand_top_ratio_mode', 'fixed', '--vehicle_top_ratio', '.05']
        args = reference.parse_args()
    finally:
        sys.argv = saved
    window, stride = reference._window_and_stride(args)
    train, validation, test = reference._loaders(data, args, window, stride)
    assert len(validation.dataset) == len(data.validation_sequences) and len(test.dataset) == len(data.test_sequences)
    write_json(output / 'config.json', dict(**vars(args), window=window, resolved_stride=stride,
        normal_only_training=True, smoke=smoke,
        comparison_budget='reference_10epoch_100k_not_equal_to_RCR',
        epoch_selection='fixed_final_no_test_selection'))
    started = time.monotonic()
    if method == 'isolation_forest':
        model = reference._fit_isolation(train, args); auxiliary = None
        dump(model, output / 'model.joblib')
    else:
        model, auxiliary = reference._fit_torch(data, args, train, window, torch.device('cuda'))
        atomic_checkpoint(output / 'model.pt', model.state_dict())
        if method == 'deep_svdd':
            atomic_checkpoint(output / 'center.pt', auxiliary.detach().cpu())
        else:
            assert auxiliary is None
    training_seconds = time.monotonic() - started
    scores = {}; started = time.monotonic()
    for partition, loader in [('validation', validation), ('test', test)]:
        result, endpoints = reference._score_loader(method, model, auxiliary, loader,
            torch.device('cuda'), len(mapping['sample_ids'][partition]))
        assert all(len(x) == 1 for x in result)
        assert all(e.tolist() == [window-1] for e in endpoints)
        scores[partition] = np.concatenate(result)
    write_json(output / 'runtime.json', dict(training_seconds=training_seconds,
        inference_seconds=time.monotonic()-started, gpu=torch.cuda.get_device_name(0)))
    evaluate(output, mapping, scores, window, method, dict(model=str(output / ('model.joblib' if method == 'isolation_forest' else 'model.pt'))))


def run_module(module, *args):
    subprocess.run([sys.executable, '-u', '-m', module, *map(str, args)], check=True)


def ablation_scores(model, datasets):
    scores = {}
    model.eval()
    for partition in ('validation', 'test'):
        result = []
        with torch.inference_mode():
            for values, _, _ in DataLoader(datasets[partition], batch_size=128, shuffle=False):
                values = values.cuda()
                # Match cached float32 residual followed by float64 Lite arithmetic.
                error = (values-model(values)).cpu().numpy().astype(np.float64)
                result.append(np.square(error).mean((1, 2)))
        scores[partition] = np.concatenate(result)
    return scores


def rcr_fold(job, output, sequences, mapping, smoke):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').is_file():
        assert json.loads((output / 'complete.json').read_text())['smoke'] == smoke
        print('SKIP_COMPLETE', job['unit'], flush=True); return
    datasets = {p: reference.WindowDataset(v, 127, 127) for p, v in sequences.items()}
    shared_mapping = dict(mapping, window=127, stride=127)
    for backbone, no_graph, no_temporal in [('full', False, False), ('without_feature_graph', True, False), ('without_temporal', False, True)]:
        seed_all(3407)
        root = output / backbone; root.mkdir(exist_ok=True)
        protocol = dict(dataset=f'Brand{mapping["brand"]}', brand=mapping['brand'], fold=mapping['fold'],
            seed=3407, fold_seed=0, split='strict_normal_validation', normalization=mapping['normalization'],
            window=127, stride=127, feature_graph=not no_graph, temporal=not no_temporal,
            epochs=1 if smoke else 50, patience=8, lr=.001, weight_decay=.0001,
            batch_size=128, smoke=smoke, epoch_selection='normal_validation_only', no_test_selection=True)
        if (root / 'protocol.json').is_file():
            assert json.loads((root / 'protocol.json').read_text()) == protocol
        write_json(root / 'protocol.json', protocol); write_json(root / 'mapping.json', shared_mapping)
        args = SimpleNamespace(seed=3407, no_feature_graph=no_graph, no_temporal=no_temporal,
                               epochs_backbone=protocol['epochs'])
        with stage(f'fold{mapping["fold"]} backbone {backbone}'):
            model = backbone_train(args, root, datasets, 7)
        if backbone != 'full':
            arm = 'raw_no_feature_graph' if no_graph else 'raw_no_temporal'
            evaluate(output / 'analysis' / arm, mapping, ablation_scores(model, datasets), 127, arm,
                     dict(backbone=str(root / 'backbone_best.pt'), score='raw_reconstruction_mse'))
            del model; gc.collect(); torch.cuda.empty_cache(); continue
        cache_residuals(model, datasets, mapping['sample_ids'], (1, 2), root)
        del model; gc.collect(); torch.cuda.empty_cache()
        for condition in ('continuous', 'none'):
            with stage(f'fold{mapping["fold"]} calibration {condition}'):
                run_module('src.runners.run_rcr_lite', '--root', root, '--condition', condition)
            with stage(f'fold{mapping["fold"]} relation {condition}'):
                run_module('src.runners.run_rcr_relation', '--root', root, '--condition', condition,
                           '--modulation', 'mlp' if condition == 'continuous' else 'none',
                           '--epochs', '1' if smoke else '30', '--seed', '3407')
        for arm, unit, column in [('raw', 'lite_continuous', 0), ('scale', 'lite_continuous', 1),
                                  ('lite', 'lite_continuous', 5), ('net', 'net_continuous_mlp', 5),
                                  ('net_no_condition', 'net_none_none', 5)]:
            scores = {p: np.load(root / unit / p / 'scores.npy', mmap_mode='r')[:, column]
                      for p in ('validation', 'test')}
            evaluate(output / 'analysis' / arm, mapping, scores, 127, arm,
                     dict(backbone=str(root / 'backbone_best.pt'), unit=unit, score_column=column))
    assert all((output / 'analysis' / arm / 'complete.json').is_file() for arm in RCR_ARMS)
    write_json(output / 'complete.json', dict(smoke=smoke, fold=mapping['fold'], brand=mapping['brand'],
               arms=list(RCR_ARMS), shared_core_backbone=True, no_fault_training=True))


def stream(command, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    print('$', ' '.join(command), flush=True)
    with path.open('w') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            process.stdout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--shard', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-index', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text()); validate_plan(plan)
    if args.validate_only:
        print('PLAN_VALID 45 jobs / 75 evaluated cells / five folds / four disjoint shards'); return
    shard = plan['shards'][args.shard]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.job_index is not None:
        assert torch.cuda.is_available(), 'CUDA required; run outside sandbox'
        torch.set_num_threads(4); seed_all(3407)
        job = shard['jobs'][args.job_index]
        data, sequences, mapping = load_fold(plan['brand'], job['fold'], args.smoke)
        with stage(job['unit']):
            if job['kind'] == 'reference':
                baseline(job, args.output / job['unit'], data, mapping, args.smoke)
            else:
                rcr_fold(job, args.output / job['unit'], sequences, mapping, args.smoke)
        return
    state_path = args.output / 'protocol_manifest.json'
    actual_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    state = dict(status='running', account=shard['account'], dataset=shard['dataset_source'],
        brand=plan['brand'], shard=args.shard, model_seed=3407, fold_seed=0, git_commit=actual_commit,
        plan_sha256=file_hash(args.plan), split=plan['split'], normalization=plan['normalization'],
        smoke=args.smoke, expected_jobs=len(shard['jobs']), jobs=shard['jobs'], completed=[], failed=[],
        expected_cells=sum(7 if j['kind']=='rcr' else 1 for j in shard['jobs']),
        threshold_quantiles=[.95,.99], score_aggregation='one_window_per_snippet_then_vehicle_top5pct',
        started_at=time.time(), current_unit=None)
    write_json(state_path, state)
    try:
        for index, job in enumerate(shard['jobs']):
            state.update(current_unit=job['unit'], current_index=index+1); write_json(state_path, state)
            print(f'START [{index+1}/{len(shard["jobs"])}] {job["unit"]}', flush=True)
            command = [sys.executable, '-u', '-m', 'src.runners.run_brand_rcr_matrix',
                       '--plan', str(args.plan.resolve()), '--shard', args.shard,
                       '--output', str(args.output.resolve()), '--job-index', str(index)]
            if args.smoke: command.append('--smoke')
            log = args.output / 'logs' / (job['unit'].replace('/', '_') + '.log')
            started = time.time(); code = stream(command, log)
            record = dict(unit=job['unit'], returncode=code, seconds=time.time()-started, log=str(log))
            if code or not (args.output / job['unit'] / 'complete.json').is_file():
                state['failed'].append(record); raise RuntimeError(f'Failed {job["unit"]}; log={log}')
            state['completed'].append(record); write_json(state_path, state)
            print(f'PROGRESS {len(state["completed"])}/{len(shard["jobs"])} failed={len(state["failed"])}', flush=True)
        state.update(status='complete', current_unit=None, ended_at=time.time())
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), ended_at=time.time()); raise
    finally:
        write_json(state_path, state)


if __name__ == '__main__':
    main()
