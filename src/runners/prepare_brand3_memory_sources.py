"""Rebuild one shared frozen RCR backbone per Brand3 fold for memory probes.

Old audited outputs lack residuals/checkpoints. This is a separately identified
local rerun, not a recovery or reproduction of the remote backbone weights.
"""
import argparse
import fcntl
import gc
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.data.external_protocol import load_external_protocol_data
from src.runners.run_rcr import atomic_json, backbone_train, cache_residuals
from src.runners.run_rcr_calibration_sensitivity import fit_mean_and_rms, make_calibration
from src.runners.run_rcr_memory_probe import digest
from src.runners.run_unified_external_baselines import WindowDataset


def audited_source(fold):
    shard, version = ('brand024', 3) if fold in (0, 2, 4) else ('brand13', 2)
    base = Path(f'runs/kaggle_downloads/rcr_next_{shard}_version{version}')
    inventory = base/'output_inventory.json'
    audit = json.loads(inventory.read_text())
    if audit['status'] != 'complete' or not audit['paged_to_end'] or not audit['scriptVersionId']:
        raise ValueError('Brand3 historical version has not passed inventory audit')
    source = base/audit['artifact_prefix']/f'sources/Brand3/fold{fold}/full'
    return source, dict(kernel=audit['kernel'], version=version,
        scriptVersionId=audit['scriptVersionId'], inventory_sha256=digest(inventory),
        historical_source=str(source.resolve()), historical_checkpoint_reused=False)


def check_fold(data, historical_mapping, historical_protocol):
    ids = data.metadata['sample_ids']
    cars = dict(train=[v.split(':', 1)[0] for v in ids['train']],
                validation=list(map(str, data.validation_entity_ids)), test=list(map(str, data.entity_ids)))
    labels = dict(train=[0]*len(ids['train']),
                  validation=[int(v[0]) for v in data.validation_labels],
                  test=[int(v[0]) for v in data.test_labels])
    sets = {p: set(cars[p]) for p in cars}
    if sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test']:
        raise ValueError('Vehicle leakage')
    if set(labels['train']) != {0} or set(labels['validation']) != {0} or set(labels['test']) != {0, 1}:
        raise ValueError('Invalid normal-only partitions')
    for p in ('train', 'validation', 'test'):
        if len(ids[p]) != historical_mapping['windows'][p] or len(set(ids[p])) != len(ids[p]):
            raise ValueError('Snippet cohort differs from audited Brand3 protocol')
        if p != 'train':
            if cars[p] != list(map(str, historical_mapping[p+'_cars'])) or labels[p] != historical_mapping[p+'_labels']:
                raise ValueError('Vehicle order or labels differ from audited Brand3 protocol')
    scaler = data.metadata['normalization']
    old_scaler = historical_protocol['normalization']
    if scaler['kind'] != old_scaler['kind']:
        raise ValueError('Normalization family differs')
    for name in ('data_min', 'data_max'):
        np.testing.assert_allclose(scaler[name], old_scaler[name], rtol=0, atol=1e-7)
    return cars, labels


def prepare_fold(base, fold):
    historical, provenance = audited_source(fold)
    old_protocol = json.loads((historical/'protocol.json').read_text())
    old_mapping = json.loads((historical/'mapping.json').read_text())
    root = base/'sources'/f'fold{fold}'/'full'
    root.mkdir(parents=True, exist_ok=True)
    marker = root/'memory_source_complete.json'
    if marker.exists():
        print('BRAND3 MEMORY SOURCE ALREADY COMPLETE', fold, flush=True)
        return
    print('BRAND3 MEMORY SOURCE LOAD', fold, flush=True)
    data = load_external_protocol_data('Brand3', brand_fold=fold, brand_fold_seed=0, seed=3407,
        brand_split_protocol='strict_normal_validation', brand_normalization='minmax', brand_progress=True)
    cars, labels = check_fold(data, old_mapping, old_protocol)
    sequences = dict(train=data.train_sequences, validation=data.validation_sequences, test=data.test_sequences)
    datasets = {p: WindowDataset(values, 127, 127, seed=3407) for p, values in sequences.items()}
    ids = data.metadata['sample_ids']
    if any(len(datasets[p]) != len(ids[p]) for p in datasets):
        raise ValueError('Expected exactly one 127-point window per Brand3 snippet')
    mapping = dict(dataset='Brand3', fold=fold, sample_ids=ids, cars=cars, labels=labels,
        vehicles={p: sorted(set(v)) for p, v in cars.items()},
        feature_columns=data.metadata['feature_columns'], control_indices=[1, 2],
        windows={p: len(v) for p, v in datasets.items()}, normalization=data.metadata['normalization'])
    protocol = dict(dataset='Brand3', fold=fold, seed=3407, fold_seed=0,
        split='strict_normal_validation', normalization=data.metadata['normalization'],
        window=127, stride=127, feature_graph=True, temporal=True, epochs=50, patience=8,
        lr=.001, weight_decay=.0001, batch_size=128, smoke=False,
        epoch_selection='normal_validation_only', no_test_selection=True,
        backbone_provenance='new_local_backbone_shared_frozen_by_all_A_B_arms',
        historical_protocol_audit=provenance)
    for filename, value in [('protocol.json', protocol), ('mapping.json', mapping)]:
        path = root/filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Output protocol/data changed; use a new directory')
        atomic_json(path, value)
    rows = [dict(sample_id=sample, partition=p, car=car, sample_label=label)
            for p in datasets for sample, car, label in zip(ids[p], cars[p], labels[p])]
    pd.DataFrame(rows).to_csv(root/'metadata.csv', index=False)
    print('BRAND3 MEMORY SOURCE READY', json.dumps(dict(fold=fold, windows=mapping['windows'],
        vehicles={p: len(v) for p, v in mapping['vehicles'].items()})), flush=True)
    torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    np.random.seed(3407); random.seed(3407)
    torch.backends.cudnn.benchmark = False
    args = SimpleNamespace(seed=3407, no_feature_graph=False, no_temporal=False, epochs_backbone=50)
    model = backbone_train(args, root, datasets, 7)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache_residuals(model.eval(), datasets, ids, (1, 2), root)
    del model, data, sequences, datasets
    gc.collect(); torch.cuda.empty_cache()
    residual = np.load(root/'residuals/train/residual.npy', mmap_mode='r')
    q = np.load(root/'residuals/train/condition.npy', mmap_mode='r')
    qm, qs, coef, rms = fit_mean_and_rms(q, residual)
    calibration, _ = make_calibration(q, qm, qs, coef, rms, .05)
    target = root/'lite_continuous'
    target.mkdir(exist_ok=True)
    calibration.save(target/'calibration.npz')
    atomic_json(marker, dict(fold=fold, normal_training_only=True, full_data=True,
        frozen_backbone=json.loads((root/'backbone_complete.json').read_text()),
        historical_protocol_verified=True, historical_checkpoint_reused=False,
        windows=mapping['windows'], calibration_min_std=.05))
    print('BRAND3 MEMORY SOURCE COMPLETE', fold, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/internal/rcr_memory_brand3_20261025_v1')
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or not all(f in range(5) for f in args.folds):
        parser.error('Specify unique folds in 0..4')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; execute outside sandbox')
    torch.set_num_threads(4)
    base = Path(args.output).resolve()
    base.mkdir(parents=True, exist_ok=True)
    protocol = dict(task='RCR-MEM-BRAND3-20261025-v1', dataset='Brand3', folds=args.folds,
        backbone_seed=3407, fold_seed=0, window=127, stride=127,
        split='strict_normal_validation', normalization='train_fold_minmax',
        max_epochs=50, patience=8, gpu=torch.cuda.get_device_name(0),
        backbone_source='new_local_fivefold; historical remote weights unavailable',
        ks=[1, 2, 4, 6, 8], primary_score='magnitude', threshold_quantiles=[.95, .99],
        primary_threshold=.95, aggregate='window_top5pct_then_snippet_top5pct_per_vehicle',
        normal_fit_only=True, no_test_tuning=True, phase_c_started=False,
        preparation_implementation_sha256=digest(__file__))
    manifest = base/'protocol_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Preparation protocol changed; use a new directory')
    atomic_json(manifest, protocol)
    with open('/tmp/enhancedmtadgat_brand3_memory_gpu0.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for fold in args.folds:
            prepare_fold(base, fold)
    atomic_json(base/'sources_complete.json', dict(folds=args.folds, all_sources_complete=True))
    print('BRAND3 MEMORY ALL SOURCES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
