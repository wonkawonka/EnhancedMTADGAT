"""Existing baseline implementations on the exact RCR mixed-cycle split.

Independent adapter, without changing formal model/loader logic. Uses the old
10-epoch/100k-window baseline budget; explicitly NOT equal-budget RCR evidence.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pywt
import torch
from joblib import dump as joblib_dump
from sklearn.preprocessing import MinMaxScaler
from src.data.ch_battery_utils import load_ch_battery_mixed_cycle_research_split
from src.data.external_protocol import ExternalProtocolData
from src.runners import run_unified_external_baselines as reference
from src.runners.run_rcr import atomic_json, atomic_checkpoint
from scripts.evaluate_rcr import metrics

METHODS = ['mtad_gat', 'tranad', 'anomaly_transformer', 'dcdetector', 'patchad', 'omnianomaly', 'isolation_forest']
COMPONENTS = ['raw', 'A3_trend', 'D3_low', 'D2_mid', 'D1_high']


def _minimum_wavelet_length(wavelet='db4', level=3):
    length = 1
    filter_length = pywt.Wavelet(wavelet).dec_len
    while pywt.dwt_max_level(length, filter_length) < level:
        length += 1
    return length


def _wavelet_component(sequence, component, wavelet='db4', level=3):
    if component == 'raw':
        return np.asarray(sequence, dtype=np.float32)
    keep_by_name = {'A3_trend': 0, 'D3_low': 1, 'D2_mid': 2, 'D1_high': 3}
    keep = keep_by_name[component]
    minimum = _minimum_wavelet_length(wavelet, level)
    reconstructed = np.empty_like(sequence, dtype=np.float32)
    for feature in range(sequence.shape[1]):
        values = np.asarray(sequence[:, feature], dtype=np.float64)
        padding = max(0, minimum - len(values))
        ready = np.pad(values, (0, padding), mode='symmetric' if len(values) > 1 else 'edge') if padding else values
        coefficients = pywt.wavedec(ready, wavelet, mode='symmetric', level=level)
        selected = [np.zeros_like(item) for item in coefficients]
        selected[keep] = coefficients[keep]
        reconstructed[:, feature] = pywt.waverec(selected, wavelet, mode='symmetric')[:len(values)]
    return reconstructed


def apply_component(split, component, wavelet='db4', level=3):
    """Reconstruct one component and fit its scaler on normal train samples only."""
    if component == 'raw':
        return {'component': component, 'wavelet': None, 'level': None,
                'post_component_scaler': 'existing_training_normal_raw_minmax'}
    transformed = {}
    for partition in ['train', 'validation', 'test']:
        transformed[partition] = {
            key: _wavelet_component(sequence, component, wavelet, level)
            for key, sequence in split[partition].items()
        }
    scaler = MinMaxScaler().fit(np.concatenate(list(transformed['train'].values()), axis=0))
    for partition in transformed:
        split[partition] = {
            key: scaler.transform(sequence).astype(np.float32, copy=False)
            for key, sequence in transformed[partition].items()
        }
    return {
        'component': component, 'wavelet': wavelet, 'level': level,
        'post_component_scaler': 'training_normal_component_minmax',
        'scaler_min': scaler.data_min_.tolist(), 'scaler_max': scaler.data_max_.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chemistry', choices=['LFP', 'NCM'], required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--component', choices=COMPONENTS, default='raw')
    parser.add_argument('--output', required=True)
    parser.add_argument('--rcr-root', required=True, help='Completed/started full-backbone root with authoritative mapping')
    cli = parser.parse_args()
    if cli.method != 'isolation_forest' and not torch.cuda.is_available():
        raise RuntimeError('Run CUDA outside sandbox')
    root = Path(cli.output); root.mkdir(parents=True, exist_ok=True)
    if (root/'complete.json').exists():
        print('REFERENCE ALREADY COMPLETE', root, flush=True)
        return
    # Obtain all current baseline defaults without editing its parser or model code.
    saved_argv = sys.argv
    try:
        sys.argv = ['reference', '--method', cli.method, '--dataset', 'CH_LFP_MIXED_CYCLE',
                    '--output_dir', str(root), '--lookback', '32', '--window_stride', '4',
                    '--epochs', '10', '--batch_size', '64', '--train_sample_limit', '100000',
                    '--evaluation_sample_limit', '0',
                    *([] if cli.method == 'isolation_forest' else ['--require_cuda'])]
        args = reference.parse_args()
    finally:
        sys.argv = saved_argv
    args.dataset = f'CH_{cli.chemistry}_MIXED_CYCLE'
    window, stride = reference._window_and_stride(args)
    torch.set_num_threads(4)
    torch.manual_seed(3407); torch.cuda.manual_seed_all(3407); np.random.seed(3407); random.seed(3407)
    split = load_ch_battery_mixed_cycle_research_split(chemistry=cli.chemistry, seed=3407)
    ids = {p: sorted(k for k, x in split[p].items() if len(x) >= 32) for p in ['train', 'validation', 'test']}
    mapping = json.loads((Path(cli.rcr_root)/'mapping.json').read_text())
    assert ids == mapping['sample_ids']
    assert split['feature_columns'] == mapping['feature_columns']
    for key in ['train_vins', 'validation_vins', 'test_normal_vins']:
        assert split[key] == mapping[key]
    for p in ['train', 'validation']:
        assert all(split[p+'_metadata'][k]['sample_label'] == 0 for k in ids[p])
    component_protocol = apply_component(split, cli.component)
    data = ExternalProtocolData(dataset=args.dataset,
        train_sequences=[split['train'][k] for k in ids['train']],
        validation_sequences=[split['validation'][k] for k in ids['validation']],
        test_sequences=[split['test'][k] for k in ids['test']],
        validation_labels=[np.array([0]) for _ in ids['validation']],
        test_labels=[np.array([split['test_metadata'][k]['sample_label']]) for k in ids['test']],
        validation_entity_ids=ids['validation'], entity_ids=ids['test'],
        evaluation_kind='sample_ranking', metadata=dict(feature_columns=split['feature_columns']))
    atomic_json(root/'config.json', dict(**vars(args), component_protocol=component_protocol,
                window=window, resolved_stride=stride,
                comparison_budget='legacy_10epoch_100k_not_equal_to_RCR',
                epoch_selection='fixed_final_no_test_selection', source=reference.SOURCE[cli.method],
                normal_only_training=True, split_mapping=str(Path(cli.rcr_root)/'mapping.json')))
    train, validation, test = reference._loaders(data, args, window, stride)
    started = time.perf_counter()
    if cli.method == 'isolation_forest':
        model = reference._fit_isolation_points(data, args)
        auxiliary = None
        joblib_dump(model, root/'model.joblib')
    else:
        model, auxiliary = reference._fit_torch(data, args, train, window, torch.device('cuda'))
        atomic_checkpoint(root/'model.pt', model.state_dict())
        assert auxiliary is None, 'Unexpected nonserialized baseline auxiliary state'
    training_seconds = time.perf_counter() - started
    frames = {}
    for partition, loader in [('validation', validation), ('test', test)]:
        if cli.method == 'isolation_forest':
            scores = reference._score_isolation_points(model, data.validation_sequences if partition == 'validation' else data.test_sequences)
            endpoints = [np.arange(len(score), dtype=np.int64) for score in scores]
        else:
            scores, endpoints = reference._score_loader(cli.method, model, auxiliary, loader,
                                                        torch.device('cuda'), len(ids[partition]))
        assert len(scores) == len(ids[partition]) and all(len(s) for s in scores)
        np.savez_compressed(root/f'{partition}_windows.npz', score=np.concatenate(scores),
            sequence_ids=np.concatenate([np.full(len(s), i, dtype=np.int64) for i, s in enumerate(scores)]),
            end_indices=np.concatenate(endpoints), sample_ids=np.asarray(ids[partition]))
        frames[partition] = pd.DataFrame(dict(sample_id=ids[partition],
            score=[reference._topk_sample_score(s, .05)[0] for s in scores],
            label=[split[partition+'_metadata'][k]['sample_label'] for k in ids[partition]]))
        frames[partition].to_csv(root/f'{partition}_samples.csv', index=False)
    rows = []
    for q in [.95, .99]:
        threshold = float(frames['validation'].score.quantile(q))
        rows.append(dict(method=cli.method, chemistry=cli.chemistry, component=cli.component,
                         q=q, threshold=threshold,
                         **metrics(frames['test'].label, frames['test'].score, threshold)))
    pd.DataFrame(rows).to_csv(root/'metrics.csv', index=False)
    atomic_json(root/'complete.json', dict(method=cli.method, chemistry=cli.chemistry,
                component=cli.component, training_seconds=training_seconds,
                normal_validation_thresholds=[.95, .99], no_test_selection=True,
                equal_training_budget_to_rcr=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
