"""Existing baseline implementations on the exact RCR mixed-cycle split.

Independent adapter, without changing formal model/loader logic. Uses the old
10-epoch/100k-window baseline budget; explicitly NOT equal-budget RCR evidence.
"""
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.data.ch_battery_utils import load_ch_battery_mixed_cycle_research_split
from src.data.external_protocol import ExternalProtocolData
from src.runners import run_unified_external_baselines as reference
from src.runners.run_rcr import atomic_json, atomic_checkpoint
from scripts.evaluate_rcr import metrics

METHODS = ['mtad_gat', 'tranad', 'anomaly_transformer', 'dcdetector', 'patchad', 'omnianomaly']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chemistry', choices=['LFP', 'NCM'], required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rcr-root', required=True, help='Completed/started full-backbone root with authoritative mapping')
    cli = parser.parse_args()
    if not torch.cuda.is_available():
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
                    '--evaluation_sample_limit', '0', '--require_cuda']
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
    data = ExternalProtocolData(dataset=args.dataset,
        train_sequences=[split['train'][k] for k in ids['train']],
        validation_sequences=[split['validation'][k] for k in ids['validation']],
        test_sequences=[split['test'][k] for k in ids['test']],
        validation_labels=[np.array([0]) for _ in ids['validation']],
        test_labels=[np.array([split['test_metadata'][k]['sample_label']]) for k in ids['test']],
        validation_entity_ids=ids['validation'], entity_ids=ids['test'],
        evaluation_kind='sample_ranking', metadata=dict(feature_columns=split['feature_columns']))
    atomic_json(root/'config.json', dict(**vars(args), window=window, resolved_stride=stride,
                comparison_budget='legacy_10epoch_100k_not_equal_to_RCR',
                epoch_selection='fixed_final_no_test_selection', source=reference.SOURCE[cli.method],
                normal_only_training=True, split_mapping=str(Path(cli.rcr_root)/'mapping.json')))
    train, validation, test = reference._loaders(data, args, window, stride)
    model, auxiliary = reference._fit_torch(data, args, train, window, torch.device('cuda'))
    atomic_checkpoint(root/'model.pt', model.state_dict())
    assert auxiliary is None, 'Unexpected nonserialized baseline auxiliary state'
    frames = {}
    for partition, loader in [('validation', validation), ('test', test)]:
        scores, endpoints = reference._score_loader(cli.method, model, auxiliary, loader,
                                                    torch.device('cuda'), len(ids[partition]))
        assert len(scores) == len(ids[partition]) and all(len(s) for s in scores)
        np.savez_compressed(root/f'{partition}_windows.npz', score=np.concatenate(scores),
            sequence_ids=np.concatenate([np.full(len(s), i, dtype=np.int64) for i, s in enumerate(scores)]),
            end_indices=np.concatenate(endpoints), sample_ids=np.asarray(ids[partition]))
        frames[partition] = pd.DataFrame(dict(sample_id=ids[partition],
            score=[reference._topk_sample_score(s, .05) for s in scores],
            label=[split[partition+'_metadata'][k]['sample_label'] for k in ids[partition]]))
        frames[partition].to_csv(root/f'{partition}_samples.csv', index=False)
    rows = []
    for q in [.95, .99]:
        threshold = float(frames['validation'].score.quantile(q))
        rows.append(dict(method=cli.method, chemistry=cli.chemistry, q=q, threshold=threshold,
                         **metrics(frames['test'].label, frames['test'].score, threshold)))
    pd.DataFrame(rows).to_csv(root/'metrics.csv', index=False)
    atomic_json(root/'complete.json', dict(method=cli.method, chemistry=cli.chemistry,
                normal_validation_thresholds=[.95, .99], no_test_selection=True,
                equal_training_budget_to_rcr=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
