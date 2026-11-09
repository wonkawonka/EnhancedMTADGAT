"""Config-driven Brand1 external comparisons, sharing the RCR vehicle folds."""
import argparse
import gc
import fcntl
import hashlib
import json
import os
import random
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from scripts.evaluate_rcr import metrics
from src.data.external_protocol import load_external_protocol_data
from src.runners import run_unified_external_baselines as baseline
from src.runners.run_brand_rcr_matrix import vehicle_aggregate
from src.runners.run_rcr import atomic_checkpoint, atomic_json


METRICS = ['ap', 'auroc', 'precision', 'recall', 'f1', 'fpr']
LABELS = dict(mtad_gat='MTAD-GAT', tranad='TranAD', anomaly_transformer='Anomaly Transformer',
              dcdetector='DCdetector', patchad='PatchAD', omnianomaly='OmniAnomaly',
              isolation_forest='Isolation Forest', deep_svdd='Deep SVDD')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expand(plan):
    assert plan['dataset'] == 'Brand1' and plan['folds'] == list(range(5))
    assert plan['fold_seed'] == 0 and plan['split'] == 'strict_normal_validation'
    assert plan['normalization'] == 'minmax'
    assert plan['threshold_quantile'] == .95 and plan['vehicle_top_ratio'] == .05
    assert plan['train_sample_limit'] == plan['evaluation_sample_limit'] == 0
    assert plan['methods'] == list(LABELS)
    assert set(plan['seeds']).issubset({3407, 3408, 3409})
    jobs = [dict(seed=s, fold=f, method=m, id=f'{m}/seed{s}/fold{f}')
            for s in plan['seeds'] for f in plan['folds'] for m in plan['methods']]
    assert len(jobs) == plan['expected_jobs'] == len({j['id'] for j in jobs})
    return jobs


def build_args(plan, job, output):
    saved = sys.argv
    try:
        sys.argv = ['brand1_comparison', '--dataset', 'Brand1', '--method', job['method'],
                    '--seed', str(job['seed']), '--output_dir', str(output),
                    '--lookback', str(plan['native_windows'][job['method']]),
                    '--window_stride', str(plan['native_strides'][job['method']]),
                    '--epochs', str(plan['epochs']), '--svdd_pretrain_epochs', str(plan['svdd_pretrain_epochs']),
                    '--batch_size', str(plan['batch_size']), '--learning_rate', str(plan['learning_rate']),
                    '--train_sample_limit', '0', '--evaluation_sample_limit', '0',
                    '--brand_fold', str(job['fold']), '--brand_fold_seed', '0',
                    '--brand_normalization', 'minmax', '--brand_split_protocol', 'strict_normal_validation',
                    '--brand_top_ratio_mode', 'fixed', '--vehicle_top_ratio', '.05',
                    '--threshold_quantile', '.95', '--require_cuda']
        return baseline.parse_args()
    finally:
        sys.argv = saved


def load_fold(plan, fold):
    source = Path(plan['predecessor']) / 'preparation/sources' / f'Brand1_fold{fold}'
    mapping = json.loads((source / 'mapping.json').read_text())
    data = load_external_protocol_data('Brand1', brand_fold=fold, brand_fold_seed=0, seed=3407,
                                      brand_split_protocol='strict_normal_validation',
                                      brand_normalization='minmax', brand_progress=True)
    assert data.metadata['sample_ids'] == mapping['sample_ids'], 'Cohort differs from RCR'
    assert data.metadata['normalization'] == mapping['normalization'], 'Scaler differs from RCR'
    assert all(int(y[0]) == 0 for y in data.validation_labels)
    cars = [set(mapping['cars'][p]) for p in ['train', 'validation', 'test']]
    assert not (cars[0] & cars[1] or cars[0] & cars[2] or cars[1] & cars[2])
    return data, mapping, source


def run_job(plan, job, data, mapping, source):
    output = Path(plan['output']) / 'jobs' / job['id']
    output.mkdir(parents=True, exist_ok=True)
    seed = job['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    args = build_args(plan, job, output)
    window, stride = baseline._window_and_stride(args)
    assert window == plan['native_windows'][job['method']]
    train, validation, test = baseline._loaders(data, args, window, stride)
    assert len(train.dataset) == len(data.train_sequences)
    assert len(validation.dataset) == len(data.validation_sequences)
    assert len(test.dataset) == len(data.test_sequences)
    atomic_json(output / 'config.json', dict(args=vars(args), window=window, stride=stride,
                checkpoint_selection=plan['checkpoint_selection'], cohort_reference=str(source / 'mapping.json'),
                cohort_sha256=digest(source / 'mapping.json'), native_implementation=baseline.SOURCE[job['method']]))
    started = time.monotonic()
    if job['method'] == 'isolation_forest':
        model = baseline._fit_isolation(train, args)
        auxiliary = None
        joblib.dump(model, output / 'model.joblib')
    else:
        model, auxiliary = baseline._fit_torch(data, args, train, window, torch.device('cuda'))
        atomic_checkpoint(output / 'model.pt', model.state_dict())
        if auxiliary is not None:
            atomic_checkpoint(output / 'center.pt', auxiliary.detach().cpu())
    training_seconds = time.monotonic() - started
    frames = {}
    for part, loader in [('validation', validation), ('test', test)]:
        scores, ends = baseline._score_loader(job['method'], model, auxiliary, loader,
                        torch.device('cuda'), len(mapping['sample_ids'][part]))
        assert all(len(s) == 1 for s in scores)
        assert all(e.tolist() == [window - 1] for e in ends)
        values = np.concatenate(scores)
        assert np.isfinite(values).all()
        frame = pd.DataFrame(dict(sample_id=mapping['sample_ids'][part],
                                 car=mapping['cars'][part], label=mapping['labels'][part], score=values))
        frame.to_csv(output / f'{part}_snippet_scores.csv', index=False)
        frames[part] = vehicle_aggregate(frame)
        frames[part].to_csv(output / f'{part}_vehicle_scores.csv', index=False)
        np.savez_compressed(output / f'{part}_window_scores.npz', scores=values,
                            sample_ids=np.asarray(mapping['sample_ids'][part]),
                            end_indices=np.full(len(values), window - 1, dtype='int16'))
    assert frames['validation'].label.eq(0).all()
    threshold = float(frames['validation'].score.quantile(.95))
    row = dict(job=job['id'], dataset='Brand1', method=job['method'], seed=seed, fold=job['fold'],
               threshold=threshold, **metrics(frames['test'].label, frames['test'].score, threshold))
    atomic_json(output / 'metrics.json', row)
    atomic_json(output / 'complete.json', dict(job=job, training_seconds=training_seconds,
                total_seconds=time.monotonic() - started, normal_only_training=True,
                no_test_selection=True, ranking_level='vehicle', threshold_source='normal_validation_P95'))
    print('BRAND1_EXTERNAL_RESULT', json.dumps(row), flush=True)
    del model, auxiliary, train, validation, test
    gc.collect(); torch.cuda.empty_cache()


def summarize(plan, jobs):
    root = Path(plan['output'])
    rows = [json.loads((root / 'jobs' / j['id'] / 'metrics.json').read_text()) for j in jobs
            if (root / 'jobs' / j['id'] / 'complete.json').exists()]
    if not rows:
        return
    pd.DataFrame(rows).to_csv(root / 'all_metrics.csv', index=False)
    entries = []
    predecessor = Path(plan['predecessor']) / 'all_metrics.csv'
    with predecessor.open() as stream:
        import csv
        rcr = [r for r in csv.DictReader(stream) if r['name'] == 'R1_full' and r['method'] == 'scale'
               and int(r['seed']) in plan['seeds']]
    for name, selected in [('RCR Scale', rcr)] + [(LABELS[m], [r for r in rows if r['method'] == m]) for m in plan['methods']]:
        valid = len(selected) == 5 * len(plan['seeds'])
        values = {}
        if valid:
            for metric in METRICS:
                per_seed = [statistics.mean(float(r[metric]) for r in selected if int(r['seed']) == s)
                            for s in plan['seeds']]
                values[metric] = (statistics.mean(per_seed), statistics.stdev(per_seed) if len(per_seed)>1 else None)
        entries.append((name, values, len(selected)))
    best = {k: (min if k=='fpr' else max)(round(v[k][0],4) for _,v,_ in entries if v) for k in METRICS}
    lines = ['| 模型 | Brand1 AP ↑ | Brand1 AUROC ↑ | Brand1 P ↑ | Brand1 R ↑ | Brand1 F1 ↑ | Brand1 FPR ↓ | 完成折数 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, values, count in entries:
        cells = []
        for k in METRICS:
            value = '待补'
            if values:
                mean, std = values[k]
                value = f'{mean:.4f}' + (f'±{std:.4f}' if std is not None else '')
                if round(mean,4) == best[k]: value = f'**{value}**'
            cells.append(value)
        lines.append('| '+name+' | '+' | '.join(cells)+f' | {count}/{5*len(plan["seeds"])} |')
    lines += ['', '| 协议项 | 本轮设置 |', '| --- | --- |',
              f'| 种子 | {plan["seeds"]}；先五折均值，再跨种子均值±样本标准差 |',
              '| 数据划分及归一化 | 与RCR相同；正常训练MinMax；正常训练/验证车辆与测试车辆互斥 |',
              '| 评分与阈值 | 各模型原生异常分数；车辆Top5%片段均值；正常验证P95 |',
              '| 训练预算 | 全量正常训练，神经模型30轮；Deep SVDD另有5轮AE预训练；IF无epoch |',
              '| 选点区别 | 外部模型固定最终权重；RCR选正常验证Huber最低点；不声称训练协议完全一致 |',
              f'| 任务 | {plan["plan_name"]}；外部完成{len(rows)}/{len(jobs)}；不改论文最终四表 |', '']
    report = Path(plan['report']); report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='configs/external/brand1_comparison_20261108.json', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text()); jobs = expand(plan)
    if not args.execute:
        print(json.dumps(dict(jobs=len(jobs), seeds=plan['seeds'], methods=plan['methods'],
                              gpu=plan['gpu'], output=plan['output'], epochs=plan['epochs']), ensure_ascii=False))
        return
    root = Path(plan['output']); root.mkdir(parents=True, exist_ok=True)
    def state(**kw):
        completed = sum((root / 'jobs' / j['id'] / 'complete.json').exists() for j in jobs)
        atomic_json(root / 'status.json', dict(updated_at=time.time(), **{
                    'expected_jobs': len(jobs), 'completed_jobs': completed, **kw}))
    state(status='waiting_predecessor', predecessor=plan['predecessor'])
    while not (Path(plan['predecessor']) / 'complete.json').exists():
        time.sleep(10)
    assert torch.cuda.is_available(), 'Run CUDA outside the sandbox'
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    sources = [__file__, baseline.__file__, 'src/data/external_protocol.py', 'src/runners/run_brand_rcr_matrix.py']
    manifest = dict(plan=plan, sources={p:digest(p) for p in sources}, gpu=torch.cuda.get_device_name(0))
    path = root / 'protocol_manifest.json'
    if path.exists(): assert json.loads(path.read_text()) == manifest
    else: atomic_json(path, manifest)
    loaded = None
    try:
        with open('/tmp/enhancedmtadgat_rcr_revalidation.lock', 'a+') as gpu_lock, threadpool_limits(limits=4):
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            for job in jobs:
                if (root / 'jobs' / job['id'] / 'complete.json').exists(): continue
                guard = plan.get('host_disk_guard')
                while guard and shutil.disk_usage(guard['path']).free < guard['min_free_bytes']:
                    state(status='waiting_disk', active_job=job['id'], min_free_bytes=guard['min_free_bytes'])
                    time.sleep(10)
                if loaded != job['fold']:
                    if loaded is not None:
                        del data, mapping, source
                        gc.collect()
                    state(status='loading_fold', active_job=job['id'])
                    data, mapping, source = load_fold(plan, job['fold'])
                    loaded = job['fold']
                state(status='running', active_job=job['id'], expected_jobs=len(jobs))
                run_job(plan, job, data, mapping, source)
                summarize(plan, jobs)
        assert all(digest(p)==sha for p,sha in manifest['sources'].items()), 'Source changed during execution'
        atomic_json(root / 'complete.json', dict(jobs=len(jobs), all_complete=True))
        state(status='complete', completed_jobs=len(jobs))
    except Exception:
        state(status='failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
