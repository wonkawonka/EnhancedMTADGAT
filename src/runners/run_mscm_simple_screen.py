"""Single-seed, full-cohort screening of the simplified A1/A2/A2-T/A3/A3-T/A5.

Run CUDA outside the sandbox. Stages have independently capped epoch budgets.
No historical reconstruction weights or test-selected hyperparameters are used.
"""
import argparse
import fcntl
import gc
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from src.models.mscm_simple import SimpleMSCM
from src.models.mscm_reconstruction import FixedRobustResidualCalibration
from src.runners.full_validation_support import normal_split, ranked_frames, evaluate_frames
from src.runners.run_local_residual_models import source_for
from src.runners.run_mscm_development import InputArrays
from src.runners.run_rcr import atomic_json, atomic_checkpoint, state_hash
from src.runners.run_rcr_memory_probe import digest
from src.runners.report_metric_format import bold_best_metrics


TASK = 'MSCM-SIMPLE-SCREEN-20261025-s3407-T20-M30'
ARMS = ('A1', 'A2', 'A2-T', 'A3', 'A3-T', 'A5')
PARTITIONS = ('train', 'validation', 'test')
UNITS = [f'Brand3_fold{i}' for i in range(5)] + ['NCM', 'LFP']
DEFAULT_ROOT = 'runs/internal/mscm_simple_screen_20261025_s3407_t20_m30'
# Training writes a disposable progress draft.  The audited report is published
# by analyze_mscm_simple_screen only after the complete requested matrix exists.
# Keeping these paths separate prevents a resumed arm from truncating the
# previously reviewed report.
DEFAULT_REPORT = 'report/analysis/rcr/MSCM_SIMPLE_SCREEN_20261025.generated.md'

# Reference results are copied from RCR_REPORT_ZH.md and reordered uniformly as
# AP / AUROC / F1 / Recall / FPR. They stay outside the paired structure deltas
# because their training budgets are not identical to this screening task.
REFERENCE_MODELS = {
    'RCR Scale（主报告）': {
        'LFP': (.8113, .7715, .4656, .3171, .0563),
        'NCM': (.8755, .8189, .6831, .5412, .0543),
        'Brand3': (.8728, .9240, .7236, .6889, .0994),
    },
    'MTAD-GAT †': {
        'LFP': (.7245, .6231, .3638, .2318, .0531),
        'NCM': (.8029, .7400, .5042, .3462, .0338),
        'Brand3': (.8714, .9251, .5899, .6000, .1105),
    },
    'TranAD †': {
        'LFP': (.7814, .7457, .3380, .2117, .0512),
        'NCM': (.7922, .7288, .4657, .3161, .0517),
        'Brand3': (.8509, .8934, .6316, .5778, .1111),
    },
    'Anomaly Transformer †': {
        'LFP': (.8240, .8137, .4223, .2780, .0481),
        'NCM': (.5625, .4699, .0611, .0322, .0285),
        'Brand3': (.5473, .7611, .1267, .0889, .0778),
    },
    'DCdetector †': {
        'LFP': (.5330, .4656, .0554, .0296, .0481),
        'NCM': (.5925, .5607, .1071, .0587, .0457),
        'Brand3': (.2330, .1689, .0000, .0000, .0550),
    },
    'PatchAD †': {
        'LFP': (.6976, .6645, .3379, .2148, .0700),
        'NCM': (.5582, .4814, .0849, .0455, .0311),
        'Brand3': (.6039, .6767, .2291, .1556, .0444),
    },
    'OmniAnomaly †': {
        'LFP': (.6759, .5922, .2761, .1681, .0619),
        'NCM': (.7581, .7023, .3226, .1977, .0351),
        'Brand3': (.4973, .6815, .1379, .0889, .0778),
    },
    'Isolation Forest §': {
        'LFP': (.7157, .6619, .3268, .2047, .0600),
        'NCM': (.5624, .5249, .0667, .0354, .0338),
        'Brand3': (.7628, .8296, .6100, .5556, .1439),
    },
    'Deep SVDD †': {
        'LFP': (.8405, .8058, .5135, .3618, .0587),
        'NCM': None,
        'Brand3': None,
    },
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EncodedArrays:
    """The encoder is frozen, so reuse its exact features for every adaptation."""
    def __init__(self, root, inputs):
        self.inputs = inputs
        self.arrays = {s: np.load(root / f'{s}_h.npy', mmap_mode='r') for s in PARTITIONS}
        self.resident = {}
        needed = sum(self.arrays[s].nbytes for s in ('train', 'validation'))
        free, _ = torch.cuda.mem_get_info()
        if needed < .6 * free:
            for s in ('train', 'validation'):
                self.resident[s] = torch.from_numpy(np.array(self.arrays[s])).cuda()
        print('ENCODED CACHE', root.name, list(self.resident), 'bytes', needed, flush=True)

    def get(self, partition, index):
        if partition in self.resident:
            if isinstance(index, np.ndarray):
                index = torch.as_tensor(index, device='cuda')
            return self.resident[partition][index]
        return torch.from_numpy(np.array(self.arrays[partition][index])).cuda()


def validation_loss(model, inputs, batch, encoded=None):
    model.eval()
    total = torch.zeros((), device='cuda', dtype=torch.float64)
    count = len(inputs.arrays['validation'])
    with torch.inference_mode():
        for lo in range(0, count, batch):
            ix = slice(lo, lo + batch)
            x = inputs.get('validation', ix)
            y = model(x) if encoded is None else model.decode(encoded.get('validation', ix))[0]
            total += F.huber_loss(y, x).double() * len(x)
    return float(total / count)


def train(model, inputs, fit_indices, output, args, stage, epochs, encoded=None):
    output.mkdir(parents=True, exist_ok=True)
    model.configure_stage(stage)
    if (output / 'complete.json').exists():
        model.load_state_dict(torch.load(output / 'best.pt', map_location='cuda', weights_only=True))
        assert state_hash(model) == json.loads((output / 'complete.json').read_text())['model_hash']
        return model.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5,
                                                         threshold=1e-4, min_lr=1e-6)
    history, start, stale, best_epoch = [], 1, 0, 0
    if (output / 'last.pt').exists():
        saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, start = saved['history'], saved['epoch'] + 1
        best, anchor, stale, best_epoch = (saved[k] for k in ('best', 'anchor', 'stale', 'best_epoch'))
        torch.set_rng_state(saved['rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        best = anchor = validation_loss(model, inputs, args.batch, encoded)
        atomic_checkpoint(output / 'best.pt', model.state_dict())
        atomic_json(output / 'initial.json', dict(parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in parameters), normal_validation_huber=best,
                    stage=stage, epoch_budget=epochs, fit_windows=len(fit_indices)))
    stop = 'epoch_cap'
    for epoch in range(start, epochs + 1):
        if stale >= args.patience:
            stop = 'normal_validation_early_stop'
            break
        clock = time.monotonic()
        model.train()
        if stage == 'adapt':
            model.encoder.eval()
        order = np.random.default_rng(args.seed + epoch + (10000 if stage == 'adapt' else 0)).permutation(fit_indices)
        total = torch.zeros((), device='cuda', dtype=torch.float64)
        soft = torch.zeros(8, device='cuda', dtype=torch.float64)
        hard = torch.zeros_like(soft)
        tokens = 0
        for lo in range(0, len(order), args.batch):
            ix = order[lo:lo + args.batch]
            x = inputs.get('train', ix)
            optimizer.zero_grad(set_to_none=True)
            if encoded is None:
                y, weights = model(x), None
            else:
                y, weights = model.decode(encoded.get('train', ix))
            loss = F.huber_loss(y, x)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: {output}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
            total += loss.detach().double() * len(x)
            if weights is not None:
                w = weights.detach()
                soft += w.double().sum((0, 1))
                hard += torch.bincount(w.argmax(-1).flatten(), minlength=8)
                tokens += w.shape[0] * w.shape[1]
        val = validation_loss(model, inputs, args.batch, encoded)
        if not np.isfinite(val):
            raise FloatingPointError(f'Nonfinite normal validation loss: {output}')
        if val < best:
            best, best_epoch = val, epoch
            atomic_checkpoint(output / 'best.pt', model.state_dict())
        if val < anchor - max(1e-12, abs(anchor) * 1e-4):
            anchor, stale = val, 0
        else:
            stale += 1
        scheduler.step(val)
        row = dict(epoch=epoch, train_huber=float(total / len(order)), normal_validation_huber=val,
                   best_epoch=best_epoch, seconds=time.monotonic() - clock,
                   lr=optimizer.param_groups[0]['lr'], stage=stage)
        if tokens:
            row['effective_soft_prototypes'] = float(torch.exp(-((soft / tokens) *
                                               (soft / tokens).clamp_min(1e-30).log()).sum()))
            atomic_json(output / f'memory_epoch_{epoch:03d}.json', dict(
                soft_occupancy=(soft / tokens).cpu().tolist(),
                hard_occupancy=(hard / tokens).cpu().tolist(), normal_fit_only=True))
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        atomic_checkpoint(output / 'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), history=history, epoch=epoch, best=best, anchor=anchor,
            stale=stale, best_epoch=best_epoch, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        atomic_json(output / 'progress.json', row)
        print(json.dumps(dict(event='SIMPLE_EPOCH', path=str(output), **row)), flush=True)
    final = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
    atomic_checkpoint(output / 'final.pt', final['model'])
    model.load_state_dict(torch.load(output / 'best.pt', map_location='cuda', weights_only=True))
    atomic_json(output / 'complete.json', dict(model_hash=state_hash(model), best_epoch=best_epoch,
        epochs_executed=len(history), max_epochs=epochs, normal_validation_huber=best, stop_reason=stop,
        at_budget_boundary=best_epoch == epochs, fit_windows=len(fit_indices), test_labels_used=False,
        encoder_frozen=stage == 'adapt', seconds=sum(r['seconds'] for r in history),
        parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in parameters)))
    return model.eval()


def encode(pretrained, inputs, output, args):
    output.mkdir(parents=True, exist_ok=True)
    encoder_hash = state_hash(pretrained.encoder)
    for partition in PARTITIONS:
        marker = output / f'{partition}_complete.json'
        if marker.exists():
            assert json.loads(marker.read_text())['encoder_hash'] == encoder_hash
            continue
        n, t, _ = inputs.arrays[partition].shape
        values = np.lib.format.open_memmap(output / f'{partition}_h.npy', mode='w+',
                                          dtype='float32', shape=(n, t, 32))
        with torch.inference_mode():
            for lo in range(0, n, args.batch):
                values[lo:lo + args.batch] = pretrained.encoder(
                    inputs.get(partition, slice(lo, lo + args.batch))).cpu().numpy()
        values.flush()
        atomic_json(marker, dict(encoder_hash=encoder_hash, windows=n, no_quantization=True))
    return EncodedArrays(output, inputs)


def initialize_memory(model, encoded, fit_indices, output, args):
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(fit_indices, min(2048, len(fit_indices)), replace=False))
    parts = []
    for lo in range(0, len(chosen), args.batch):
        parts.append(encoded.get('train', chosen[lo:lo + args.batch]).flatten(0, 1).cpu().numpy())
    features = np.concatenate(parts)
    selected = rng.choice(len(features), min(65536, len(features)), replace=False)
    model.memory.initialize(features[selected], args.seed)
    np.save(output / 'memory_initialization_window_indices.npy', chosen)
    np.save(output / 'memory_initialization_token_indices.npy', selected)
    atomic_json(output / 'memory_initialization.json', dict(normal_reconstruction_fit_only=True,
        calibration_vins_excluded=True, windows=len(chosen), time_step_vectors=len(selected),
        one_shared_bank=True, slots=8, similarity='cosine', temperature=.25,
        method='direction-normalized K-means, original-space cluster means'))


def infer(model, inputs, encoded, output, args, replacement=None, reference_mean=None):
    """Exact variable moments preserve existing squared-error scoring."""
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    diagnostics = {}
    for partition in PARTITIONS:
        marker = output / f'{partition}_inference_complete.json'
        if marker.exists():
            assert json.loads(marker.read_text())['model_hash'] == state_hash(model)
            continue
        n = len(inputs.arrays[partition])
        moments = np.lib.format.open_memmap(output / f'{partition}_moments.npy', mode='w+',
                                          dtype='float64', shape=(n, 4, 7))
        soft, hard = np.zeros(8), np.zeros(8)
        distances, count = [], 0
        state = np.lib.format.open_memmap(output / f'{partition}_memory_weights.npy', mode='w+',
            dtype='float32', shape=(n, 8)) if model.memory is not None and replacement is None else None
        permutation = np.random.default_rng(args.seed + 919).permutation(n) if replacement == 'shuffle' else None
        delta_total, original_total = 0., 0.
        with torch.inference_mode():
            for lo in range(0, n, args.batch):
                ix = slice(lo, lo + args.batch)
                h = encoded.get(partition, ix)
                reference = None
                if replacement == 'mean':
                    reference = reference_mean.expand_as(h)
                elif replacement == 'shuffle':
                    reference = model.memory(encoded.get(partition, permutation[ix]))[0]
                y, weights = model.decode(h, reference)
                x = inputs.get(partition, ix)
                residual = (x - y).double()
                absolute = residual.abs()
                m = torch.stack((residual.mean(1), residual.var(1, unbiased=False),
                                 absolute.mean(1), absolute.var(1, unbiased=False)), 1)
                moments[ix] = m.cpu().numpy()
                if replacement is not None:
                    original = model.decode(h)[0]
                    delta_total += float((y - original).double().square().sum())
                    original_total += float(original.double().square().sum())
                if state is not None:
                    w = weights.double()
                    state[ix] = w.mean(1).cpu().numpy()
                    soft += w.sum((0, 1)).cpu().numpy()
                    hard += torch.bincount(w.argmax(-1).flatten(), minlength=8).cpu().numpy()
                    count += w.shape[0] * w.shape[1]
                    p = F.normalize(model.memory.prototypes, dim=-1)
                    similarity = F.normalize(h, dim=-1) @ p.T
                    distances.append((1 - similarity.amax(-1)).mean(1).cpu().numpy())
        moments.flush()
        if state is not None:
            state.flush()
            soft, hard = soft / count, hard / count
            p = model.memory.prototypes.detach().cpu().numpy()
            off = ~np.eye(len(p), dtype=bool)
            cosine = p @ p.T / np.maximum(np.linalg.norm(p, axis=1)[:, None] *
                                         np.linalg.norm(p, axis=1)[None], 1e-12)
            distance = np.concatenate(distances)
            np.save(output / f'{partition}_memory_distance.npy', distance)
            diagnostics[partition] = dict(soft_occupancy=soft.tolist(), hard_occupancy=hard.tolist(),
                effective_soft_prototypes=float(np.exp(-(soft * np.log(np.maximum(soft, 1e-30))).sum())),
                hard_used_prototypes=int((hard > .01).sum()),
                pairwise_cosine_mean=float(cosine[off].mean()),
                pairwise_distance_mean=float(np.linalg.norm(p[:, None] - p[None], axis=-1)[off].mean()),
                cosine_distance_quantiles=np.quantile(distance, [.05, .5, .95, .99]).tolist())
            atomic_json(output / f'{partition}_memory_diagnostics.json', diagnostics[partition])
        atomic_json(marker, dict(model_hash=state_hash(model), windows=n, replacement=replacement,
            reconstruction_relative_squared_change=delta_total / max(original_total, 1e-30)))
        print('SIMPLE INFERRED', output, partition, n, flush=True)


def evaluate(unit, source, mapping, output, calibration_indices, calibration_source=None):
    names = ['raw', 'cal_abs']
    if calibration_source is None:
        train_m = np.load(output / 'train_moments.npy', mmap_mode='r')[calibration_indices]
        q = np.load(source / 'residuals/train/condition.npy', mmap_mode='r')[calibration_indices]
        calibration = FixedRobustResidualCalibration().fit(q, train_m[:, 2], train_m[:, 3])
    else:
        calibration = FixedRobustResidualCalibration.load(calibration_source / 'calibration_absolute.npz')
    calibration.save(output / 'calibration_absolute.npz')
    values = {}
    for partition in PARTITIONS:
        m = np.load(output / f'{partition}_moments.npy', mmap_mode='r')
        q = np.load(source / 'residuals' / partition / 'condition.npy', mmap_mode='r')
        values[partition] = np.column_stack(((m[:, 1] + m[:, 0] ** 2).mean(1),
                                            calibration.energy(q, m[:, 2], m[:, 3])))
        assert np.isfinite(values[partition]).all()
        np.save(output / f'{partition}_window_scores.npy', values[partition])
    frames, snippets = ranked_frames(source, mapping, values, names)
    evaluate_frames(unit, output, frames, names, snippets)
    atomic_json(output / 'complete.json', dict(unit=unit, scores=names, full_cohort=True,
        calibration_holdout=True, calibration_refit=calibration_source is None,
        primary_score='raw', primary_threshold='normal validation P95', test_selection=False))


def memory_replacements(model, inputs, encoded, fit_indices, output, source, mapping, cal_indices, args):
    mean = torch.zeros(32, device='cuda', dtype=torch.float64)
    count = 0
    with torch.inference_mode():
        for lo in range(0, len(fit_indices), args.batch):
            h = encoded.get('train', fit_indices[lo:lo + args.batch])
            retrieved, _ = model.memory(h)
            mean += retrieved.double().sum((0, 1))
            count += retrieved.shape[0] * retrieved.shape[1]
    mean = (mean / count).float()[None, None]
    np.save(output / 'normal_fit_retrieved_mean.npy', mean.cpu().numpy())
    for replacement in ['mean', 'shuffle']:
        target = output / ('A6_' + replacement)
        if (target / 'complete.json').exists():
            continue
        infer(model, inputs, encoded, target, args, replacement, mean)
        evaluate(output.parent.name, source, mapping, target, cal_indices, output)


def summarize(root, units, report):
    all_rows, training = [], []
    for unit in units:
        for arm in ARMS:
            p = root / unit / arm
            if not (p / 'complete.json').exists():
                continue
            df = pd.read_csv(p / 'metrics.csv')
            df['arm'], df['seed'], df['task'] = arm, 3407, TASK
            df['dataset'] = 'Brand3' if unit.startswith('Brand3') else unit
            all_rows.append(df)
            t = json.loads((p / 'training/complete.json').read_text())
            training.append(dict(unit=unit, arm=arm, **t))
    completed = len(training)
    text = ['# MSCM：模型一单种子结构筛选', '',
            f'任务：`{TASK}`。已完成 {completed}/{len(units) * len(ARMS)} 个结构实验。', '',
            '本轮为全正常窗口、seed=3407；预训练最多20轮，后续最多30轮，正常验证早停 patience=8。'
            '先训练无 Memory 编码器与弱 Decoder，再冻结编码器训练 Memory/普通融合分支与 Decoder。'
            'A1、A3、A5 复用相同多尺度预训练最佳权重；A2 使用单尺度预训练权重；'
            'A2-T 与 A3-T 分别使用单尺度和多尺度 Transformer×2 预训练权重。'
            '全部从新随机初始化开始，不复用旧 RCR 或复杂 Memory 权重。', '',
            '原始正常训练 VIN 预先分出20%用于独立校准，不参与编码器、Decoder 或 K-means 拟合。'
            '输入归一化沿用既有正常训练集统计，其中包含这部分校准 VIN；'
            '因此这里只隔离重构参数拟合与残差校准拟合，不宣称全流程独立校准。'
            '校准仅拟合连续条件均值与固定稳健尺度；阈值由正常验证 P95 决定。'
            '主比较使用原始平方残差评分；校准后结果单独比较。'
            'CH 保持片段 Top5% 窗口评分，Brand3 再按 VIN Top5% 片段聚合。', '',
            '新 A1/A2/A2-T/A3/A3-T/A5 编号来自本次简化设计，与历史 A4、A6 编号不同。'
            '这是已多次查看过测试结果的数据上的开发筛选，不是独立确认性验证。', '']
    if all_rows:
        table = pd.concat(all_rows, ignore_index=True)
        table.to_csv(root / 'completed_metrics.csv', index=False)
        pd.DataFrame(training).to_csv(root / 'training_completion.csv', index=False)
        unit_counts = table.query('q == .95 and method == "raw"').groupby('dataset').unit.nunique().to_dict()
        macro = table.query('q == .95').groupby(['arm', 'method', 'dataset'])[
            ['ap', 'auroc', 'precision', 'recall', 'f1', 'fpr']].mean()
        macro.to_csv(root / 'macro_metrics.csv')
        labels = {'A1': '多尺度，无 Memory', 'A2': '单尺度 + Memory',
                  'A2-T': '单尺度 + Transformer×2 + Memory',
                  'A3': '多尺度 + Memory', 'A3-T': '多尺度 + Transformer×2 + Memory',
                  'A5': '多尺度 + 同容量普通融合'}
        text += ['## RCR 与主要外部模型参考对比', '',
                 '本表把当前候选结构的完整 `cal_abs` 输出与 RCR、主要外部模型并列。'
                 '当前候选均为本任务 seed3407；外部结果来自 `RCR_REPORT_ZH.md`，† 为既有预算参考，'
                 '通常为10 epoch、最多100k正常训练窗口，训练预算与当前20+30轮全量训练不同。'
                 '因此本表用于定位性能，不用于配对归因或显著性结论。', '',
                 'Brand3统一列五折算术均值；RCR Scale使用主报告值 '
                 '`AP 0.8728 / AUROC 0.9240 / F1 0.7236`，不是另一轮本地重训检查点的'
                 '`0.8685 / 0.9156 / 0.6789`。§ Isolation Forest 的Brand3预处理不同，'
                 '仅作补充参考。外部产物未完成的单元保留“待补”。', '',
                 '| 方法 | LFP AP / AUROC / F1 / Recall / FPR | NCM AP / AUROC / F1 / Recall / FPR | Brand3 AP / AUROC / F1 / Recall / FPR |',
                 '| --- | --- | --- | --- |']
        for model, datasets in REFERENCE_MODELS.items():
            cells = []
            for dataset in ['LFP', 'NCM', 'Brand3']:
                values = datasets[dataset]
                cells.append('待补' if values is None else ' / '.join(f'{value:.4f}' for value in values))
            text.append('| ' + model + ' | ' + ' | '.join(cells) + ' |')
        for arm in ARMS:
            cells = []
            for dataset in ['LFP', 'NCM', 'Brand3']:
                key = (arm, 'cal_abs', dataset)
                units_done = table.query(
                    'q == .95 and method == "cal_abs" and arm == @arm and dataset == @dataset').unit.nunique()
                expected = sum(u.startswith('Brand3') for u in units) if dataset == 'Brand3' else int(dataset in units)
                if key not in macro.index or units_done < expected:
                    cells.append('待补')
                else:
                    row = macro.loc[key]
                    cells.append(' / '.join(f'{row[k]:.4f}' for k in ['ap', 'auroc', 'f1', 'recall', 'fpr']))
            text.append('| ' + arm + ' ' + labels[arm] + ' + cal_abs | ' + ' | '.join(cells) + ' |')
        text += ['', '粗体只表示本参考表中已完成数值的逐指标最优，不表示协议完全一致或统计显著。', '']
        for score, title in [('raw', '原始残差：结构主比较'), ('cal_abs', '重新拟合 RCR 条件均值 + 固定稳健尺度')]:
            text += [f'## {title}', '',
                     '| 方法 | LFP AP / AUROC / F1 / Recall / FPR | NCM AP / AUROC / F1 / Recall / FPR | Brand3 AP / AUROC / F1 / Recall / FPR |',
                     '| --- | --- | --- | --- |']
            for arm in ARMS:
                cells = []
                for dataset in ['LFP', 'NCM', 'Brand3']:
                    key = (arm, score, dataset)
                    units_done = table.query('q == .95 and method == @score and arm == @arm and dataset == @dataset').unit.nunique()
                    expected = sum(u.startswith('Brand3') for u in units) if dataset == 'Brand3' else int(dataset in units)
                    if key not in macro.index or units_done < expected:
                        cells.append('待补')
                    else:
                        row = macro.loc[key]
                        cells.append(' / '.join(f'{row[k]:.4f}' for k in ['ap', 'auroc', 'f1', 'recall', 'fpr']))
                text.append('| ' + arm + ' ' + labels[arm] + ' | ' + ' | '.join(cells) + ' |')
            text += ['', f'已有结果的数据单元：{unit_counts}。Brand3 只有所请求折全部完成后才填均值。', '']
        text += ['## 实际训练轮数', '',
                 '| 数据单元 | 方法 | 后续实际轮数 | 最佳轮数 | 最佳正常验证 Huber | 是否达到预算边界 |',
                 '| --- | --- | --- | --- | --- | --- |']
        for row in training:
            text.append(f'| {row["unit"]} | {row["arm"]} | {row["epochs_executed"]} | {row["best_epoch"]} | '
                        f'{row["normal_validation_huber"]:.7f} | {row["at_budget_boundary"]} |')
        text += ['', '预训练与后续成本分别记录于各目录的 `pretrain_*/complete.json` 和 `training/complete.json`；'
                     '共享预训练的计算成本不重复计算，但每个方法的逻辑预算均包含预训练。', '']
        text += ['## 预训练成本与参数量', '',
                 '| 数据单元 | 预训练编码器 | 实际 / 上限 epoch | 最佳 epoch | 训练秒数 |',
                 '| --- | --- | --- | --- | --- |']
        pre_records = []
        for unit in units:
            for kind in ['multi', 'single', 'single_T', 'multi_T']:
                p = root / unit / ('pretrain_' + kind) / 'complete.json'
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                pre_records.append(dict(unit=unit, kind=kind, **d))
                text.append(f'| {unit} | {kind} | {d["epochs_executed"]} / {d["max_epochs"]} | '
                            f'{d["best_epoch"]} | {d["seconds"]:.1f} |')
        pd.DataFrame(pre_records).to_csv(root / 'pretraining_completion.csv', index=False)
        text += ['', 'A3 与 A5 总参数均为18495、后续可训练参数均为2567。'
                 'A2-T/A3-T 分别比 A2/A3 增加两层 Transformer，参数量更大；'
                 '提升不能完全排除容量因素。', '']
        text += ['## Memory 替换诊断（A6，不另行训练）', '',
                 '冻结 A3 的全部权重和原校准器，分别用正常拟合集的平均检索表示、'
                 '分区内固定随机排列的检索表示替换 Memory 输出。本节检测指标统一按 A3 原阈值复算，'
                 '不采用替换分支各自的验证阈值。', '']
        diag_rows = []
        for unit in units:
            base = root / unit / 'A3'
            if not (base / 'complete.json').exists():
                continue
            original = pd.read_csv(base / 'metrics.csv').query('q == .95').set_index('method')
            thresholds = json.loads((base / 'thresholds.json').read_text())
            for replacement in ['mean', 'shuffle']:
                child = base / ('A6_' + replacement)
                if not (child / 'complete.json').exists():
                    continue
                d = json.loads((child / 'test_inference_complete.json').read_text())
                frame = pd.read_csv(child / 'test_scores.csv', dtype={'vin': str})
                from scripts.evaluate_rcr import metrics as score_metrics
                for name in ['raw', 'cal_abs']:
                    result = score_metrics(frame.sample_label, frame[name], thresholds[name]['0.95'])
                    diag_rows.append(dict(unit=unit, replacement=replacement, score=name,
                        relative_squared_output_change=d['reconstruction_relative_squared_change'],
                        delta_ap=result['ap'] - original.loc[name, 'ap'],
                        delta_auroc=result['auroc'] - original.loc[name, 'auroc'],
                        delta_f1=result['f1'] - original.loc[name, 'f1'], **result))
        if diag_rows:
            pd.DataFrame(diag_rows).to_csv(root / 'memory_replacement_diagnostics.csv', index=False)
            text += ['| 数据单元 | 替换方式 | 测试输出相对平方变化 | ΔAP（raw） | ΔAUROC（raw） | ΔF1（固定原阈值） |',
                     '| --- | --- | --- | --- | --- | --- |']
            for d in diag_rows:
                if d['score'] == 'raw':
                    text.append(f'| {d["unit"]} | {d["replacement"]} | {d["relative_squared_output_change"]:.6g} | '
                                f'{d["delta_ap"]:+.4f} | {d["delta_auroc"]:+.4f} | {d["delta_f1"]:+.4f} |')
            text.append('')
        text += ['## 结构比较', '',
                 '以下差值一律为左侧减右侧，按相同数据单元配对。正数仅表示本轮单种子的数值趋势，'
                 '不能作为跨种子稳定性或统计显著性的证明。', '',
                 '| 比较 | 评分 | LFP ΔAP / ΔAUROC / ΔF1 | NCM ΔAP / ΔAUROC / ΔF1 | Brand3 ΔAP / ΔAUROC / ΔF1 |',
                 '| --- | --- | --- | --- | --- |']
        comparisons = []
        for left, right in [('A3', 'A1'), ('A3', 'A2'), ('A3', 'A5'),
                            ('A2-T', 'A2'), ('A3-T', 'A3'), ('A3-T', 'A2-T')]:
            for score in ['raw', 'cal_abs']:
                cells = []
                for dataset in ['LFP', 'NCM', 'Brand3']:
                    expected_units = [u for u in units if ('Brand3' if u.startswith('Brand3') else u) == dataset]
                    a = table.query('q == .95 and method == @score and arm == @left and dataset == @dataset').set_index('unit')
                    b = table.query('q == .95 and method == @score and arm == @right and dataset == @dataset').set_index('unit')
                    if not expected_units or set(a.index) != set(expected_units) or set(b.index) != set(expected_units):
                        cells.append('待补')
                        continue
                    delta = (a.loc[expected_units, ['ap', 'auroc', 'f1']] - b.loc[expected_units, ['ap', 'auroc', 'f1']]).mean()
                    comparisons.append(dict(left=left, right=right, score=score, dataset=dataset, **delta.to_dict()))
                    cells.append(' / '.join(f'{delta[k]:+.4f}' for k in ['ap', 'auroc', 'f1']))
                text.append(f'| {left} − {right} | {score} | ' + ' | '.join(cells) + ' |')
        pd.DataFrame(comparisons).to_csv(root / 'paired_structure_deltas.csv', index=False)
        text += ['', '原型占用率、有效原型数、原型距离与检索距离分位数保存于每个 Memory 模型的 '
                 '`*_memory_diagnostics.json`；训练逐轮占用保存在 `training/memory_epoch_*.json`。'
                 '若最佳点仍在20或30轮上限附近，只能标记本轮受预算限制，不能据此宣称已收敛。', '']
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(bold_best_metrics('\n'.join(text) + '\n'))
    atomic_json(root / 'summary_status.json', dict(completed=completed, expected=len(units) * len(ARMS),
        requested_matrix_complete=completed == len(units) * len(ARMS), report=str(report)))


def run_unit(unit, root, args):
    unit_root = root / unit
    if all((unit_root / arm / 'complete.json').exists() for arm in ARMS) and all(
            (unit_root / 'A3' / ('A6_' + r) / 'complete.json').exists() for r in ['mean', 'shuffle']):
        print('SIMPLE UNIT REUSED', unit, 'all completed arms and diagnostics preserved', flush=True)
        return
    source, _ = source_for(unit)
    input_root = Path(args.input_cache) / 'inputs' / unit
    marker = json.loads((input_root / 'complete.json').read_text())
    assert marker['source_mapping_sha256'] == digest(source / 'mapping.json')
    np.testing.assert_equal(json.loads((input_root / 'mapping.json').read_text()),
                            json.loads((source / 'mapping.json').read_text()))
    fit_indices, cal_indices, mapping = normal_split(source, root / 'cohorts' / unit, args.seed)
    inputs = InputArrays(input_root)
    unit_root = root / unit
    unit_root.mkdir(parents=True, exist_ok=True)
    for kind, pre_arm, arms in [('multi', 'pre_multi', ['A1', 'A3', 'A5']),
                                ('single', 'pre_single', ['A2']),
                                ('single_T', 'pre_single_T', ['A2-T']),
                                ('multi_T', 'pre_multi_T', ['A3-T'])]:
        if all((unit_root / arm / 'complete.json').exists() for arm in arms) and (
                kind != 'multi' or all((unit_root / 'A3' / ('A6_' + r) / 'complete.json').exists()
                                        for r in ['mean', 'shuffle'])):
            continue
        seed_all(args.seed)
        pretrained = SimpleMSCM(pre_arm).cuda()
        pretrained = train(pretrained, inputs, fit_indices, unit_root / ('pretrain_' + kind), args,
                           'pretrain', args.pretrain_epochs)
        encoded = encode(pretrained, inputs, unit_root / ('encoded_' + kind), args)
        for arm in arms:
            output = unit_root / arm
            output.mkdir(parents=True, exist_ok=True)
            seed_all(args.seed)
            model = SimpleMSCM(arm).cuda()
            model.load_pretrained(pretrained)
            if model.memory is not None and not (output / 'training/last.pt').exists():
                initialize_memory(model, encoded, fit_indices, output, args)
            model = train(model, inputs, fit_indices, output / 'training', args,
                          'adapt', args.adapt_epochs, encoded)
            assert state_hash(model.encoder) == state_hash(pretrained.encoder)
            if not (output / 'complete.json').exists():
                infer(model, inputs, encoded, output, args)
                evaluate(unit, source, mapping, output, cal_indices)
            if arm == 'A3':
                memory_replacements(model, inputs, encoded, fit_indices, output, source, mapping, cal_indices, args)
            summarize(root, args.units, Path(args.report))
            del model
            torch.cuda.empty_cache()
        del pretrained, encoded
        gc.collect()
        torch.cuda.empty_cache()
    del inputs
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(unit_root / 'complete.json', dict(unit=unit, arms=list(ARMS), seed=args.seed,
        memory_replacements=['mean', 'shuffle'], new_simplified_architecture=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT_ROOT)
    parser.add_argument('--input-cache', default='runs/internal/mscm_mcr_development_20261025_v2')
    parser.add_argument('--report', default=DEFAULT_REPORT)
    parser.add_argument('--units', nargs='+', choices=UNITS, default=UNITS)
    parser.add_argument('--seed', type=int, choices=[3407], default=3407)
    parser.add_argument('--pretrain-epochs', type=int, default=20)
    parser.add_argument('--adapt-epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(root, args.units, Path(args.report))
        return
    assert 0 < args.pretrain_epochs <= 30 and 0 < args.adapt_epochs <= 30
    assert args.patience > 0
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required outside sandbox')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    protocol = dict(task=TASK, units=args.units, arms=list(ARMS), seed=args.seed,
        pretrain_epochs=args.pretrain_epochs, adaptation_epochs=args.adapt_epochs,
        max_total_epochs_per_arm=args.pretrain_epochs + args.adapt_epochs,
        patience=args.patience, scheduler_patience=5, batch=args.batch, hidden=32, slots=8,
        kernels=[3, 7, 15], single_scale_kernel=7, temperature=.25, memory_trainable=True,
        variable_graph=True, decoder='concat(H,Hm) -> Linear(64,32) -> GELU -> Linear(32,7)',
        transformer_control='A2-T/A3-T: two Transformer layers after single/multiscale fusion, before Memory; heads4/ffn64/dropout0.1',
        loss='normal-only Huber; no auxiliary loss', encoder_frozen_during_adaptation=True,
        initialization='scratch; shared best normal-pretraining checkpoint per scale setting',
        capacity_control='ordinary bias-free 32->4->32 MLP; exactly same total/adaptation parameter count as A3',
        calibration_holdout_fraction=.2, primary_score='raw squared residual',
        normalization='reuse original normal-training statistics including calibration subset; only optimizer/K-means fitting VINs are held out',
        additional_score='refitted conditional absolute mean / fixed robust scale',
        primary_threshold='normal validation P95', test_labels_used_for_selection=False,
        memory_replacement='normal-fit mean and seeded partition permutation; freeze original A3 calibration')
    if (root / 'protocol.json').exists():
        existing = json.loads((root / 'protocol.json').read_text())
        legacy = dict(protocol)
        legacy['arms'] = ['A1', 'A2', 'A3', 'A3-T', 'A5']
        legacy['transformer_control'] = ('A3-T only: two Transformer layers after multiscale fusion, '
                                         'before Memory; heads4/ffn64/dropout0.1')
        assert existing in [legacy, protocol], 'Protocol changed beyond the requested A2-T extension'
        if existing == legacy:
            atomic_json(root / 'protocol_extension_A2_T.json', dict(
                requested_extension='A2-T: single-scale + Transformer×2 + Memory; raw and cal_abs evaluation',
                previous_arms=legacy['arms'], extended_arms=list(ARMS), seed=args.seed,
                weights_reused=False, completed_arms_preserved=True, test_selection=False))
    atomic_json(root / 'protocol.json', protocol)
    with (root / 'runner.lock').open('w') as lock, threadpool_limits(limits=4):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(root / 'pipeline_status.json', dict(status='running', units=args.units))
        try:
            for unit in args.units:
                atomic_json(root / 'pipeline_status.json', dict(status='running', active_unit=unit))
                run_unit(unit, root, args)
            summarize(root, args.units, Path(args.report))
            atomic_json(root / 'complete.json', dict(task=TASK, units=args.units, arms=list(ARMS),
                                                    seed=args.seed, requested_matrix_complete=True))
            atomic_json(root / 'pipeline_status.json', dict(status='complete', active_unit=None))
            print('SIMPLE SCREEN COMPLETE', TASK, flush=True)
        except Exception:
            atomic_json(root / 'pipeline_status.json', dict(status='failed', traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
