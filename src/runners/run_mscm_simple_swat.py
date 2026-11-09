"""SWaT transfer of the simplified MSCM screening; original cache implementation.

All five structures share the same normal temporal split and endpoint protocol.
No test labels enter fitting, checkpoint selection, or threshold selection.
"""
import argparse
import fcntl
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from scripts.evaluate_rcr import metrics
from src.data.external_protocol import load_external_protocol_data
from src.models.mscm_simple import SimpleMSCM
from src.models.mscm_reconstruction import FixedRobustResidualCalibration
from src.project_paths import resolve_dataset_root
from src.runners.run_mscm_development import InputArrays
from src.runners.run_mscm_simple_screen import ARMS, PARTITIONS, seed_all, train, encode, initialize_memory
from src.runners.run_rcr import atomic_json, state_hash
from src.runners.report_metric_format import bold_best_metrics


TASK = 'MSCM-SIMPLE-SWAT-20261107-s3407'
ROOT = 'runs/internal/mscm_simple_swat_20261107_s3407'
REPORT = 'report/analysis/rcr/MSCM_SIMPLE_SWAT_20261107.md'


def temporal_fit_calibration(starts, window, normal_points, fraction=.2):
    """Discard every window crossing the fit/calibration time boundary."""
    cut = int(np.floor(normal_points * (1 - fraction)))
    starts = np.asarray(starts, dtype=np.int64)
    fit = np.flatnonzero(starts + window <= cut)
    calibration = np.flatnonzero(starts >= cut)
    assert len(fit) and len(calibration) and not np.intersect1d(fit, calibration).size
    return fit, calibration, cut


def actuator_context(x, indices):
    """Normal actuator state means and transition rates; no sensor response q."""
    if not len(indices):
        raise ValueError('SWaT actuator columns are required for conditional calibration')
    a = x[:, :, indices]
    return torch.cat((a.mean(1), a.diff(dim=1).abs().mean(1)), 1)


def prepare(root, args):
    target = root / 'inputs'
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'complete.json').exists():
        saved = json.loads((target / 'complete.json').read_text())
        assert saved['window'] == args.window and saved['stride'] == args.stride
        return saved
    data = load_external_protocol_data('SWAT', seed=args.seed, val_ratio=.1)
    names = pd.read_csv(resolve_dataset_root('SWAT') / 'processed/SWAT_feature_names.csv')['feature'].tolist()
    dimensions = data.train_sequences[0].shape[1]
    assert len(names) == dimensions
    controls = [i for i, name in enumerate(names) if re.fullmatch(r'(?:MV|P|UV)\d+', name.strip())]
    info = dict(dataset='SWAT', variables=dimensions, feature_names=names, control_indices=controls,
                control_names=[names[i] for i in controls], window=args.window, stride=args.stride,
                data_metadata=data.metadata, windows={})
    for partition in PARTITIONS:
        sequence = getattr(data, partition + '_sequences')[0]
        starts = np.arange(0, len(sequence) - args.window + 1, args.stride, dtype=np.int64)
        ends = starts + args.window - 1
        x = np.lib.format.open_memmap(target / f'{partition}_x.npy', mode='w+', dtype='float32',
                                     shape=(len(starts), args.window, dimensions))
        q = np.lib.format.open_memmap(target / f'{partition}_q.npy', mode='w+', dtype='float64',
                                     shape=(len(starts), 2 * len(controls)))
        for lo in range(0, len(starts), args.batch):
            selected = starts[lo:lo + args.batch, None] + np.arange(args.window)[None]
            values = np.asarray(sequence[selected], dtype=np.float32)
            x[lo:lo + len(values)] = values
            q[lo:lo + len(values)] = actuator_context(torch.from_numpy(values), controls).numpy()
        x.flush(); q.flush()
        np.save(target / f'{partition}_starts.npy', starts)
        np.save(target / f'{partition}_endpoints.npy', ends)
        # Same sampled-endpoint labels as the existing external SWaT runners.
        labels = np.asarray(data.test_labels[0], dtype=np.int32)[ends] if partition == 'test' else np.zeros(len(ends), dtype=np.int32)
        np.save(target / f'{partition}_labels.npy', labels)
        if partition == 'train':
            fit, calibration, cut = temporal_fit_calibration(starts, args.window, len(sequence))
            np.save(target / 'fit_window_indices.npy', fit)
            np.save(target / 'calibration_window_indices.npy', calibration)
            info.update(fit_windows=len(fit), calibration_windows=len(calibration),
                        fit_calibration_boundary=cut, boundary_crossing_windows_excluded=len(starts) - len(fit) - len(calibration))
        info['windows'][partition] = len(starts)
        print('SWAT INPUT PREPARED', partition, len(starts), flush=True)
        del x, q
    atomic_json(target / 'complete.json', info)
    return info


def infer(model, inputs, encoded, output, args, replacement=None, mean_reference=None):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    dimensions = inputs.arrays['train'].shape[-1]
    for partition in PARTITIONS:
        done = output / f'{partition}_inference_complete.json'
        if done.exists():
            assert json.loads(done.read_text())['model_hash'] == state_hash(model)
            continue
        n = len(inputs.arrays[partition])
        moments = np.lib.format.open_memmap(output / f'{partition}_moments.npy', mode='w+',
                                          dtype='float64', shape=(n, 4, dimensions))
        soft, hard, distances = np.zeros(8), np.zeros(8), []
        tokens, change, baseline = 0, 0., 0.
        permutation = np.random.default_rng(args.seed + 919).permutation(n) if replacement == 'shuffle' else None
        with torch.inference_mode():
            for lo in range(0, n, args.batch):
                ix = slice(lo, lo + args.batch)
                h = encoded.get(partition, ix)
                reference = mean_reference.expand_as(h) if replacement == 'mean' else None
                if replacement == 'shuffle':
                    reference = model.memory(encoded.get(partition, permutation[ix]))[0]
                y, weights = model.decode(h, reference)
                residual = (inputs.get(partition, ix) - y).double()
                absolute = residual.abs()
                moments[ix] = torch.stack((residual.mean(1), residual.var(1, unbiased=False),
                                           absolute.mean(1), absolute.var(1, unbiased=False)), 1).cpu().numpy()
                if replacement is not None:
                    original = model.decode(h)[0]
                    change += float((y - original).double().square().sum())
                    baseline += float(original.double().square().sum())
                if weights is not None and replacement is None:
                    w = weights.double()
                    soft += w.sum((0, 1)).cpu().numpy()
                    hard += torch.bincount(w.argmax(-1).flatten(), minlength=8).cpu().numpy()
                    tokens += w.shape[0] * w.shape[1]
                    similarity = torch.nn.functional.normalize(h, dim=-1) @ torch.nn.functional.normalize(model.memory.prototypes, dim=-1).T
                    distances.append((1 - similarity.amax(-1)).mean(1).cpu().numpy())
        moments.flush()
        if tokens:
            soft, hard = soft / tokens, hard / tokens
            p = model.memory.prototypes.detach().cpu().numpy()
            norm = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
            off = ~np.eye(8, dtype=bool)
            atomic_json(output / f'{partition}_memory_diagnostics.json', dict(
                soft_occupancy=soft.tolist(), hard_occupancy=hard.tolist(),
                effective_soft_prototypes=float(np.exp(-(soft * np.log(np.maximum(soft, 1e-30))).sum())),
                hard_used_prototypes=int((hard > .01).sum()), pairwise_cosine_mean=float((norm @ norm.T)[off].mean()),
                pairwise_distance_mean=float(np.linalg.norm(p[:, None] - p[None], axis=-1)[off].mean()),
                cosine_distance_quantiles=np.quantile(np.concatenate(distances), [.05, .5, .95, .99]).tolist()))
        atomic_json(done, dict(model_hash=state_hash(model), windows=n, replacement=replacement,
                              reconstruction_relative_squared_change=change / max(baseline, 1e-30)))
        print('SWAT INFERRED', output.name, partition, n, flush=True)


def evaluate(root, output, calibration_indices, frozen_calibration=None):
    source = root / 'inputs'
    if frozen_calibration is None:
        m = np.load(output / 'train_moments.npy', mmap_mode='r')[calibration_indices]
        q = np.load(source / 'train_q.npy', mmap_mode='r')[calibration_indices]
        calibration = FixedRobustResidualCalibration().fit(q, m[:, 2], m[:, 3])
    else:
        calibration = FixedRobustResidualCalibration.load(frozen_calibration / 'calibration_absolute.npz')
    calibration.save(output / 'calibration_absolute.npz')
    frames = {}
    for partition in PARTITIONS:
        m = np.load(output / f'{partition}_moments.npy', mmap_mode='r')
        q = np.load(source / f'{partition}_q.npy', mmap_mode='r')
        values = np.column_stack(((m[:, 1] + m[:, 0] ** 2).mean(1), calibration.energy(q, m[:, 2], m[:, 3])))
        assert np.isfinite(values).all()
        np.save(output / f'{partition}_window_scores.npy', values)
        frame = pd.DataFrame(dict(endpoint=np.load(source / f'{partition}_endpoints.npy'),
                                  sample_label=np.load(source / f'{partition}_labels.npy'),
                                  raw=values[:, 0], cal_abs=values[:, 1]))
        frame.to_csv(output / f'{partition}_scores.csv', index=False)
        frames[partition] = frame
    assert frames['validation'].sample_label.eq(0).all()
    rows, thresholds = [], {}
    original_thresholds = json.loads((frozen_calibration / 'thresholds.json').read_text()) if frozen_calibration else None
    for score in ['raw', 'cal_abs']:
        thresholds[score] = {}
        for quantile in [.95, .99]:
            threshold = original_thresholds[score][str(quantile)] if original_thresholds else float(frames['validation'][score].quantile(quantile))
            thresholds[score][str(quantile)] = threshold
            result = metrics(frames['test'].sample_label, frames['test'][score], threshold)
            rows.append(dict(dataset='SWAT', arm=output.name, method=score, q=quantile, threshold=threshold,
                             seed=3407, task=TASK, validation_fpr=float((frames['validation'][score] >= threshold).mean()), **result))
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
    atomic_json(output / 'thresholds.json', thresholds)
    atomic_json(output / 'complete.json', dict(dataset='SWAT', primary_score='raw', primary_q=.95,
        point_adjustment=False, endpoint_labels=True, test_selection=False, calibration_fit_only_normal_suffix=True,
        frozen_calibration_and_thresholds=frozen_calibration is not None))


def summarize(root, report):
    parts = [pd.read_csv(root / arm / 'metrics.csv') for arm in ARMS if (root / arm / 'complete.json').exists()]
    text = ['# MSCM：SWaT 单种子结构筛选', '', f'任务：`{TASK}`；已完成 {len(parts)}/{len(ARMS)} 组。', '',
            '结构为 A1、A2、A2-T、A3、A3-T、A5；seed=3407，预训练最多20轮、冻结编码器后最多30轮，patience=8。'
            '使用原缓存读取实现。40个输入变量，隐藏维32，单 Memory 8个原型，弱 Decoder。', '',
            '沿用仓库 SWaT 时序划分：正常流前90%为训练前缀、末10%为验证；'
            '归一化仅在正常训练前缀拟合。训练前缀再按时间分出末20%拟合校准器，'
            '跨越重构/校准边界的窗口不用于两者拟合。归一化统计包含校准段，不宣称全流程独立校准。', '',
            '窗口100、步长10，按窗口末端时间点的真实标签评估，不用窗口内任意异常标签扩展，'
            '不做 point adjustment、不插值成完整逐点评分。原始评分为窗口MSE，校准评分为绝对残差的'
            '连续条件均值加固定稳健尺度。工况上下文来自 MV/P/UV 泵阀执行器状态均值和变化率，'
            '不把电池电流/SOC索引套用于 SWaT。', '',
            'P95为本轮模型一主阈值，另列P99以便核对既有SWaT协议。不能跨阈值挑测试最优结果。', '']
    if parts:
        all_metrics = pd.concat(parts, ignore_index=True)
        all_metrics.to_csv(root / 'completed_metrics.csv', index=False)
        for quantile in [.95, .99]:
            for score in ['raw', 'cal_abs']:
                frame = all_metrics.query('q == @quantile and method == @score').set_index('arm')
                text += [f'## {score}，正常验证 P{int(quantile * 100)}', '',
                         '| 方法 | SWaT AP / AUROC / F1 / Recall / FPR |', '| --- | --- |']
                for arm in ARMS:
                    cell = '待补' if arm not in frame.index else ' / '.join(f'{frame.loc[arm, k]:.4f}' for k in ['ap', 'auroc', 'f1', 'recall', 'fpr'])
                    text.append(f'| {arm} | {cell} |')
                text.append('')
        text += ['## 训练轮数与 Memory 检查', '',
                 '| 方法 | 后续实际轮数 | 最佳验证轮数 | 总参数 | 后续训练参数 |',
                 '| --- | --- | --- | --- | --- |']
        for arm in ARMS:
            p = root / arm / 'training/complete.json'
            if p.exists():
                d = json.loads(p.read_text())
                text.append(f'| {arm} | {d["epochs_executed"]} | {d["best_epoch"]} | '
                            f'{d["parameters"]} | {d["trainable_parameters"]} |')
        text += ['', 'A3/A5 参数量由完成审核验证相等。Memory 占用率、有效原型数和原型距离存于各模型的 '
                 '`*_memory_diagnostics.json`。A3 均值/打乱替换使用冻结校准器及原阈值，不另行训练。', '',
                 '| A3 Memory 替换 | 测试输出相对平方变化 | ΔAP raw | ΔAUROC raw | ΔF1 raw（原P95阈值） |',
                 '| --- | --- | --- | --- | --- |']
        original = root / 'A3/metrics.csv'
        if original.exists() and (root / 'A3/complete.json').exists():
            baseline = pd.read_csv(original).query('q == .95 and method == "raw"').iloc[0]
            for replacement in ['mean', 'shuffle']:
                child = root / 'A3' / ('A6_' + replacement)
                if not (child / 'complete.json').exists():
                    continue
                row = pd.read_csv(child / 'metrics.csv').query('q == .95 and method == "raw"').iloc[0]
                d = json.loads((child / 'test_inference_complete.json').read_text())
                text.append(f'| {replacement} | {d["reconstruction_relative_squared_change"]:.6g} | '
                            f'{row.ap - baseline.ap:+.4f} | {row.auroc - baseline.auroc:+.4f} | {row.f1 - baseline.f1:+.4f} |')
        text += ['', '本轮为一个随机种子的结构筛选；原型占用不等于独立贡献。'
                 'A2-T/A3-T 含两层 Transformer，参数更多，其收益不能直接排除容量影响。'
                 '最佳点在预算边界时，不能宣称已充分收敛。', '']
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(bold_best_metrics('\n'.join(text) + '\n'))
    atomic_json(root / 'summary_status.json', dict(completed=len(parts), expected=len(ARMS),
        requested_matrix_complete=len(parts) == len(ARMS), report=str(report)))


def audit(root):
    rows = []
    fit = np.load(root / 'inputs/fit_window_indices.npy')
    calibration = np.load(root / 'inputs/calibration_window_indices.npy')
    starts = np.load(root / 'inputs/train_starts.npy')
    info = json.loads((root / 'inputs/complete.json').read_text())
    assert not np.intersect1d(fit, calibration).size
    assert (starts[fit] + info['window'] <= info['fit_calibration_boundary']).all()
    assert (starts[calibration] >= info['fit_calibration_boundary']).all()
    for arm in ARMS:
        output = root / arm
        training = json.loads((output / 'training/complete.json').read_text())
        assert training['encoder_frozen'] and not training['test_labels_used'] and training['epochs_executed'] <= 30
        if arm in ['A2', 'A2-T', 'A3', 'A3-T']:
            initialized = np.load(output / 'memory_initialization_window_indices.npy')
            assert np.isin(initialized, fit).all()
        test = pd.read_csv(output / 'test_scores.csv', float_precision='round_trip')
        validation = pd.read_csv(output / 'validation_scores.csv', float_precision='round_trip')
        error = 0.
        for row in pd.read_csv(output / 'metrics.csv', float_precision='round_trip').itertuples():
            np.testing.assert_allclose(row.threshold, validation[row.method].quantile(row.q), rtol=1e-12)
            for key, value in metrics(test.sample_label, test[row.method], row.threshold).items():
                error = max(error, abs(value - getattr(row, key)))
        assert error < 1e-10
        np.testing.assert_array_equal(test.endpoint.to_numpy(), np.load(root / 'inputs/test_endpoints.npy'))
        np.testing.assert_array_equal(test.sample_label.to_numpy(), np.load(root / 'inputs/test_labels.npy'))
        rows.append(dict(arm=arm, replay_error=error, endpoint_mapping_verified=True))
    a3 = json.loads((root / 'A3/training/complete.json').read_text())
    a5 = json.loads((root / 'A5/training/complete.json').read_text())
    assert a3['parameters'] == a5['parameters'] and a3['trainable_parameters'] == a5['trainable_parameters']
    pd.DataFrame(rows).to_csv(root / 'audit_units.csv', index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=ROOT)
    parser.add_argument('--report', default=REPORT)
    parser.add_argument('--seed', type=int, choices=[3407], default=3407)
    parser.add_argument('--pretrain-epochs', type=int, default=20)
    parser.add_argument('--adapt-epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    root, report = Path(args.output), Path(args.report)
    root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(root, report)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Run SWaT CUDA outside sandbox')
    assert args.pretrain_epochs <= 30 and args.adapt_epochs <= 30
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    protocol = dict(task=TASK, dataset='SWAT', seed=args.seed, arms=list(ARMS), window=args.window,
        stride=args.stride, batch=args.batch, pretrain_epochs=args.pretrain_epochs, adapt_epochs=args.adapt_epochs,
        patience=args.patience, validation_fraction=.1, calibration_fraction_of_train_prefix=.2,
        primary_quantile=.95, secondary_quantile=.99, point_adjustment=False, evaluation='sampled window-endpoint point ranking',
        control_schema='MV/P/UV actuator-tag columns; per-window means and mean absolute differences',
        normalization='existing SWaT external adapter, fitted on normal training prefix including calibration suffix',
        cache_mode='original InputArrays/EncodedArrays; no new GPU fit cache or chunk-cursor changes',
        test_selection=False, memory_slots=8, hidden=32, loss='normal Huber only')
    if (root / 'protocol.json').exists():
        assert json.loads((root / 'protocol.json').read_text()) == protocol
    atomic_json(root / 'protocol.json', protocol)
    with (root / 'runner.lock').open('w') as lock, threadpool_limits(limits=4):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(root / 'pipeline_status.json', dict(status='preparing'))
        try:
            info = prepare(root, args)
            inputs = InputArrays(root / 'inputs')
            fit = np.load(root / 'inputs/fit_window_indices.npy')
            calibration = np.load(root / 'inputs/calibration_window_indices.npy')
            for kind, pre_arm, arms in [('multi', 'pre_multi', ['A1', 'A3', 'A5']),
                                        ('single', 'pre_single', ['A2']),
                                        ('single_T', 'pre_single_T', ['A2-T']),
                                        ('multi_T', 'pre_multi_T', ['A3-T'])]:
                if all((root / arm / 'complete.json').exists() for arm in arms) and (kind != 'multi' or all(
                        (root / 'A3' / ('A6_' + replacement) / 'complete.json').exists() for replacement in ['mean', 'shuffle'])):
                    continue
                seed_all(args.seed)
                pretrained = SimpleMSCM(pre_arm, variables=info['variables']).cuda()
                pretrained = train(pretrained, inputs, fit, root / ('pretrain_' + kind), args, 'pretrain', args.pretrain_epochs)
                encoded = encode(pretrained, inputs, root / ('encoded_' + kind), args)
                for arm in arms:
                    atomic_json(root / 'pipeline_status.json', dict(status='running', arm=arm))
                    output = root / arm
                    output.mkdir(parents=True, exist_ok=True)
                    seed_all(args.seed)
                    model = SimpleMSCM(arm, variables=info['variables']).cuda()
                    model.load_pretrained(pretrained)
                    if model.memory is not None and not (output / 'training/last.pt').exists() and not (output / 'training/complete.json').exists():
                        initialize_memory(model, encoded, fit, output, args)
                    model = train(model, inputs, fit, output / 'training', args, 'adapt', args.adapt_epochs, encoded)
                    assert state_hash(model.encoder) == state_hash(pretrained.encoder)
                    if not (output / 'complete.json').exists():
                        infer(model, inputs, encoded, output, args)
                        evaluate(root, output, calibration)
                    if arm == 'A3' and not all((output / ('A6_' + r) / 'complete.json').exists() for r in ['mean', 'shuffle']):
                        mean = torch.zeros(32, device='cuda', dtype=torch.float64)
                        tokens = 0
                        with torch.inference_mode():
                            for lo in range(0, len(fit), args.batch):
                                h = encoded.get('train', fit[lo:lo + args.batch])
                                retrieved = model.memory(h)[0]
                                mean += retrieved.double().sum((0, 1)); tokens += retrieved.shape[0] * retrieved.shape[1]
                        mean = (mean / tokens).float()[None, None]
                        for replacement in ['mean', 'shuffle']:
                            child = output / ('A6_' + replacement)
                            if not (child / 'complete.json').exists():
                                infer(model, inputs, encoded, child, args, replacement, mean)
                                evaluate(root, child, calibration, output)
                    summarize(root, report)
                    del model
                    torch.cuda.empty_cache()
                del pretrained, encoded
                gc.collect(); torch.cuda.empty_cache()
            checks = audit(root)
            summarize(root, report)
            atomic_json(root / 'complete.json', dict(task=TASK, requested_matrix_complete=True, audited_units=len(checks), report=str(report)))
            atomic_json(root / 'pipeline_status.json', dict(status='complete'))
            print('SWAT SIMPLE SCREEN COMPLETE', TASK, flush=True)
        except Exception:
            import traceback
            atomic_json(root / 'pipeline_status.json', dict(status='failed', traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
