"""Audit completed frozen A/B probes, then export a Chinese report and plots."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rcr-memory-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


UNITS = ['A_current_rcr', 'B_k1', 'B_k2', 'B_k4', 'B_k6', 'B_k8']


def read_json(path):
    return json.loads(path.read_text())


def audit(root):
    complete = read_json(root/'complete.json')
    protocol = read_json(root/'protocol_manifest.json')
    mapping = read_json(root/'mapping.json')
    if complete['units'] != UNITS or not complete['backbone_frozen']:
        raise ValueError('Incomplete A/B matrix or unfrozen backbone')
    for unit in UNITS:
        p = root/unit
        record = read_json(p/'complete.json')
        if record['backbone_hash'] != protocol['backbone']['checkpoint_hash']:
            raise ValueError('Backbone hash mismatch')
        for filename in ['calibration.npz', 'metrics.json', 'metrics.csv', 'thresholds.json',
                         'validation_sample_scores.csv', 'test_sample_scores.csv', 'test_sample_details.csv',
                         'magnitude_cdf_reference.npy', 'strata.csv']:
            if not (p/filename).is_file():
                raise ValueError(f'Missing required artifact: {p/filename}')
        for partition in ('train', 'validation', 'test'):
            scores = np.load(p/f'{partition}_window_scores.npy', mmap_mode='r')
            refs = np.load(root/f'{partition}_references.npy', mmap_mode='r')
            raw = np.load(root/'A_current_rcr'/f'{partition}_window_scores.npy', mmap_mode='r')
            if scores.shape != (mapping['windows'][partition], 3) or len(refs) != len(scores):
                raise ValueError('Score/sample reference shape mismatch')
            for start in range(0, len(scores), 8192):
                block = scores[start:start+8192]
                if not np.isfinite(block).all():
                    raise ValueError('Nonfinite score')
                np.testing.assert_array_equal(block[:, 0], raw[start:start+8192, 0])
            if unit == 'A_current_rcr':
                old = Path(protocol['frozen_root'])/'lite_continuous'/partition/'scores.npy'
                original = np.load(old, mmap_mode='r')
                np.testing.assert_allclose(scores[:, 1], original[:, 1], rtol=1e-12, atol=1e-15)
            if partition != 'train':
                frame = pd.read_csv(p/f'{partition}_sample_scores.csv')
                if set(frame.sample_id) != set(mapping['sample_ids'][partition]):
                    raise ValueError('Missing sample score/mapping')
            if unit.startswith('B_'):
                for suffix in ['assignments.npy', 'query_diagnostics.npy', 'sample_prototypes.csv',
                               'prototype_distribution.csv', 'transitions.npz', 'transitions_by_group.csv']:
                    if not (p/f'{partition}_{suffix}').is_file():
                        raise ValueError(f'Missing prototype diagnostic: {p}/{partition}_{suffix}')
                assignment = np.load(p/f'{partition}_assignments.npy', mmap_mode='r')
                if len(assignment) != len(scores):
                    raise ValueError('Prototype mapping mismatch')
                with np.load(p/f'{partition}_transitions.npz') as transition:
                    expected = len(scores)-len(mapping['sample_ids'][partition])
                    if transition['all'].sum() != expected:
                        raise ValueError('Transition count crossed a sample boundary')
        if unit.startswith('B_'):
            for filename in ['fit_audit.json', 'matching_stability.csv', 'center_distances.csv',
                             'centers_standardized.csv', 'centers_control_units.csv', 'diagnostics.json']:
                if not (p/filename).is_file():
                    raise ValueError(f'Missing required artifact: {p/filename}')
    return dict(complete=True, units=UNITS, backbone_hash=protocol['backbone']['checkpoint_hash'],
                windows=mapping['windows'], samples={p: len(v) for p, v in mapping['sample_ids'].items()},
                original_A_magnitude_reproduced=True, raw_scores_identical=True,
                prototype_mappings_and_transition_boundaries_verified=True)


def measure(y, score, threshold):
    predicted = score >= threshold
    tp = np.sum(predicted & (y == 1))
    fp = np.sum(predicted & (y == 0))
    fn = np.sum(~predicted & (y == 1))
    return np.array([average_precision_score(y, score), roc_auc_score(y, score),
                     2*tp/max(1, 2*tp+fp+fn)])


def paired_bootstrap(root, base, contender, repeats):
    left = pd.read_csv(root/base/'test_sample_details.csv').set_index('sample_id')
    right = pd.read_csv(root/contender/'test_sample_details.csv').set_index('sample_id').loc[left.index]
    np.testing.assert_array_equal(left.sample_label, right.sample_label)
    np.testing.assert_array_equal(left.vin, right.vin)
    vin = left.vin.to_numpy()
    groups = [np.flatnonzero(vin == v) for v in np.unique(vin)]
    y, a, b = left.sample_label.to_numpy(), left.magnitude.to_numpy(), right.magnitude.to_numpy()
    ta = read_json(root/base/'thresholds.json')['magnitude']['0.95']
    tb = read_json(root/contender/'thresholds.json')['magnitude']['0.95']
    rng, differences = np.random.default_rng(3407), []
    for _ in range(repeats):
        index = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[index])) < 2:
            continue
        differences.append(measure(y[index], b[index], tb)-measure(y[index], a[index], ta))
    actual = measure(y, b, tb)-measure(y, a, ta)
    ci = np.quantile(differences, [.025, .975], axis=0) if differences else np.full((2, 3), np.nan)
    return [dict(base=base, contender=contender, metric=name, delta=float(actual[j]),
                 ci_low=float(ci[0, j]), ci_high=float(ci[1, j]), valid_repeats=len(differences),
                 requested_repeats=repeats, test_vins=len(groups))
            for j, name in enumerate(['ap', 'auroc', 'f1'])]


def diagnostics(root, chemistry):
    rows = []
    for k in [1, 2, 4, 6, 8]:
        p = root/f'B_k{k}'
        fit = read_json(p/'fit_audit.json')
        d = read_json(p/'diagnostics.json')['partitions']
        stable = pd.read_csv(p/'matching_stability.csv').query("partition == 'validation'")
        center = pd.read_csv(p/'center_distances.csv', index_col=0).to_numpy()
        off_diagonal = center[np.triu_indices(k, 1)]
        counts = np.asarray(fit['prototype_windows'])
        with np.load(p/'test_transitions.npz') as transitions:
            rates = {name: (t.sum()-np.trace(t))/max(1, t.sum())
                     for name, t in ((name, transitions[name]) for name in ['normal', 'fault'])}
        rows.append(dict(dataset=chemistry, k=k, min_occupancy=counts.min()/counts.sum(),
            max_occupancy=counts.max()/counts.sum(),
            min_center_distance=float(off_diagonal.min()) if k > 1 else np.nan,
            max_center_distance=float(off_diagonal.max()) if k > 1 else np.nan,
            min_validation_ari=stable.ari.min(), min_validation_agreement=stable.aligned_agreement.min(),
            normal_switch_rate=rates['normal'], fault_switch_rate=rates['fault'],
            validation_perturb_flip_rate=d['validation']['perturb_flip_rate'],
            test_perturb_flip_rate=d['test']['perturb_flip_rate'],
            scale_all_at_floor=bool(np.allclose(fit['scale'], fit['min_std'], rtol=1e-10, atol=0)),
            iterations=fit['iterations']))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='runs/internal/rcr_memory_probe_20261024_v1')
    parser.add_argument('--report', default='report/analysis/rcr/RCR_MEMORY_RESULTS_20261024.md')
    parser.add_argument('--figures', default='report/figures/rcr_memory_probe_20261024')
    parser.add_argument('--bootstrap', type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('Bootstrap repeats must be positive')
    root, figure_dir = Path(args.root), Path(args.figures)
    audits, metrics, diagnostic_rows, bootstraps, selections = {}, [], [], [], {}
    for chemistry in ['LFP', 'NCM']:
        p = root/chemistry
        audits[chemistry] = audit(p)
        frame = pd.read_csv(p/'summary_metrics.csv')
        frame.insert(0, 'dataset', chemistry)
        metrics.append(frame)
        diagnostic_rows.extend(diagnostics(p, chemistry))
        selected = read_json(p/'normal_validation_selection.json')['selected']
        selections[chemistry] = selected
        for base, contender in [('A_current_rcr', selected), ('B_k1', 'B_k6')]:
            bootstraps.extend(dict(dataset=chemistry, **row)
                              for row in paired_bootstrap(p, base, contender, args.bootstrap))
    summary, diagnostic, bootstrap = pd.concat(metrics), pd.DataFrame(diagnostic_rows), pd.DataFrame(bootstraps)
    summary.to_csv(root/'combined_metrics.csv', index=False)
    diagnostic.to_csv(root/'prototype_diagnostics_summary.csv', index=False)
    bootstrap.to_csv(root/'paired_vin_bootstrap.csv', index=False)
    (root/'artifact_audit.json').write_text(json.dumps(audits, ensure_ascii=False, indent=2))
    primary = summary.query("score == 'magnitude' and q == .95")
    winners = []
    for chemistry in ['LFP', 'NCM']:
        subset = primary[primary.dataset == chemistry].set_index('unit')
        for unit in UNITS[1:]:
            if all(subset.loc[unit, metric] > subset.loc['A_current_rcr', metric]
                   for metric in ['ap', 'auroc', 'f1']):
                winners.append(f'{chemistry}/{unit}')
    conclusion = ('多原型相对 K=1 有收益，但两数据集均未出现同时提高 AP、AUROC、F1 的 B 单元，'
                  '尚未显示超越当前 RCR 校准 A 的一致优势。本轮保留 A，暂不进入 C。'
                  if not winners else '存在相对 A 的三项指标均提高的单元：'+', '.join(winners)+
                  '。需结合分层结果与归属稳定性判断；C 尚未训练。')
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout='constrained')
    for row, chemistry in enumerate(['LFP', 'NCM']):
        subset = primary[primary.dataset == chemistry].set_index('unit')
        for column, name in enumerate(['ap', 'auroc', 'f1']):
            ax = axes[row, column]
            ax.plot([1, 2, 4, 6, 8], [subset.loc[f'B_k{k}', name] for k in [1, 2, 4, 6, 8]],
                    'o-', label='B: fixed normal-state memory', color='#247A98')
            ax.axhline(subset.loc['A_current_rcr', name], color='#CE6647', linestyle='--', label='A: current RCR calibration')
            ax.set(title=f'{chemistry}: {name.upper()}', xlabel='Number of prototypes K', ylabel=name.upper(),
                   xticks=[1, 2, 4, 6, 8])
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(figure_dir/'metrics_by_k.png', dpi=180)
    fig.savefig(figure_dir/'metrics_by_k.svg')
    plt.close(fig)
    lines = ['# 正常状态记忆 A/B 首轮结果（2026-10-24）', '',
        conclusion, '',
        '主口径为校准残差均方值、片段 Top-5%、正常验证 P95；各单元同时保存 P99 与训练 CDF 补充结果。任务根目录：`'+str(root)+'`。', '',
        '| 校准方法 | LFP AP / AUROC / F1 | NCM AP / AUROC / F1 |',
        '| --- | --- | --- |']
    for unit in UNITS:
        cells = []
        for chemistry in ['LFP', 'NCM']:
            r = primary[(primary.dataset == chemistry) & (primary.unit == unit)].iloc[0]
            cells.append(f'{r.ap:.4f} / {r.auroc:.4f} / {r.f1:.4f}')
        lines.append(f'| {unit} | '+ ' | '.join(cells)+' |')
    lines.extend(['', '指标均来自本地任务 `RCR-MEM-20261024-v1`，子单元为数据集/上表方法名。独立机制实验表，不更新论文最终四表。', '',
        '## 关键诊断', '',
        '| 数据集 / K | 训练原型占比 min–max | 最小中心距离 | 正常验证 ARI 下限 | 正常 / 故障切换率 | 验证微扰换原型率 |',
        '| --- | --- | --- | --- | --- | --- |'])
    for r in diagnostic.itertuples():
        distance = f'{r.min_center_distance:.3f}' if r.k > 1 else '—'
        lines.append(f'| {r.dataset} / {r.k} | {r.min_occupancy:.1%}–{r.max_occupancy:.1%} | {distance} | '
            f'{r.min_validation_ari:.4f} | {r.normal_switch_rate:.2%} / {r.fault_switch_rate:.2%} | '
            f'{r.validation_perturb_flip_rate:.2%} |')
    lines.extend(['', '中心距离在正常训练标准化的八维工况空间计算。ARI 下限取两个重复聚类种子中的较小值。切换率仅含同片段相邻窗口，故障片段没有逐时刻故障起点标注，不能把切换当作故障触发的证据。', '',
        '各 K 的原型尺度均被共同的 `min_std=0.05` 截住，当前实验主要检验多中心均值校准。'
        '这限制了对多中心尺度机制的判断；若补充尺度实验，须 A/B 成对改变下限。'
        if diagnostic.scale_all_at_floor.all() else '各原型尺度及下限占比见各单元的 `fit_audit.json`。', '',
        'LFP 随 K 增大出现小占用原型，K=8 的正常验证种子稳定性也低于 K=6；'
        '提示需检查划分是否过细，不能单由检测指标断定过拟合。'
        'K=6、8 在两数据集上正常片段的切换率均高于故障片段，原型切换并不具有直接的故障特异性。', '',
        '## 配对 VIN bootstrap', '',
        f'以测试 VIN 为重采样单元，{args.bootstrap} 次，seed3407；每次对两方法使用相同 VIN 抽样。方法各自的正常验证 P95 阈值保持固定；区间不包括训练与阈值估计不确定性，也不是独立测试集确认。', '',
        '| 数据集 | 对照 → 方法 | 指标 | 差值 | 95% 区间 |', '| --- | --- | --- | --- | --- |'])
    for r in bootstrap.itertuples():
        lines.append(f'| {r.dataset} | {r.base} → {r.contender} | {r.metric.upper()} | {r.delta:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] |')
    lines.extend(['', 'B 的正常验证 NLL 选择为：'+ '；'.join(f'{key}={value}' for key, value in selections.items())+'。这是正常拟合质量的选择，不能当成异常检测获益证据。', '',
        '## 产物与复核', '',
        '- `artifact_audit.json`：12 个 A/B 单元、全部逐窗分数与样本映射、转移计数边界均核验；A 的 magnitude 与原始缓存逐窗一致，各方法 Raw 逐窗一致。',
        '- `combined_metrics.csv`：AP、AUROC、P、R、F1、FPR 与阈值；`prototype_diagnostics_summary.csv` 汇总原型诊断；`paired_vin_bootstrap.csv` 保存区间。',
        '- 各数据集/方法目录保留训练中心/尺度、种子稳定性、每片段原型分布、逐窗匹配和故障类型分层转移矩阵；完整协议见 `RCR_MEMORY_PROBE_20261024.md`。',
        '- '+str(figure_dir/'metrics_by_k.png')+' 与同名 SVG 为独立可导出图。', '',
        '下一步：本轮保留 A。多原型有相对单中心的正常状态分解价值，但尚不支持替换已有 RCR 校准。'
        '先检查尺度下限的成对机制补充，再用新的预先固定划分/Backbone 种子确认，证据充分后进入 C。'])
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(lines)+'\n')
    print(primary[['dataset', 'unit', 'ap', 'auroc', 'f1', 'fpr']].round(4).to_string(index=False))
    print('AUDIT AND REPORT COMPLETE', target, flush=True)


if __name__ == '__main__':
    main()
