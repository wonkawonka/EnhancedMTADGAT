"""Summarize only completed Brand3 A/B folds; paired bootstrap respects cars."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rcr-memory-brand3-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.runners.run_brand3_memory_probe import KS, UNITS, audit_fold
from src.runners.run_rcr import atomic_json


METRICS = ['ap', 'auroc', 'precision', 'recall', 'f1', 'fpr']


def paired_car_bootstrap(vehicles, pairs, repeats=500):
    """One stratified resampling weight per car, shared across folds/methods."""
    unique = pd.concat([entry['test'][['car', 'label']] for entry in vehicles.values()]).drop_duplicates()
    if not unique.groupby('car').label.nunique().eq(1).all():
        raise ValueError('Conflicting labels across folds')
    labels = unique.groupby('car').label.first().sort_index()
    groups = [labels[labels == y].index.to_numpy() for y in [0, 1]]
    folds = sorted({fold for fold, unit in vehicles})
    arrays = {}
    for fold, unit in vehicles:
        entry = vehicles[fold, unit]
        frame = entry['test'].set_index('car').sort_index()
        arrays[fold, unit] = (frame.index, frame.label.to_numpy(), frame.score.to_numpy(), entry['threshold'])

    def calculate(y, score, threshold, weight):
        pred = score >= threshold
        tp = weight[pred & (y == 1)].sum()
        fp = weight[pred & (y == 0)].sum()
        fn = weight[~pred & (y == 1)].sum()
        return np.array([average_precision_score(y, score, sample_weight=weight),
            roc_auc_score(y, score, sample_weight=weight), 2*tp/max(1e-30, 2*tp+fp+fn)])

    def differences(base, contender, counts):
        result = []
        for fold in folds:
            ai, ay, a, at = arrays[fold, base]
            bi, by, b, bt = arrays[fold, contender]
            np.testing.assert_array_equal(ai, bi)
            np.testing.assert_array_equal(ay, by)
            weight = counts.loc[ai].to_numpy()
            if any(weight[ay == y].sum() == 0 for y in [0, 1]):
                return None
            result.append(calculate(ay, b, bt, weight)-calculate(ay, a, at, weight))
        return np.mean(result, axis=0)

    observed = {pair: differences(*pair, pd.Series(1., index=labels.index)) for pair in pairs}
    draws = {pair: [] for pair in pairs}
    rng = np.random.default_rng(3407)
    for _ in range(repeats):
        counts = pd.Series(0., index=labels.index)
        for group in groups:
            sampled, count = np.unique(rng.choice(group, len(group), replace=True), return_counts=True)
            counts.loc[sampled] = count
        for pair in pairs:
            delta = differences(*pair, counts)
            if delta is not None:
                draws[pair].append(delta)
    rows = []
    for base, contender in pairs:
        low, high = np.quantile(draws[base, contender], [.025, .975], axis=0)
        for j, metric in enumerate(['ap', 'auroc', 'f1']):
            rows.append(dict(base=base, contender=contender, metric=metric,
                delta=float(observed[base, contender][j]), ci_low=float(low[j]), ci_high=float(high[j]),
                valid_draws=len(draws[base, contender]), requested_draws=repeats,
                unique_vehicles=len(labels), unique_fault_vehicles=int(labels.sum())))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs/internal/rcr_memory_brand3_20261025_v1')
    parser.add_argument('--report', default='report/analysis/rcr/BRAND3_MEMORY_CALIBRATION_20261025.md')
    parser.add_argument('--bootstrap', type=int, default=500)
    args = parser.parse_args()
    base = Path(args.root)
    all_rows, diagnostics, vehicles, audits = [], [], {}, {}
    for fold in range(5):
        source, output = base/'sources'/f'fold{fold}'/'full', base/'experiments'/f'fold{fold}'
        if not (output/'complete.json').exists():
            raise ValueError(f'Fold {fold} is incomplete; no fivefold report')
        mapping = json.loads((source/'mapping.json').read_text())
        audits[fold] = audit_fold(source, output, mapping)
        rows = pd.read_csv(output/'summary_metrics.csv')
        all_rows.append(rows)
        for unit in UNITS:
            target = output/unit
            val = pd.read_csv(target/'validation_vehicle_scores.csv', dtype={'car': str})
            test = pd.read_csv(target/'test_vehicle_scores.csv', dtype={'car': str})
            vehicles[fold, unit] = dict(test=test[['car', 'label', 'magnitude']].rename(columns={'magnitude': 'score'}),
                threshold=float(val.magnitude.quantile(.95)))
        for k in KS:
            target = output/f'B_k{k}'
            fit = json.loads((target/'fit_audit.json').read_text())
            d = json.loads((target/'diagnostics.json').read_text())['partitions']
            stability = pd.read_csv(target/'matching_stability.csv').query("partition == 'validation'")
            centers = pd.read_csv(target/'center_distances.csv', index_col=0).to_numpy()
            distances = centers[np.triu_indices(k, 1)]
            counts = np.asarray(fit['prototype_windows'])
            diagnostics.append(dict(fold=fold, k=k, min_occupancy=float(counts.min()/counts.sum()),
                max_occupancy=float(counts.max()/counts.sum()),
                min_center_distance=float(distances.min()) if k > 1 else np.nan,
                validation_ari_min=float(stability.ari.min()),
                validation_agreement_min=float(stability.aligned_agreement.min()),
                validation_perturb_flip_rate=d['validation']['perturb_flip_rate'],
                test_perturb_flip_rate=d['test']['perturb_flip_rate'],
                normal_far_fraction=d['test']['class_summary']['0']['far_fraction'],
                fault_far_fraction=d['test']['class_summary']['1']['far_fraction'],
                scale_all_at_floor=bool(np.allclose(fit['scale'], fit['min_std'], rtol=1e-10)),
                temporal_switch_rate=None))
    metrics = pd.concat(all_rows, ignore_index=True)
    primary = metrics.query("score == 'magnitude' and q == .95")
    macro = primary.groupby('unit')[METRICS].agg(['mean', 'std']).reindex(UNITS)
    diagnostic = pd.DataFrame(diagnostics)
    bootstrap = paired_car_bootstrap(vehicles, [('A_current_rcr', 'B_k6'), ('B_k1', 'B_k6')], args.bootstrap)
    metrics.to_csv(base/'all_fold_metrics.csv', index=False)
    macro.to_csv(base/'macro_fold_summary.csv')
    diagnostic.to_csv(base/'prototype_diagnostics_summary.csv', index=False)
    bootstrap.to_csv(base/'paired_car_bootstrap.csv', index=False)
    atomic_json(base/'artifact_audit.json', audits)
    figdir = Path('report/figures/brand3_memory_calibration_20261025')
    figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout='constrained')
    for ax, metric in zip(axes, ['ap', 'auroc', 'f1']):
        ax.errorbar(KS, [macro.loc[f'B_k{k}', (metric, 'mean')] for k in KS],
            yerr=[macro.loc[f'B_k{k}', (metric, 'std')] for k in KS], marker='o', capsize=3,
            color='#247A98', label='B: fixed normal-state calibration')
        ax.axhline(macro.loc['A_current_rcr', (metric, 'mean')], color='#CE6647', linestyle='--', label='A: current RCR calibration')
        ax.set(title=f'Brand3: {metric.upper()}', xlabel='K (bars: fold SD)', ylabel=metric.upper(), xticks=KS)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(figdir/'metrics_by_k.png', dpi=180)
    fig.savefig(figdir/'metrics_by_k.svg')
    plt.close(fig)
    lines = ['# Brand3 五折多原型残差校准（2026-10-25）', '',
        '五折共 30 个 A/B 单元已完成并核验。该阶段沿用前一轮残差校准问题；随后另做用户指定的 Encoder 隐特征 Memory 互补性验证，两者不混为同一机制。', '',
        '每折一个新本地 RCR Backbone（seed3407），A/B 共享其冻结残差；旧远程归档缺少 Backbone 与残差，因此本结果不声称复现历史 checkpoint。车辆隔离五折、训练正常 MinMax、127 点一片段一窗、车辆 Top-5%、正常验证车辆 P95/P99。以下为 P95 主幅值口径，AP/AUROC 与阈值无关。', '',
        '| 校准方法 | AP | AUROC | P | R | F1 | FPR |', '| --- | --- | --- | --- | --- | --- | --- |']
    for unit in UNITS:
        lines.append('| '+unit+' | '+' | '.join(f'{macro.loc[unit, (m, "mean")]:.4f} ± {macro.loc[unit, (m, "std")]:.4f}' for m in METRICS)+' |')
    lines.extend(['', '± 为五折标准差，不表示五个独立故障队列；重复出现的故障车辆不能作为独立样本。完整逐折、P99 和 CDF 补充值在任务输出中。', '',
        '## 原型诊断', '',
        '| K | 最小占用（五折最小） | 原型最小距离（五折最小） | 验证 ARI（五折/重复种子最小） | 验证微扰换原型率（五折均值） | 测试正常/故障远离原型比例（五折均值） |',
        '| --- | --- | --- | --- | --- | --- |'])
    for k in KS:
        d = diagnostic[diagnostic.k == k]
        distance = f'{d.min_center_distance.min():.3f}' if k > 1 else '—'
        lines.append(f'| {k} | {d.min_occupancy.min():.2%} | {distance} | {d.validation_ari_min.min():.4f} | '
            f'{d.validation_perturb_flip_rate.mean():.2%} | {d.normal_far_fraction.mean():.2%} / {d.fault_far_fraction.mean():.2%} |')
    lines.extend(['', '远离原型：最近中心距离超过该原型正常训练距离 P95，仅作诊断，不参与异常分数或阈值选择。中心距离使用正常训练标准化工况空间。', '',
        '**片段内切换率不适用**：每片段只有一窗，片段间没有确认时间顺序，不伪造 0% 切换率。保存车辆内不同片段的原型占用、熵及微扰稳定性，不能将其称作故障触发切换。', '',
        '## 配对车辆 bootstrap', '',
        f'全局按车辆标签分层重采样 {args.bootstrap} 次（seed3407），同一车辆的权重在各折和方法中保持一致；比较五折等权平均，阈值固定为各方法正常验证 P95。区间不包含训练与阈值估计不确定性。', '',
        '| 对照 → 方法 | 指标 | 差值 | 95% 区间 |', '| --- | --- | --- | --- |'])
    for r in bootstrap.itertuples():
        lines.append(f'| {r.base} → {r.contender} | {r.metric.upper()} | {r.delta:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] |')
    lines.extend(['', '## 产物与下一阶段', '',
        f'- 任务：`RCR-MEM-BRAND3-20261025-v1`，目录 `{base}`。`artifact_audit.json` 保存 30 个单元的逐窗分数、原型归属与车辆指标复算结果。',
        '- `all_fold_metrics.csv`、`macro_fold_summary.csv`、`prototype_diagnostics_summary.csv`、`paired_car_bootstrap.csv` 保存全部数值；各折目录含模型、分数、阈值、片段与车辆映射。',
        f'- 图：`{figdir}/metrics_by_k.png` 与 SVG。论文最终四表未改。',
        '- 下一阶段固定为原始 X → 同一个冻结 Encoder → 隐特征 H → 正常 Memory；对比 Memory、原样 RCR 与固定融合。不是再次对残差做 K-means，也不是完整 MEMTO 复现。'])
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(lines)+'\n')
    atomic_json(base/'complete.json', dict(folds=list(range(5)), units=30, artifact_audit_complete=True,
        next_stage='frozen_encoder_hidden_memory', residual_memory_phase_c_started=False))
    print(macro.round(4).to_string())
    print('BRAND3 CALIBRATION REPORT COMPLETE', target, flush=True)


if __name__ == '__main__':
    main()
