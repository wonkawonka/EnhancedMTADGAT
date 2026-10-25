"""Audit original-X Encoder memory and evaluate complementarity with RCR."""
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

from src.models.rcr import EmpiricalCDF
from src.models.rcr_feature_memory import EncoderFeatureMemory
from src.runners.run_rcr import atomic_json
from src.runners.run_brand3_encoder_memory import SCORE_NAMES, KS, PARTITIONS, aggregate_vehicle, evaluate, fuse_percentiles
from src.runners.summarize_brand3_memory_probe import METRICS, paired_car_bootstrap


def audit_fold(base, output, fold):
    root = output/'experiments'/f'fold{fold}'
    complete = json.loads((root/'complete.json').read_text())
    mapping = json.loads((root/'mapping.json').read_text())
    source = base/'sources'/f'fold{fold}'/'full'
    backbone = json.loads((source/'backbone_complete.json').read_text())
    if complete['backbone_hash'] != backbone['checkpoint_hash'] or complete['memory_fit_uses_residuals']:
        raise ValueError('Memory used residuals or a different Backbone')
    feature_record = json.loads((output/'features'/f'fold{fold}'/'complete.json').read_text())
    if not feature_record['original_RCR_residuals_reproduced'] or not feature_record['backbone_unchanged']:
        raise ValueError('Encoder extraction failed original RCR replay')
    models = {k: EncoderFeatureMemory.load(root/f'memory_k{k}/memory.npz') for k in KS}
    cdfs = {}
    train_h = np.load(output/'features'/f'fold{fold}'/'train_features.npy', mmap_mode='r')
    for k, memory in models.items():
        reference = np.load(root/f'memory_k{k}/training_distance_cdf.npy')
        np.testing.assert_array_equal(np.sort(memory.score(train_h)), reference)
        np.testing.assert_allclose(memory.feature_mean, train_h.astype('float64').mean(0), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(memory.feature_std, np.maximum(train_h.astype('float64').std(0), 1e-8), rtol=1e-12, atol=1e-15)
        cdfs[k] = EmpiricalCDF().fit(reference)
    vehicles, windows = {}, 0
    for partition in PARTITIONS:
        h = np.load(output/'features'/f'fold{fold}'/f'{partition}_features.npy', mmap_mode='r')
        scores = np.load(root/f'{partition}_window_scores.npy', mmap_mode='r')
        original = np.load(base/'experiments'/f'fold{fold}'/'A_current_rcr'/f'{partition}_window_scores.npy', mmap_mode='r')
        if scores.shape != (len(h), len(SCORE_NAMES)) or not np.isfinite(scores).all():
            raise ValueError('Incomplete or nonfinite window score')
        np.testing.assert_array_equal(scores[:, :2], original[:, 1:3])
        for k, memory in models.items():
            column = SCORE_NAMES.index(f'Memory_k{k}')
            labels, distance, _ = memory.query(h)
            np.testing.assert_array_equal(labels, np.load(root/f'memory_k{k}'/f'{partition}_assignments.npy'))
            np.testing.assert_array_equal(distance, scores[:, column])
            percentile = cdfs[k].transform(distance)
            np.testing.assert_array_equal(percentile, scores[:, column+1])
            np.testing.assert_array_equal(fuse_percentiles(scores[:, 1], percentile), scores[:, column+2])
        windows += len(h)
        if partition != 'train':
            snippets, vehicle = aggregate_vehicle(scores, mapping, partition)
            saved = pd.read_csv(root/f'{partition}_vehicle_scores.csv', dtype={'car': str})
            np.testing.assert_array_equal(vehicle.car, saved.car)
            np.testing.assert_array_equal(vehicle.label, saved.label)
            np.testing.assert_allclose(vehicle[SCORE_NAMES], saved[SCORE_NAMES], rtol=1e-12, atol=1e-15)
            mapped = pd.read_csv(root/f'{partition}_snippet_scores.csv', dtype={'car': str})
            np.testing.assert_array_equal(mapped.sample_id, snippets.sample_id)
            vehicles[partition] = vehicle
    rows, thresholds = evaluate(vehicles['validation'], vehicles['test'], fold)
    saved = pd.read_csv(root/'metrics.csv')
    fresh = pd.DataFrame(rows)
    np.testing.assert_array_equal(saved.method, fresh.method)
    np.testing.assert_allclose(saved[['threshold']+METRICS], fresh[['threshold']+METRICS], rtol=1e-12, atol=1e-15)
    result = dict(fold=fold, complete=True, backbone_hash=backbone['checkpoint_hash'],
        hidden_feature_windows=windows, RCR_unchanged=True, memory_fit_normal_H_only=True,
        replayed_score_columns=SCORE_NAMES, all_memory_distances_and_fixed_fusions_recomputed=True,
        sample_and_vehicle_mappings_verified=True, vehicle_metrics_recomputed=True)
    atomic_json(root/'artifact_audit.json', result)
    return fresh, vehicles, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='runs/internal/rcr_memory_brand3_20261025_v1')
    parser.add_argument('--root', default='runs/internal/rcr_encoder_memory_brand3_20261025_v1')
    parser.add_argument('--report', default='report/analysis/rcr/BRAND3_ENCODER_MEMORY_RESULTS_20261025.md')
    parser.add_argument('--bootstrap', type=int, default=500)
    args = parser.parse_args()
    base, output = Path(args.base), Path(args.root)
    metrics, vehicles, audits, overlaps, diagnostic_rows, correlations = [], {}, {}, [], [], []
    for fold in range(5):
        frame, scored, audits[fold] = audit_fold(base, output, fold)
        metrics.append(frame)
        for method in ['RCR', 'RCR_CDF', 'Memory_k6', 'Fusion_k6']:
            test = scored['test'][['car', 'label', method]].rename(columns={method: 'score'})
            vehicles[fold, method] = dict(test=test, threshold=float(scored['validation'][method].quantile(.95)))
        test = scored['test']
        threshold = {m: float(scored['validation'][m].quantile(.95)) for m in ['RCR', 'Memory_k6', 'Fusion_k6']}
        for row in test.itertuples():
            rcr, memory, fusion = (getattr(row, m) >= threshold[m] for m in ['RCR', 'Memory_k6', 'Fusion_k6'])
            overlaps.append(dict(fold=fold, car=row.car, label=row.label,
                RCR_detected=bool(rcr), Memory_detected=bool(memory), Fusion_detected=bool(fusion),
                category='both' if rcr and memory else 'RCR_only' if rcr else 'Memory_only' if memory else 'neither'))
        correlations.append(dict(fold=fold,
            vehicle_RCR_memory_spearman=float(test.RCR.corr(test.Memory_k6, method='spearman'))))
        root = output/'experiments'/f'fold{fold}'
        for k in KS:
            directory = root/f'memory_k{k}'
            fit = json.loads((directory/'fit_audit.json').read_text())
            d = json.loads((directory/'diagnostics.json').read_text())['partitions']
            stable = pd.read_csv(directory/'matching_stability.csv').query("partition == 'validation'")
            center = pd.read_csv(directory/'center_distances.csv', index_col=0).to_numpy()
            counts = np.asarray(fit['prototype_windows'])
            diagnostic_rows.append(dict(fold=fold, k=k, min_occupancy=float(counts.min()/counts.sum()),
                max_occupancy=float(counts.max()/counts.sum()),
                min_center_distance=float(center[np.triu_indices(k, 1)].min()) if k > 1 else np.nan,
                validation_ari_min=float(stable.ari.min()), validation_agreement_min=float(stable.aligned_agreement.min()),
                validation_perturb_flip_rate=d['validation']['perturb_flip_rate'],
                normal_far_fraction=d['test']['class_summary']['0']['far_fraction'],
                fault_far_fraction=d['test']['class_summary']['1']['far_fraction'],
                memory_CDF_test_saturation=json.loads((root/'cdf_saturation.json').read_text())['test'][f'Memory_CDF_k{k}']))
    all_metrics = pd.concat(metrics, ignore_index=True)
    primary = all_metrics[all_metrics.q == .95]
    macro = primary.groupby('method')[METRICS].agg(['mean', 'std']).reindex(SCORE_NAMES)
    bootstrap = paired_car_bootstrap(vehicles,
        [('RCR', 'Fusion_k6'), ('RCR_CDF', 'Fusion_k6'), ('RCR', 'Memory_k6')], args.bootstrap)
    all_metrics.to_csv(output/'all_fold_metrics.csv', index=False)
    macro.to_csv(output/'macro_fold_summary.csv')
    bootstrap.to_csv(output/'paired_car_bootstrap.csv', index=False)
    diagnostic = pd.DataFrame(diagnostic_rows)
    diagnostic.to_csv(output/'prototype_diagnostics_summary.csv', index=False)
    overlap = pd.DataFrame(overlaps)
    overlap.to_csv(output/'vehicle_prediction_overlap.csv', index=False)
    overlap.groupby(['label', 'category']).size().rename('vehicle_fold_records').to_csv(output/'prediction_overlap_counts.csv')
    pd.DataFrame(correlations).to_csv(output/'branch_vehicle_correlations.csv', index=False)
    atomic_json(output/'artifact_audit.json', audits)
    figdir = Path('report/figures/brand3_encoder_memory_20261025')
    figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
    names, display = ['RCR', 'Memory_k6', 'Fusion_k6', 'RCR_CDF'], ['RCR', 'Memory K=6', 'Fusion K=6', 'RCR CDF']
    for ax, metric in zip(axes, ['ap', 'auroc', 'f1']):
        ax.bar(np.arange(4), [macro.loc[m, (metric, 'mean')] for m in names],
            yerr=[macro.loc[m, (metric, 'std')] for m in names], capsize=3,
            color=['#CE6647', '#247A98', '#639B66', '#B8B3A9'])
        ax.set(title=metric.upper(), xticks=np.arange(4), xticklabels=display, ylim=(0, 1.05), ylabel='Fivefold mean (bars: fold SD)')
        ax.tick_params(axis='x', labelrotation=20)
        ax.grid(axis='y', alpha=.2)
    fig.savefig(figdir/'primary_comparison.png', dpi=180)
    fig.savefig(figdir/'primary_comparison.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
    for ax, metric in zip(axes, ['ap', 'auroc', 'f1']):
        for prefix, label, color in [('Memory', 'Memory alone', '#247A98'), ('Fusion', 'Fixed fusion', '#639B66')]:
            ax.plot(KS, [macro.loc[f'{prefix}_k{k}', (metric, 'mean')] for k in KS], 'o-', label=label, color=color)
        ax.axhline(macro.loc['RCR', (metric, 'mean')], color='#CE6647', linestyle='--', label='Unchanged RCR')
        ax.axhline(macro.loc['RCR_CDF', (metric, 'mean')], color='#B8B3A9', linestyle=':', label='RCR CDF control')
        ax.set(title=metric.upper(), xlabel='H-memory K (primary: K=6)', xticks=KS)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(figdir/'k_sensitivity.png', dpi=180)
    fig.savefig(figdir/'k_sensitivity.svg')
    plt.close(fig)
    delta = {m: float(macro.loc['Fusion_k6', (m, 'mean')]-macro.loc['RCR', (m, 'mean')]) for m in ['ap', 'auroc', 'f1']}
    negative_intervals = bootstrap[(bootstrap.base == 'RCR') & (bootstrap.contender == 'Fusion_k6')].ci_high.lt(0).all()
    conclusion = ('当前 H 时间均值池化＋固定原型＋等权融合未显示互补收益；K=6 的三项差值区间均为负。'
                  if negative_intervals else '当前设置的互补性须结合三项指标区间及 RCR-CDF 对照判断，不能仅看单个 K 的点估计。')
    lines = ['# Brand3：Encoder 隐特征 Memory 与 RCR 互补性（2026-10-25）', '',
        conclusion+'已完成原始 X → 同一冻结 Encoder → 隐特征 H → 正常 Memory 的五折验证。Memory 不读残差或校准后残差，RCR 逐窗口和车辆分数均复算确认未改变。', '',
        '预先固定 K=6 的融合相对原样 RCR：'+ '，'.join(f'{m.upper()} {v:+.4f}' for m, v in delta.items())+'。是否支持稳定互补性须同时看下列配对车辆区间及 RCR-CDF 对照，不能只选测试集最好的 K。', '',
        '## 三条路线与对照', '',
        '| 方法 | AP | AUROC | P | R | F1 | FPR |', '| --- | --- | --- | --- | --- | --- | --- |']
    for method in names:
        label = dict(RCR='原样 RCR', Memory_k6='Encoder H Memory（K=6）', Fusion_k6='Memory＋RCR 固定融合（K=6）', RCR_CDF='RCR-CDF 额外对照')[method]
        lines.append('| '+label+' | '+' | '.join(f'{macro.loc[method, (m, "mean")]:.4f} ± {macro.loc[method, (m, "std")]:.4f}' for m in METRICS)+' |')
    lines.extend(['', '± 为五折标准差。阈值使用各方法正常验证车辆 P95；P99 同时保存。AP/AUROC 不依赖阈值。原样 RCR 是本轮 A 的工况均值/尺度校准幅值分数，并非另训关系头后的 RCR-Net 融合分数。', '',
        '## 固定协议', '',
        '- H：原始七变量训练折 MinMax 时序输入经过原有 Backbone，取时间 Encoder 输出、Decoder 之前的 `[B,127,32]`，对时间求均值得 `[B,32]`。不使用残差、条件残差或响应残差构建 Memory。',
        '- Memory：仅正常训练 H 估计标准化与 K-means 原型；最近原型的平均平方距离为分数。K=6 为预先指定主比较，K=1/2/4/8 为完整敏感性报告。记忆推理时冻结。',
        '- RCR：直接复用前阶段 A 的已有窗口分数，既不改校准也不重训 Backbone。H 提取时额外核验 Backbone hash、全量重构残差与工况特征重放。',
        '- 融合：窗口级 `0.5*normal_train_CDF(RCR)+0.5*normal_train_CDF(Memory)`，然后片段/车辆 Top-5%，正常验证车辆 P95/P99。没有测试选 K、选权重或按测试最优阈值调参。',
        '- RCR-CDF：只对原有 RCR 做相同训练 CDF 变换，用于排除 CDF 与聚合顺序本身带来的变化。若融合仅优于原样 RCR，却未优于此对照，不能把变化归因于 Memory 互补信息。',
        '- 这是 MEMTO 记忆表示思想的初步验证，不是完整 MEMTO 记忆约束重构复现，也不预先规定未来模型必须把 Memory 放在这里。', '',
        '## 全部 K（不做测试集最优筛选）', '',
        '| K | Memory AP / AUROC / F1 | 固定融合 AP / AUROC / F1 |', '| --- | --- | --- |'])
    for k in KS:
        cells = [' / '.join(f'{macro.loc[f"{prefix}_k{k}", (m, "mean")]:.4f}' for m in ['ap', 'auroc', 'f1']) for prefix in ['Memory', 'Fusion']]
        lines.append(f'| {k} | '+' | '.join(cells)+' |')
    lines.extend(['', '## 配对车辆不确定性', '',
        f'按唯一车辆和标签分层重采样 {args.bootstrap} 次（seed3407），同一辆车在不同折复用同一权重；五折等权。阈值保持各方法正常验证 P95。故障车辆跨折重复，不当作独立故障样本。区间不涵盖 Backbone 或验证阈值估计的不确定性。', '',
        '| 对照 → 方法 | 指标 | 差值 | 95% 区间 |', '| --- | --- | --- | --- |'])
    for row in bootstrap.itertuples():
        lines.append(f'| {row.base} → {row.contender} | {row.metric.upper()} | {row.delta:+.4f} | [{row.ci_low:+.4f}, {row.ci_high:+.4f}] |')
    lines.extend(['', '## 原型、稳定性与误差互补', '',
        '| K | 最小训练占用（跨折最小） | 验证 ARI 下限（跨折/种子最小） | 验证微扰换原型率（跨折均值） | 测试 Memory-CDF 饱和率（跨折均值） |',
        '| --- | --- | --- | --- | --- |'])
    for k in KS:
        d = diagnostic[diagnostic.k == k]
        lines.append(f'| {k} | {d.min_occupancy.min():.2%} | {d.validation_ari_min.min():.4f} | '
            f'{d.validation_perturb_flip_rate.mean():.2%} | {d.memory_CDF_test_saturation.mean():.2%} |')
    counts = overlap.groupby(['label', 'category']).size()
    lines.extend(['', 'K=6、正常验证 P95 的分支判定重叠（单位为车辆-折记录，不是独立车辆数）：', '',
        '| 车辆标签 | 两者都报异常 | 仅 RCR | 仅 Memory | 两者均未报异常 |', '| --- | --- | --- | --- | --- |'])
    for label in [0, 1]:
        lines.append('| '+('正常' if label == 0 else '故障')+' | '+ ' | '.join(str(counts.get((label, cat), 0)) for cat in ['both', 'RCR_only', 'Memory_only', 'neither'])+' |')
    memory_only_faults = int(counts.get((1, 'Memory_only'), 0))
    lines.extend(['', f'该工作点 Memory 额外检出 {memory_only_faults} 条 RCR 漏检的故障车辆-折记录。'
        '固定融合是否改善检测仍以融合指标与区间为准；不能仅凭低分支相关性宣称有效。'
        '每片段一窗且片段时间顺序未经确认，故障原型时间切换率不适用。保留各车辆的原型占用、熵及正常/故障样本分布。', '',
        '本次结论仅针对当前冻结 Encoder 的时间均值池化 H、固定正常原型与预设等权融合。'
        '没有记忆约束的联合表示训练，不能据此判定完整 MEMTO 无效；也不根据这次测试的最优 K 或权重追调正式结果。', '',
        '## 产物', '',
        f'- 任务 `RCR-HMEM-BRAND3-20261025-v1`：`{output}`。五折共同使用前阶段的五个 seed3407 Backbone；重复聚类种子只衡量聚类稳定性，不代表完整流程多 seed。',
        '- `artifact_audit.json`：全量原型距离、CDF、融合分数、映射及车辆指标复算；RCR 逐窗分数与前阶段完全一致。',
        '- `all_fold_metrics.csv`、`macro_fold_summary.csv`、`paired_car_bootstrap.csv`、`prototype_diagnostics_summary.csv`、`vehicle_prediction_overlap.csv` 保存指标、诊断与分支错误重叠。',
        f'- 图：`{figdir}/primary_comparison.png`、`k_sensitivity.png` 及各自 SVG。论文最终四表未改。'])
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(lines)+'\n')
    atomic_json(output/'complete.json', dict(task='RCR-HMEM-BRAND3-20261025-v1', folds=list(range(5)),
        memory_uses_residuals=False, frozen_RCR_unchanged=True, primary_k=6, fixed_alpha=.5,
        methods=SCORE_NAMES, artifact_audit_complete=True, no_test_tuning=True))
    print(macro.loc[names].round(4).to_string())
    print('ENCODER MEMORY REPORT COMPLETE', target, flush=True)


if __name__ == '__main__':
    main()
