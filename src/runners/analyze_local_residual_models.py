"""Audit paired local-calibration/conditional-relation tests and export evidence."""
import argparse
import json
import os
import hashlib
import re
import shutil
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rcr-local-models-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.evaluate_rcr import metrics
from scripts.analyze_rcr_calibration_distributions import paired_vin_bootstrap
from src.models.local_residual_models import ReliableConditionalCalibration, ConditionalLOORelation
from src.models.rcr import LinearResidualCalibration, EmpiricalCDF
from src.models.rcr_fitting import slices
from src.runners.run_rcr import atomic_json
from src.runners.run_rcr_calibration_sensitivity import aggregate
from src.runners.run_local_residual_models import (NAMES, PRIMARY, PARTITIONS, SPECS, spec_name, source_for, load_context, rank_level)
from src.runners.summarize_brand3_memory_probe import paired_car_bootstrap, METRICS


UNITS = ['LFP', 'NCM']+[f'Brand3_fold{i}' for i in range(5)]
PAIRS = [('A', 'Full_selected'), ('Full_selected', 'M1'), ('A', 'M1'),
         ('M1', 'Relation_conditional'), ('Relation_unconditional', 'Relation_conditional'),
         ('Marginal_diag', 'Relation_conditional'), ('M1', 'M2'), ('M1_CDF', 'M2')]


def read_table(path, **kwargs):
    # Adjacent CDF percentiles can collapse with pandas' default float parser,
    # changing ties and hence AP. Preserve the original double values exactly.
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def audit_local_table(output, brand):
    test = read_table(output/'test_snippet_scores.csv', dtype={'vin': str})
    thresholds = json.loads((output/'thresholds.json').read_text())
    original = read_table(output/'local_metrics.csv')
    rows = []
    for regime, frame in test.groupby('operating_regime'):
        normal, faults = frame[frame.sample_label == 0], frame[frame.sample_label == 1]
        for fault_type, fault in faults.groupby('fault_type'):
            if len(normal) < 10 or len(fault) < 10:
                continue
            scored = rank_level(pd.concat([normal, fault]), brand)
            if scored.groupby('sample_label').vin.nunique().min() < (3 if brand else 1):
                continue
            common = dict(group=regime+'/'+str(fault_type), regime=regime, fault_type=fault_type,
                normal_snippets=len(normal), fault_snippets=len(fault),
                normal_vins=normal.vin.nunique(), fault_vins=fault.vin.nunique())
            for name in PRIMARY:
                rows.append(dict(**common, method=name,
                    **metrics(scored.sample_label, scored[name], thresholds[name]['0.95'])))
    audited = pd.DataFrame(rows)
    audited.to_csv(output/'local_metrics_audited.csv', index=False)
    left, right = (f.set_index(['group', 'method']).sort_index() for f in [original, audited])
    if not left.index.equals(right.index):
        raise ValueError('Local cohort changed during roundtrip audit')
    difference = (left[METRICS]-right[METRICS]).abs()
    return dict(local_groups=audited.group.nunique(),
        max_original_CSV_parse_metric_difference=float(difference.to_numpy().max(initial=0)),
        authoritative_local_table='local_metrics_audited.csv')


def audit_unit(root, unit, batch=4096):
    output = root/unit
    record = json.loads((output/'complete.json').read_text())
    if record['score_names'] != NAMES:
        raise ValueError('Incomplete method matrix')
    source, reference = source_for(unit)
    mapping, context, brand = load_context(source, output)
    cal = ReliableConditionalCalibration.load(output/'M1_calibration.npz')
    models = {name: ConditionalLOORelation.load(output/f'relation_{name}.npz') for name in ['conditional', 'unconditional']}
    folds = np.load(output/'normal_training_crossfit_folds.npy')
    codes = context['train']['vehicles']
    if any(len(np.unique(folds[codes == code])) != 1 for code in np.unique(codes)):
        raise ValueError('A normal VIN spans reliability folds')
    selection = json.loads((output/'scale_selection.json').read_text())
    winner = min((n for n in selection['normal_validation_nll'] if n.startswith('Full_')),
                 key=selection['normal_validation_nll'].get)
    if selection['test_labels_used'] or not selection['matched_for_reliable']:
        raise ValueError('Invalid scale selection')
    if cal.scale_spec != selection['selected_scale']:
        raise ValueError('M1 does not use the preselected paired scale')
    checked = 0
    for partition in PARTITIONS:
        c = context[partition]
        scores = np.load(output/f'{partition}_window_scores.npy', mmap_mode='r')
        original = np.load(reference/f'{partition}_window_scores.npy' if brand else reference/partition/'scores.npy', mmap_mode='r')
        if scores.shape != (len(c['q']), len(NAMES)):
            raise ValueError('Score shape mismatch')
        np.testing.assert_array_equal(scores[:, :2], original[:, :2])
        for ix in slices(len(scores), batch):
            z = cal.transform(c['q'][ix], c['residual'][ix], c['groups'][ix])
            np.testing.assert_allclose(scores[ix, NAMES.index('M1')], np.square(z).mean((1, 2)), rtol=1e-9, atol=1e-12)
            for kind, name in [('conditional', 'Relation_conditional'), ('unconditional', 'Relation_unconditional')]:
                model = models[kind]
                direct = (np.square(z-model.predict(c['q'][ix], z)).mean(1)/model.variances).mean(1)
                np.testing.assert_allclose(scores[ix, NAMES.index(name)], direct, rtol=1e-8, atol=1e-9)
            if not np.isfinite(scores[ix]).all():
                raise ValueError('Nonfinite score')
        with np.load(output/'normal_train_CDFs.npz') as data:
            pm = EmpiricalCDF().fit(data['M1']).transform(scores[:, NAMES.index('M1')])
            pr = EmpiricalCDF().fit(data['Relation_conditional']).transform(scores[:, NAMES.index('Relation_conditional')])
            np.testing.assert_array_equal(scores[:, NAMES.index('M2')], .5*(pm+pr))
        checked += len(scores)
        if partition != 'train':
            snippets = aggregate(scores, c['refs'], mapping['sample_ids'][partition], NAMES)
            snippets = c['frame'].merge(snippets, on='sample_id', validate='one_to_one')
            recalculated = rank_level(snippets, brand)
            saved = read_table(output/f'{partition}_scores.csv', dtype={'vin': str})
            np.testing.assert_array_equal(saved.vin, recalculated.vin)
            np.testing.assert_allclose(saved[NAMES], recalculated[NAMES], rtol=1e-12, atol=1e-14)
    frames = {p: read_table(output/f'{p}_scores.csv', dtype={'vin': str}) for p in ['validation', 'test']}
    for row in json.loads((output/'metrics.json').read_text()):
        threshold = float(frames['validation'][row['method']].quantile(row['q']))
        np.testing.assert_allclose(threshold, row['threshold'], rtol=1e-12, atol=1e-14)
        for key, value in metrics(frames['test'].sample_label, frames['test'][row['method']], threshold).items():
            np.testing.assert_allclose(value, row[key], rtol=1e-12, atol=1e-14)
    local_audit = audit_local_table(output, brand)
    result = dict(unit=unit, full_primary_score_windows_replayed=checked, baseline_scores_exact=True,
        relation_scores_replayed_by_explicit_target_excluded_predictions=True,
        VIN_crossfit_verified=True, normal_validation_scale_winner=winner, local=local_audit, complete=True)
    atomic_json(output/'artifact_audit.json', result)
    del context
    return result, frames


def residual_distributions(root, unit):
    output = root/unit
    frame = read_table(output/'test_snippet_scores.csv', dtype={'vin': str})
    source, reference = source_for(unit)
    mapping = json.loads((source/'mapping.json').read_text())
    base = LinearResidualCalibration.load(source/'lite_continuous/calibration.npz')
    cal = ReliableConditionalCalibration.load(output/'M1_calibration.npz')
    q = np.load(source/'residuals/test/condition.npy', mmap_mode='r')
    r = np.load(source/'residuals/test/residual.npy', mmap_mode='r')
    refs = np.load(source/'residuals/test/references.npy', mmap_mode='r')
    edges = json.loads((output/'regime_edges.json').read_text())
    cycles = sorted(edges)
    f = frame.set_index('sample_id').loc[mapping['sample_ids']['test']]
    boundaries = np.r_[0, np.flatnonzero(np.diff(refs[:, 0]))+1, len(refs)]
    rng = np.random.default_rng(3407)
    arrays = {name: [] for name in ['raw', 'full_centered', 'reliable_centered', 'M1_standardized']}
    strengths = []
    for index, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        selected = rng.choice((hi-lo)*r.shape[1], 16, replace=False)
        windows, time = lo+selected//r.shape[1], selected%r.shape[1]
        qr = np.asarray(q[windows], dtype=np.float64)
        raw = np.asarray(r[windows, time], dtype=np.float64)
        cycle = f.iloc[index].cycle_kind
        groups = (cycles.index(cycle)*9+np.searchsorted(edges[cycle]['current_mean'], qr[:, 0], side='right')*3+
                  np.searchsorted(edges[cycle]['soc_mean'], qr[:, 4], side='right'))
        full_mean = base.parameters(qr)[0]
        mu, sigma = cal.parameters(qr, groups)
        arrays['raw'].append(raw)
        arrays['full_centered'].append(raw-full_mean)
        arrays['reliable_centered'].append(raw-mu)
        arrays['M1_standardized'].append((raw-mu)/sigma)
        strengths.append(float(cal.strength(qr, groups).mean()))
    arrays = {key: np.stack(values) for key, values in arrays.items()}
    np.savez_compressed(output/'paired_residual_distribution_points.npz',
        sample_ids=np.asarray(f.index), features=np.asarray(mapping['feature_columns']), **arrays)
    rows = []
    labels = f.sample_label.to_numpy()
    for label in [0, 1]:
        for name, values in arrays.items():
            for j, feature in enumerate(mapping['feature_columns']):
                quantiles = np.quantile(np.abs(values[labels == label, :, j]), [.05, .25, .5, .75, .95, .99])
                rows.append(dict(unit=unit, label=label, representation=name, feature=feature,
                    units='standardized residual' if name == 'M1_standardized' else 'original residual units',
                    **dict(zip(['p05', 'p25', 'median', 'p75', 'p95', 'p99'], quantiles))))
    pd.DataFrame(rows).to_csv(output/'absolute_residual_quantiles.csv', index=False)
    f = f.copy(); f['sampled_mean_gate'] = strengths
    for name in ['Full_mean', 'Reliable_mean']:
        f[name+'_removed_energy_fraction'] = 1-f[name]/np.maximum(f.Raw, 1e-30)
    f[['vin', 'sample_label', 'fault_type', 'operating_regime', 'sampled_mean_gate',
       'Full_mean_removed_energy_fraction', 'Reliable_mean_removed_energy_fraction']].to_csv(output/'energy_retention.csv')
    return pd.DataFrame(rows)


def ncm_local_analysis(root, repeats):
    output = root/'NCM'
    local = read_table(output/'local_metrics_audited.csv')
    pivot = local.pivot(index='group', columns='method', values='auroc')
    existing = read_table('runs/internal/rcr_calibration_distribution_20261024/NCM/matched_operating_vin_bootstrap.csv')
    if set(existing.group) != set(pivot.index):
        raise ValueError('Historical and current 43-bin cohorts differ')
    for row in existing.itertuples():
        np.testing.assert_allclose(pivot.loc[row.group, 'A']-pivot.loc[row.group, 'Raw'], row.delta_auroc, rtol=1e-10, atol=1e-12)
    existing.to_csv(output/'all_43_baseline_paired_VIN_intervals.csv', index=False)
    declining = existing[existing.delta_auroc < 0]
    test = read_table(output/'test_snippet_scores.csv', dtype={'vin': str})
    rows = []
    for old in declining.itertuples():
        regime, fault = old.group.split('/', 1)
        group = test[(test.operating_regime == regime) & ((test.sample_label == 0) | (test.fault_type == fault))]
        interval = paired_vin_bootstrap(group, 'A', 'M1', repeats)
        gate_interval = paired_vin_bootstrap(group, 'Full_selected', 'M1', repeats)
        identity = dict(group=old.group, normal_snippets=int((group.sample_label == 0).sum()),
            fault_snippets=int((group.sample_label == 1).sum()), normal_vins=interval['normal_vins'],
            fault_vins=interval['fault_vins'], baseline_delta_auroc=old.delta_auroc,
            baseline_ci_low=old.auroc_ci_low, baseline_ci_high=old.auroc_ci_high,
            M1_minus_A_delta_auroc=interval['delta_auroc'], M1_minus_A_ci_low=interval['auroc_ci_low'],
            M1_minus_A_ci_high=interval['auroc_ci_high'],
            M1_minus_Full_delta_auroc=gate_interval['delta_auroc'],
            M1_minus_Full_ci_low=gate_interval['auroc_ci_low'], M1_minus_Full_ci_high=gate_interval['auroc_ci_high'])
        for name in ['Raw', 'A', 'Full_selected', 'M1', 'M2']:
            row = local[(local.group == old.group) & (local.method == name)].iloc[0]
            for metric in ['ap', 'auroc', 'recall', 'fpr']:
                identity[name+'_'+metric] = row[metric]
        rows.append(identity)
    result = pd.DataFrame(rows)
    result.to_csv(output/'original_15_declining_bins_paired_evaluation.csv', index=False)
    facts = dict(eligible_bins=len(pivot), baseline_declines=int((pivot.A < pivot.Raw).sum()),
        baseline_unadjusted_CI_strictly_negative=int((existing.auroc_ci_high < 0).sum()),
        M1_below_raw=int((pivot.M1 < pivot.Raw).sum()), M1_below_A=int((pivot.M1 < pivot.A).sum()),
        original_15_M1_improved_vs_A=int((result.M1_minus_A_delta_auroc > 1e-12).sum()),
        original_15_M1_CI_positive=int((result.M1_minus_A_ci_low > 1e-12).sum()),
        original_15_M1_CI_negative=int((result.M1_minus_A_ci_high < -1e-12).sum()),
        original_15_gate_improved_vs_matched_full=int((result.M1_minus_Full_delta_auroc > 1e-12).sum()),
        original_15_gate_decreased_vs_matched_full=int((result.M1_minus_Full_delta_auroc < -1e-12).sum()),
        original_15_gate_ties_vs_matched_full=int((result.M1_minus_Full_delta_auroc.abs() <= 1e-12).sum()),
        original_15_gate_CI_positive=int((result.M1_minus_Full_ci_low > 1e-12).sum()),
        original_15_gate_CI_negative=int((result.M1_minus_Full_ci_high < -1e-12).sum()),
        all_43_gate_improved_vs_matched_full=int(((pivot.M1-pivot.Full_selected) > 1e-12).sum()),
        all_43_gate_decreased_vs_matched_full=int(((pivot.M1-pivot.Full_selected) < -1e-12).sum()),
        numeric_zero_tolerance=1e-12,
        post_hoc=True, multiple_comparisons_corrected=False)
    atomic_json(output/'local_analysis_facts.json', facts)
    return result, facts


def enhance_report(root, target):
    """Add interpretation and the complete paired scale sweep without refitting."""
    summary = read_table(root/'summary_metrics.csv')
    intervals = read_table(root/'paired_VIN_intervals.csv')
    facts = json.loads((root/'NCM/local_analysis_facts.json').read_text())
    def value(dataset, method, metric):
        return float(summary[(summary.dataset == dataset) & (summary.method == method)][metric].iloc[0])
    ci = intervals[(intervals.dataset == 'NCM') & (intervals.base == 'M1') &
                   (intervals.contender == 'Relation_conditional') & (intervals.metric == 'auroc')].iloc[0]
    interpretation = ['<!-- mechanism-summary:start -->',
        '**结论：CH 支持继续研究条件尺度；本版固定可靠性门控没有净增益，条件联合关系的正向证据主要来自 NCM；Brand3 没有支持两套新增机制替换原校准。**', '',
        f'1. CH 的完整条件尺度校准（不加门控）由正常验证选择 0.0001 下限：LFP AUROC '
        f'{value("LFP","A","auroc"):.4f}→{value("LFP","Full_selected","auroc"):.4f}，NCM '
        f'{value("NCM","A","auroc"):.4f}→{value("NCM","Full_selected","auroc"):.4f}；两项配对区间均为正。'
        '相同尺度下加入门控，两个体系的 AUROC 均略降，不能把尺度收益记到门控名下。',
        f'2. 原 NCM 15 个下降分箱中，M1 相对旧 A 有 {facts["original_15_M1_improved_vs_A"]} 个点估计改善；'
        f'但门控相对同尺度完整校准仅 {facts["original_15_gate_improved_vs_matched_full"]} 个改善、'
        f'{facts["original_15_gate_decreased_vs_matched_full"]} 个下降、{facts["original_15_gate_ties_vs_matched_full"]} 个不变。'
        f'门控净贡献的未校正区间为正/负分别 {facts["original_15_gate_CI_positive"]}/{facts["original_15_gate_CI_negative"]} 个。'
        '原 15 个新增配对区间均保存在 CSV，不能宣称门控稳定保留了故障信息。',
        f'3. NCM 条件关系评分 AUROC 为 {value("NCM","Relation_conditional","auroc"):.4f}，高于 M1 的 '
        f'{value("NCM","M1","auroc"):.4f}，配对差值 {ci.delta:+.4f} [{ci.ci_low:+.4f}, {ci.ci_high:+.4f}]；'
        '也超过无条件 LOO 与对角边缘对照。这是最小条件联合模型的证据，并非新神经掩码头的结构贡献已成立。LFP 未观察到同样的独立关系收益。',
        f'4. Brand3 的正常验证尺度最优不等于检测最优：M1 AUROC {value("Brand3","M1","auroc"):.4f}，'
        f'低于原 A 的 {value("Brand3","A","auroc"):.4f}。不能改用测试 AP 最好的尺度来隐藏这一反例。', '',
        '一个具体报警反例是 `discharge_I3_S2/high_resistance`：AUROC 从 A 的 0.7730 升至 M1 的 0.7966，'
        '但 Recall 从 0.30 降至 0.20，FPR 仍为 0.0337。排序改善不等于固定阈值报警效果改善。',
        '<!-- mechanism-summary:end -->', '']
    sweep = ['<!-- scale-sweep:start -->', '## 固定下限与稳健尺度的完整成对结果', '',
        '| 校准与尺度 | LFP AP / AUROC / F1 | NCM AP / AUROC / F1 | Brand3 五折 AP / AUROC / F1 |',
        '| --- | --- | --- | --- |']
    scale_rows = []
    for spec in SPECS:
        for family in ['Full', 'Reliable']:
            method = family+'_'+spec_name(spec)
            cells = []
            for dataset in ['LFP', 'NCM', 'Brand3']:
                row = summary[(summary.dataset == dataset) & (summary.method == method)].iloc[0]
                cells.append(' / '.join(f'{row[m]:.4f}' for m in ['ap', 'auroc', 'f1']))
                scale_rows.append(row.to_dict())
            sweep.append('| '+('完整' if family == 'Full' else '门控')+' / '+spec+' | '+' | '.join(cells)+' |')
    sweep.extend(['', '这些数值全部报告，不以测试指标选尺度。CH 的 0.05 对照尺度均退化为常数；'
        '稳健尺度是训练正常窗口 RMS 中位数，不能写成逐点 MAD。Brand3 的五折标准差保存于 `summary_metrics.csv`。', '',
        '| 单元 | 正常验证选择尺度 | 正常训练全局交叉拟合 g | 训练状态 g 范围 |', '| --- | --- | --- | --- |'])
    for unit in UNITS:
        calibration = ReliableConditionalCalibration.load(root/unit/'M1_calibration.npz')
        selection = json.loads((root/unit/'scale_selection.json').read_text())
        sweep.append(f'| {unit} | {selection["selected_scale"]} | {calibration.global_gate:.4f} | '
            f'{calibration.gates.min():.4f}–{calibration.gates.max():.4f} |')
    sweep.extend(['', 'g 范围只列训练状态的基本规则；超出训练工况支持域的连续衰减另由 P99 半径规则决定。',
                  '<!-- scale-sweep:end -->', ''])
    pd.DataFrame(scale_rows).to_csv(root/'paired_scale_sweep_summary.csv', index=False)
    body = target.read_text()
    body = re.sub(r'<!-- mechanism-summary:start -->.*?<!-- mechanism-summary:end -->\n*', '', body, flags=re.S)
    body = re.sub(r'<!-- scale-sweep:start -->.*?<!-- scale-sweep:end -->\n*', '', body, flags=re.S)
    title, rest = body.split('\n', 1)
    body = title+'\n\n'+'\n'.join(interpretation)+rest.lstrip('\n')
    body = body.replace('## NCM 原 15 个下降分箱', '\n'.join(sweep)+'\n## NCM 原 15 个下降分箱')
    new_facts = (f'原 43 个合格分箱中 {facts["baseline_declines"]} 个 AUROC 点估计下降，'
        f'{facts["baseline_unadjusted_CI_strictly_negative"]} 个未校正区间严格为负。'
        f'M1 相对 Raw 下降 {facts["M1_below_raw"]}/43 个，相对 A 下降 {facts["M1_below_A"]}/43 个；'
        f'原 15 个中 {facts["original_15_M1_improved_vs_A"]} 个点估计相对 A 改善，'
        f'{facts["original_15_M1_CI_positive"]} 个未校正配对区间为正，{facts["original_15_M1_CI_negative"]} 个为负。'
        '区间排除零的判定使用 1e-12 数值容差，不把浮点尾差算作正效应。所有局部筛选均为事后探索，未作多重比较校正，不作为独立确认。')
    body = re.sub(r'原 43 个合格分箱中[^\n]+', new_facts, body)
    body = body.replace('`local_metrics.csv`', '`local_metrics_audited.csv`')
    retention_rows = []
    for dataset in ['LFP', 'NCM', 'Brand3']:
        units = [dataset] if dataset != 'Brand3' else [f'Brand3_fold{i}' for i in range(5)]
        for label in [0, 1]:
            medians = []
            for unit in units:
                frame = read_table(root/unit/'energy_retention.csv')
                group = frame[frame.sample_label == label]
                medians.append(group[['Full_mean_removed_energy_fraction',
                    'Reliable_mean_removed_energy_fraction', 'sampled_mean_gate']].median())
            values = pd.DataFrame(medians).mean()
            retention_rows.append(dict(dataset=dataset, label=label, **values.to_dict()))
    retention = pd.DataFrame(retention_rows)
    retention.to_csv(root/'amplitude_retention_summary.csv', index=False)
    amplitude = ['<!-- amplitude-summary:start -->', '## 同单位残差幅值的实际变化', '',
        '| 数据集 | 样本 | 完整去均值能量削减中位数 | 门控去均值能量削减中位数 | 诊断 g 中位数 |',
        '| --- | --- | --- | --- | --- |']
    for row in retention.itertuples():
        amplitude.append(f'| {row.dataset} | '+('正常' if row.label == 0 else '故障')+
            f' | {row.Full_mean_removed_energy_fraction:.2%} | {row.Reliable_mean_removed_energy_fraction:.2%} | '
            f'{row.sampled_mean_gate:.4f} |')
    amplitude.extend(['', '能量削减定义为 `1 - 去均值后片段Top-5%能量 / Raw片段Top-5%能量`，'
        '两边均使用同一原始残差单位。负数表示去均值后能量增加。Brand3 列为各折中位数的等权平均；'
        'g 为每片段固定16点的诊断中位数，不能当作全窗口精确平均。', '',
        'LFP 正常能量明显下降，故障能量中位数反而增加；NCM 正常和故障能量均下降，程度不同。'
        '这些变化不能单独证明判别信息被抹除或保留，仍需结合配对排序区间、实际召回和误报。'
        '逐变量正常/故障绝对残差分位数和配对样本已单独保存，标准化 z 不与原始残差跨单位比较。',
        '<!-- amplitude-summary:end -->', ''])
    body = re.sub(r'<!-- amplitude-summary:start -->.*?<!-- amplitude-summary:end -->\n*', '', body, flags=re.S)
    body = body.replace('## 幅值分布与产物', '\n'.join(amplitude)+'\n## 幅值分布与产物')
    numeric_note = ('\n精度说明：审计使用 `float_precision="round_trip"`，保持近邻分数和并列排序。'
        '默认 CSV 回读在 Brand3 fold4 的个别局部指标上造成最高约 0.00327 的差异，'
        '现以 `local_metrics_audited.csv` 为准；原始 `.npy` 分数和主指标未改变。'
        '拟合/评分时的原始代码及 SHA256 已在 `implementation_snapshot/` 保存，'
        '当前 runner 已修复同一回读问题。\n')
    if '精度说明：审计使用' not in body:
        body += numeric_note
    target.write_text(body)
    local_path = root/'local_table_audit.json'
    if local_path.exists():
        local_audits = json.loads(local_path.read_text())
        audits = json.loads((root/'artifact_audit.json').read_text())
        for unit in UNITS:
            audits[unit]['local'] = local_audits[unit]
            atomic_json(root/unit/'artifact_audit.json', audits[unit])
        atomic_json(root/'artifact_audit.json', audits)
    expected = json.loads((root/'NCM/protocol_manifest.json').read_text())['implementations']
    snapshot = root/'implementation_snapshot'
    snapshot.mkdir(exist_ok=True)
    for filename, signature in expected.items():
        saved = snapshot/filename
        if not saved.exists():
            current = (Path(__file__).with_name(filename) if filename.startswith('run_')
                       else Path(__file__).parents[1]/'models'/filename)
            if hashlib.sha256(current.read_bytes()).hexdigest() != signature:
                raise ValueError('Original fitting implementation is missing and current source differs')
            shutil.copyfile(current, saved)
        if hashlib.sha256(saved.read_bytes()).hexdigest() != signature:
            raise ValueError('Original fitting implementation snapshot mismatch')
    print('LOCAL GATE CONTRAST AND PAIRED SCALE REPORT ENHANCED', target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs/internal/rcr_local_two_models_20261025_v1')
    parser.add_argument('--report', default='report/analysis/rcr/LOCAL_CALIBRATION_RELATION_MODELS_20261025.md')
    parser.add_argument('--bootstrap', type=int, default=500)
    args = parser.parse_args()
    root = Path(args.root)
    all_metrics, frames, audits, distributions = [], {}, {}, []
    for unit in UNITS:
        audits[unit], frames[unit] = audit_unit(root, unit)
        print('LOCAL MODELS FULL AUDIT COMPLETE', unit, flush=True)
        f = read_table(root/unit/'metrics.csv'); f.insert(0, 'unit', unit); all_metrics.append(f)
        distributions.append(residual_distributions(root, unit))
    data = pd.concat(all_metrics, ignore_index=True)
    data.to_csv(root/'all_metrics.csv', index=False)
    pd.concat(distributions).to_csv(root/'absolute_residual_quantiles.csv', index=False)
    intervals = []
    for dataset in ['LFP', 'NCM', 'Brand3']:
        units = [dataset] if dataset != 'Brand3' else [f'Brand3_fold{k}' for k in range(5)]
        vehicles = {}
        for fold, unit in enumerate(units):
            for method in PRIMARY:
                f = frames[unit]['test'][['vin', 'sample_label', method]].rename(
                    columns={'vin': 'car', 'sample_label': 'label', method: 'score'})
                vehicles[fold, method] = dict(test=f,
                    threshold=float(frames[unit]['validation'][method].quantile(.95)))
        result = paired_car_bootstrap(vehicles, PAIRS, args.bootstrap)
        result.insert(0, 'dataset', dataset); intervals.append(result)
        print('PAIRED VIN INTERVALS COMPLETE', dataset, flush=True)
    intervals = pd.concat(intervals, ignore_index=True)
    intervals.to_csv(root/'paired_VIN_intervals.csv', index=False)
    local, facts = ncm_local_analysis(root, args.bootstrap)
    primary = data[data.q == .95]
    summary_rows = []
    for dataset in ['LFP', 'NCM', 'Brand3']:
        subset = primary[primary.unit.str.startswith('Brand3_fold')] if dataset == 'Brand3' else primary[primary.unit == dataset]
        for method, group in subset.groupby('method'):
            summary_rows.append(dict(dataset=dataset, method=method,
                **{metric: float(group[metric].mean()) for metric in METRICS},
                **{metric+'_fold_sd': float(group[metric].std()) if len(group) > 1 else 0. for metric in METRICS}))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root/'summary_metrics.csv', index=False)
    figdir = Path('report/figures/local_calibration_relation_20261025'); figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    display_methods = ['A', 'Full_selected', 'M1', 'Relation_conditional', 'M2']
    for ax, dataset in zip(axes, ['LFP', 'NCM', 'Brand3']):
        table = summary[summary.dataset == dataset].set_index('method')
        ax.bar(np.arange(len(display_methods)), [table.loc[m, 'auroc'] for m in display_methods], color=['#999999', '#247A98', '#CE6647', '#639B66', '#9474B6'])
        ax.set(title=dataset, ylabel='AUROC', ylim=(0, 1), xticks=np.arange(len(display_methods)),
            xticklabels=['Current', 'Scale', 'M1 gate', 'Cond. LOO', 'M2 fusion'])
        ax.tick_params(axis='x', labelrotation=25); ax.grid(axis='y', alpha=.2)
    fig.savefig(figdir/'primary_auroc.png', dpi=180); fig.savefig(figdir/'primary_auroc.svg'); plt.close(fig)
    lines = ['# CH 与 Brand3：局部可靠性校准和条件联合关系最小验证（2026-10-25）', '',
        '两套机制均已在 LFP、NCM 与 Brand3 五折完成。模型一为正常 VIN 交叉拟合的固定可靠性门控＋条件尺度；模型二为同一残差上的条件 Gaussian ridge LOO。它们是可审计的最小统计模型，不等同于新的完整神经网络已经成立。', '',
        '| 方法 | LFP AP / AUROC / F1 | NCM AP / AUROC / F1 | Brand3 五折 AP / AUROC / F1 |', '| --- | --- | --- | --- |']
    for method in PRIMARY:
        cells = []
        for dataset in ['LFP', 'NCM', 'Brand3']:
            r = summary[(summary.dataset == dataset) & (summary.method == method)].iloc[0]
            cells.append(' / '.join(f'{r[m]:.4f}' for m in ['ap', 'auroc', 'f1']))
        lines.append('| '+method+' | '+' | '.join(cells)+' |')
    lines.extend(['', 'A 为已有 RCR 校准幅值，Raw/A 逐窗原样保留。Full_selected 与 M1 使用正常验证选出的同一尺度规格；前者完整减去条件均值，后者仅由正常训练确定 g(c)。Relation_conditional 是独立关系分数；M2 固定等权融合 M1 与关系训练 CDF。M1_CDF 与无条件关系、对角边缘对照用于分离 CDF、边缘重加权和条件联合关系收益。', '',
        '## 正常数据决定的规则', '',
        '- Backbone、原始训练/验证/测试划分均冻结；CH 片段 Top-5%，Brand3 原有片段/车辆 Top-5%。主阈值为正常验证 P95，P99 全部保存。Brand3 九辆故障车跨折重复，bootstrap 按全局唯一车辆复用权重；CH 同一 VIN 的相关片段成组重采样。',
        '- g(c)：五折正常 VIN 交叉拟合均值回归，在训练正常电流/SOC 状态内求预测均值的最优收缩；状态内每 VIN 等权，10 VIN 的预设收缩先验，少于 5 VIN 回退全局规则；超出正常训练工况半径 P99 连续减弱校准。推理 g 只读工况，不读响应残差。不是用测试故障学门控，也不保证故障信息一定保留。',
        '- 尺度：完整减均值和门控减均值成对比较 0.05/0.01/0.001/0.0001/0.00001；稳健尺度为正常窗口 RMS 中位数（对窗口级尾部稳健），不是逐点 MAD。完整校准正常验证 Gaussian NLL 选尺度规格，再原样用于门控分支。不能将正常 NLL 最优等同于检测最优。',
        '- 条件联合关系：每变量从预测输入中排除自身及所有包含自身的交互项；使用其他变量、连续工况及工况×其他变量交互。全量正常窗口的充分矩保留全部时刻，没有统计抽样。ridge 在 1e-4/0.01/1 中只按正常验证 NLL 选择；q 的外推界由正常训练 P1/P99 给定。',
        '- 关系预测的训练残差方差用于 Gaussian 标准化，另报不使用其他变量的对角边缘评分。融合固定 alpha=0.5，训练正常 CDF 先于片段/车辆聚合。', '',
        '## NCM 原 15 个下降分箱', '',
        f'原 43 个合格分箱中 {facts["baseline_declines"]} 个 AUROC 点估计下降，{facts["baseline_unadjusted_CI_strictly_negative"]} 个未校正区间严格为负。'
        f'M1 相对 Raw 下降 {facts["M1_below_raw"]}/43 个，相对 A 下降 {facts["M1_below_A"]}/43 个；'
        f'原 15 个中 {facts["original_15_M1_improved_vs_A"]} 个点估计相对 A 改善，{facts["original_15_M1_CI_positive"]} 个未校正配对区间为正，'
        f'{facts["original_15_M1_CI_negative"]} 个为负。所有局部筛选均为事后探索，未作多重比较校正，不作为独立确认。', '',
        '| 分箱/故障 | 正常/故障片段 | 正常/故障 VIN | A−Raw AUROC [95% CI] | M1−A AUROC [95% CI] | Recall A→M1 | FPR A→M1 |',
        '| --- | --- | --- | --- | --- | --- | --- |'])
    for r in local.itertuples():
        lines.append(f'| {r.group} | {r.normal_snippets}/{r.fault_snippets} | {r.normal_vins}/{r.fault_vins} | '
            f'{r.baseline_delta_auroc:+.4f} [{r.baseline_ci_low:+.4f}, {r.baseline_ci_high:+.4f}] | '
            f'{r.M1_minus_A_delta_auroc:+.4f} [{r.M1_minus_A_ci_low:+.4f}, {r.M1_minus_A_ci_high:+.4f}] | '
            f'{r.A_recall:.3f}→{r.M1_recall:.3f} | {r.A_fpr:.3f}→{r.M1_fpr:.3f} |')
    lines.extend(['', '43 个基线配对区间复核后完整保存；原 15 个额外保存 M1 的成对区间。`local_metrics_audited.csv` 用精确浮点回读复算所有合格分箱的 AP/AUROC/Recall/FPR 与样本量；`local_fault_composition.csv` 保存工况内故障类别组成。固定使用各方法全局正常验证阈值，不在局部分箱重新找测试最优阈值。', '',
        '## 主要配对 VIN 区间', '', '| 数据集 | 对照 → 方法 | 指标 | 差值 [95% CI] |', '| --- | --- | --- | --- |'])
    for r in intervals.itertuples():
        if r.metric in ['ap', 'auroc']:
            lines.append(f'| {r.dataset} | {r.base} → {r.contender} | {r.metric.upper()} | {r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] |')
    lines.extend(['', '区间固定随机种子 3407、500 次车辆分层重采样；方法各自的正常验证阈值固定。区间不包括 Backbone 与验证阈值估计的不确定性。所有结果均报告，不按测试 AP 选门控、尺度、ridge 或融合。', '',
        '## 幅值分布与产物', '',
        '- `absolute_residual_quantiles.csv`：正常/故障逐变量绝对残差分位数，Raw、完整去均值、门控去均值使用相同原始残差单位；标准化 z 单列，不能跨单位将压缩误认为故障信息损失。每片段固定配对抽 16 个残差点仅用于分布诊断，主检测指标来自全量窗口。',
        '- `energy_retention.csv`：每片段去均值前后的全量 Top-5% 能量比与诊断抽样平均 g。故障能量下降不自动等于检测信息抹除，须和同一批对象的排序、召回及区间合看。',
        f'- 任务 `RCR-LOCAL-TWO-MODELS-20261025-v1`，输出 `{root}`。7 个单元含全部校准、门控审计、正常车辆交叉拟合映射、充分矩、条件/无条件关系参数、阈值、逐窗口/片段/车辆分数和故障组成。',
        '- `artifact_audit.json`：Raw/A 原分数一致，M1 全量显式残差重放，关系全量显式目标排除预测重放，融合及检测指标复算。',
        f'- 图 `{figdir}/primary_auroc.png` 及 SVG。本表为独立机制验证，不更改论文最终四表；也不宣称多尺度或新神经关系头已获得验证。'])
    target = Path(args.report); target.parent.mkdir(parents=True, exist_ok=True); target.write_text('\n'.join(lines)+'\n')
    atomic_json(root/'artifact_audit.json', audits)
    enhance_report(root, target)
    atomic_json(root/'complete.json', dict(task='RCR-LOCAL-TWO-MODELS-20261025-v1', units=UNITS,
        artifact_audit_complete=True, normal_only=True, no_test_tuning=True, neural_architecture_validation=False))
    print(summary[summary.method.isin(['A', 'Full_selected', 'M1', 'Relation_conditional', 'M2'])]
        [['dataset', 'method', 'ap', 'auroc', 'f1', 'fpr']].round(4).to_string(index=False))
    print('TWO MODEL DIAGNOSTIC REPORT COMPLETE', target, flush=True)


if __name__ == '__main__':
    main()
