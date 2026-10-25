"""Read completed evidence, add state-support diagnostics, export report/figures."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rcr-scale-relation-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scripts.evaluate_rcr import metrics

from src.models.local_residual_models import ReliableConditionalCalibration
from src.models.residual_scale_probe import scale_components
from src.runners.probe_scale_relation_evidence import (UNITS, ROOT, read, source_for,
    load_context, paired_bootstrap, evaluate, rank_frame)
from src.runners.run_rcr import atomic_json


def state_support(root):
    rows, correlation, selection_rows, extra_entries, supported_rows = [], [], [], [], []
    for unit in UNITS[2:]:
        p = root/'scale'/unit
        selection = json.loads((p/'selection.json').read_text())
        source, _ = source_for(unit)
        mapping, context, _ = load_context(source, p)
        model = ReliableConditionalCalibration.load(ROOT/unit/(selection['log_family_source']+'.npz'))
        with np.load(p/'global_scales.npz') as scales:
            global_scale = scales['log']
        train_q = context['train']['q']
        radius_limit = float(np.quantile(np.square((train_q-model.q_mean)/model.q_std).mean(1), .99))
        for partition, c in context.items():
            q = c['q']; sigma = model.parameters(q, c['groups'])[1]
            factor = np.exp(np.log(sigma/global_scale).mean(1))
            radius = np.square((q-model.q_mean)/model.q_std).mean(1)
            seq = c['refs'][:, 0].astype(int)
            f = c['frame'].iloc[seq]
            for label in sorted(f.sample_label.unique()):
                mask = f.sample_label.to_numpy() == label
                rows.append(dict(unit=unit, partition=partition, label=int(label), windows=int(mask.sum()),
                    vins=int(f.loc[mask, 'vin'].nunique()), shared_factor_median=float(np.median(factor[mask])),
                    shared_factor_p05=float(np.quantile(factor[mask], .05)),
                    shared_factor_p95=float(np.quantile(factor[mask], .95)),
                    out_of_support_fraction=float(np.mean(radius[mask] > radius_limit)),
                    lower_clip_fraction=float(np.mean(np.isclose(sigma[mask], model.scale_low, rtol=1e-10, atol=1e-14))),
                    upper_clip_fraction=float(np.mean(np.isclose(sigma[mask], model.scale_high, rtol=1e-10, atol=1e-14)))))
            if partition == 'test':
                window = np.load(p/'test_window_scores.npy', mmap_mode='r')
                # First nine columns are documented by the scoring runner.
                correlation.append(dict(unit=unit,
                    spearman_shared_scale_and_mean_energy=float(spearmanr(factor, window[:, 1]).statistic),
                    spearman_shared_scale_and_global_energy=float(spearmanr(factor, window[:, 2]).statistic)))
                names = ['A', 'Mean_only', 'Global_rms', 'Global_median', 'Log_conditional',
                         'Robust_conditional', 'Log_shared_state', 'Log_variable_shape', 'Previous_selected']
                # Brand3 protocol has one complete window per snippet.
                np.testing.assert_array_equal(seq, np.arange(len(c['frame'])))
                for scope, mask in [('all', np.ones(len(q), dtype=bool)),
                                    ('within_train_P99', radius <= radius_limit)]:
                    snippets = c['frame'].copy()
                    for j, name in enumerate(names):
                        snippets[name] = window[:, j]
                    ranked = rank_frame(snippets.loc[mask], names, True)
                    for name in names:
                        supported_rows.append(dict(unit=unit, scope=scope, method=name,
                            retained_windows=int(mask.sum()), total_windows=len(q),
                            retained_normal_vins=int(ranked.loc[ranked.sample_label == 0, 'vin'].nunique()),
                            retained_fault_vins=int(ranked.loc[ranked.sample_label == 1, 'vin'].nunique()),
                            **metrics(ranked.sample_label, ranked[name], np.inf)))
        records = pd.DataFrame(selection['selection_records'])
        window_winner = records.loc[records.normal_validation_window_nll.idxmin(), 'method']
        for rule, method in [('VIN_NLL', selection['selected_nll']),
                             ('one_SE', selection['selected_one_se']), ('window_NLL', window_winner)]:
            row = records.set_index('method').loc[method].to_dict()
            selection_rows.append(dict(unit=unit, rule=rule, selected=method, **row))
        frames = {s: read(p/f'{s}_scores.csv') for s in ['validation', 'test']}
        for f in frames.values():
            f['Selected_window_NLL'] = f[window_winner]
        supplementary = root/'supplement'/unit
        supplementary.mkdir(parents=True, exist_ok=True)
        for s, f in frames.items():
            f.to_csv(supplementary/f'{s}_scores.csv', index=False)
        _, thresholds = evaluate(frames, ['A', 'Global_rms', 'Log_conditional', 'Log_shared_state',
            'Log_variable_shape', 'Selected_window_NLL'], unit, supplementary)
        extra_entries.append((frames, thresholds))
    pd.DataFrame(rows).to_csv(root/'state_support_diagnostics.csv', index=False)
    pd.DataFrame(correlation).to_csv(root/'shared_factor_correlations.csv', index=False)
    pd.DataFrame(selection_rows).to_csv(root/'normal_selection_audit.csv', index=False)
    pd.DataFrame(supported_rows).to_csv(root/'in_support_ranking_diagnostic.csv', index=False)
    pairs = [('Global_rms', 'Log_shared_state'), ('Global_rms', 'Log_variable_shape'),
             ('Log_variable_shape', 'Log_conditional'), ('A', 'Selected_window_NLL')]
    ci = paired_bootstrap(extra_entries, pairs, 500, 'Brand3')
    pd.DataFrame(ci).to_csv(root/'supplementary_paired_VIN_intervals.csv', index=False)


def make_report(root, report, figures):
    assert (root/'complete.json').exists()
    summary = read(root/'summary_metrics.csv')
    cis = pd.concat([read(root/'paired_VIN_intervals.csv'),
                     read(root/'supplementary_paired_VIN_intervals.csv').assign(part='scale')])
    overlap = read(root/'error_overlap.csv')
    sel = read(root/'normal_selection_audit.csv')
    scale = summary.query('part=="scale" and q==.95').set_index('method')
    rel = summary.query('part=="relation" and q==.95').set_index(['dataset', 'method'])

    def ci_text(dataset, base, contender, metric='auroc', part='relation', q=.95):
        r = cis[(cis.dataset == dataset) & (cis.base == base) & (cis.contender == contender) &
                (cis.metric == metric) & (cis.part == part) & (cis.q == q)].iloc[0]
        return f'{r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]'

    def cell(row):
        return f'{row.ap_mean:.4f} / {row.auroc_mean:.4f} / {row.f1_mean:.4f}'

    lines = ['# Brand3 尺度退化归因与 CH/Brand3 关系互补性验证（2026-10-25）', '',
        '**结论：Brand3 的主要退化来自随工况变化的公共尺度因子；全局收缩能够恢复原有表现，但尚未证明条件尺度稳定优于原 RCR。关系分支的可信互补证据集中在 NCM；LFP 与 Brand3 尚不支持普遍获益。**', '',
        f'任务 `{json.loads((root/"protocol_manifest.json").read_text())["task"]}`。5 个 Brand3 尺度单元、7 个关系单元全部完成；复用冻结 Backbone 和完整窗口评分，不重新训练编码器。本文是独立机制验证，未修改最终四张论文实验矩阵。', '',
        '## 1. Brand3：尺度损失发生在哪里', '',
        '保持条件均值 μ 完全不变，不加可靠性门控。全局尺度分别使用正常训练去均值能量的 RMS，以及正常窗口 RMS 中位数；后者仅对窗口级尾部稳健，不是逐点 MAD。连续条件尺度仍取此前正常验证选出的 log-RMS 模型。', '',
        '| 尺度方式 | AP / AUROC / F1 | Recall | FPR |',
        '| --- | --- | --- | --- |']
    scale_names = [('A', '原 RCR'), ('Mean_only', '只减条件均值'), ('Global_rms', '全局 RMS'),
        ('Global_median', '全局稳健中位尺度'), ('Previous_selected', '上一轮正常验证所选条件尺度'),
        ('Log_conditional', '连续条件 log-RMS'), ('Robust_conditional', '工况分箱稳健中位尺度'),
        ('Log_shared_state', '只保留公共工况尺度因子'), ('Log_variable_shape', '只保留变量相对尺度'),
        ('Selected_NLL', '正常 VIN 等权 NLL 选择收缩'), ('Selected_1SE', '正常 VIN 一标准误选择收缩')]
    for name, label in scale_names:
        r = scale.loc[name]
        lines.append(f'| {label} | {cell(r)} | {r.recall_mean:.4f} | {r.fpr_mean:.4f} |')
    lines += ['', 'Brand3 均为五折等权；同一九辆故障车跨折重复，不把 45 次故障测试当作 45 辆独立故障车。正常验证每折仅 18–19 辆，阈值 P95 的经验分辨率约为 1/18，测试 FPR 可明显偏离 5%。', '',
        '将条件尺度精确分为 `sigma_i(c)=s_global,i × a(c) × r_i(c)`，其中 `a(c)` 为七变量尺度比的几何均值，`r_i(c)` 的跨变量几何均值为 1。公共因子只改变一个窗口的整体强度，相对尺度只改变变量权重。重新执行同样 Top-5% 聚合后：', '',
        f'- 全局 RMS → 完整连续条件尺度的 AUROC 差值 {ci_text("Brand3", "Global_rms", "Log_conditional", part="scale")}。',
        f'- 全局 RMS → 只保留公共因子的 AUROC 差值 {ci_text("Brand3", "Global_rms", "Log_shared_state", part="scale")}。',
        f'- 全局 RMS → 只保留变量相对尺度的 AUROC 差值 {ci_text("Brand3", "Global_rms", "Log_variable_shape", part="scale")}。',
        f'- 相对尺度 → 完整条件尺度的 AUROC 差值 {ci_text("Brand3", "Log_variable_shape", "Log_conditional", part="scale")}。', '',
        '因此，主要损失不能归结为固定的变量权重或单个极小尺度：公共工况归一化本身已复现大部分退化。删除单变量的七项事后诊断也未将完整连续尺度恢复到原 RCR（完整数值见 `all_metrics.csv`）。相对尺度的较好点估计属于归因对照，不能据此在测试集反选新检测器或宣称它已获得独立验证。', '']
    support = read(root/'state_support_diagnostics.csv')
    supp = support.query('partition=="test"').groupby('label').mean(numeric_only=True)
    corr = read(root/'shared_factor_correlations.csv')
    lines += ['| 测试样本 | 公共尺度因子窗口中位数（五折平均） | 超出训练 q 半径 P99 占比 |',
              '| --- | --- | --- |']
    for label, title in [(0, '正常'), (1, '故障')]:
        r = supp.loc[label]
        lines.append(f'| {title} | {r.shared_factor_median:.4f} | {r.out_of_support_fraction:.2%} |')
    lines += ['', f'公共尺度与未归一化去均值能量的窗口级 Spearman 相关，五折平均为 {corr.spearman_shared_scale_and_mean_energy.mean():.4f}。高能量样本也常获得较大分母，导致排序被压缩。这是固定模型的反事实归因，不能仅凭相关或高尺度就断定故障信息被模型主动抹除；正常工况组成差异与外推均可能参与。逐折支持域、截断比例、逐变量权重和状态内标准化能量均已保存。', '',
        '## 2. 稳健尺度与全局尺度收缩', '',
        '固定网格 λ∈{0,0.25,0.5,0.75,1}，使用 `sigma²=(1−λ) sigma_global²+λ sigma_conditional²`。λ=0 是全局，λ=1 是完整条件尺度。两类尺度共 10 个候选全部报告，不按测试 AP 选择。', '',
        '| λ | log-RMS AP / AUROC / F1 | 稳健尺度 AP / AUROC / F1 |',
        '| --- | --- | --- |']
    for lam in [0, 25, 50, 75, 100]:
        lines.append(f'| {lam/100:g} | {cell(scale.loc[f"log_lambda{lam:03d}"])} | {cell(scale.loc[f"robust_lambda{lam:03d}"])} |')
    lines += ['', '| 折 | 正常 VIN NLL 选择 | 一标准误选择 | 原口径窗口等权 NLL 选择 |',
              '| --- | --- | --- |']
    for unit, group in sel.groupby('unit'):
        v = group.set_index('rule').selected
        lines.append(f'| {unit} | {v["VIN_NLL"]} | {v["one_SE"]} | {v["window_NLL"]} |')
    lines += ['', '一标准误规则：在最小正常验证 VIN 等权 NLL 加该候选 VIN 标准误范围内，选 λ 最小者；同 λ 时按正常 NLL 选尺度族。标准误来自完整 VIN，不把窗口当独立验证对象。该规则预先固定，未读取测试指标。原窗口等权选择另行复核，避免将 VIN 重加权与尺度收缩混淆。', '',
        f'- 稳健条件尺度相对原 RCR：AUROC {ci_text("Brand3", "A", "Robust_conditional", part="scale")}。稳健化能减轻连续尺度退化，仍未恢复原 RCR。',
        f'- VIN NLL 最优收缩相对原 RCR：AUROC {ci_text("Brand3", "A", "Selected_NLL", part="scale")}。',
        f'- 原窗口等权 NLL 选择相对原 RCR：AUROC {ci_text("Brand3", "A", "Selected_window_NLL", part="scale")}。',
        f'- 一标准误收缩相对上一轮所选条件尺度：AUROC {ci_text("Brand3", "Previous_selected", "Selected_1SE", part="scale")}；相对原 RCR为 {ci_text("Brand3", "A", "Selected_1SE", part="scale")}。', '',
        '**条件分布建模的可靠空间：当前可支持的是加入保守回退，避免不可靠条件尺度伤害原有评分；尚无证据支持在 Brand3 上让完整条件方差成为主分数。保留变量相对尺度、限制公共因子是后续候选，应在新的独立划分上确认，不能把这次归因对照的测试高点当正式改进。**', '',
        '## 3. RCR 与 LOO 的错检、漏检和独立检出', '',
        'A 为原 RCR；LOO 为此前在 M1 校准 z 上拟合的条件 Gaussian ridge 关系模型。另用同 z 的 M1、对角边缘评分、无条件 LOO 作对照，以区分旧/新尺度输入差异与联合关系本身。这里分析的是该统计关系分支，不能推广成新神经关系网络已成立。', '',
        '| 对照 | 数据集 | 两者均检出故障 | 仅对照检出 | 仅条件 LOO 检出 | 两者均漏检 | LOO 新增正常误报 | 补检涉及 VIN |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    main = overlap.query('q==.95 and grouping=="all"')
    for unit in ['LFP', 'NCM']:
        for base in ['A', 'M1', 'Marginal_diag']:
            r = main[(main.unit == unit) & (main.base == base)].iloc[0]
            lines.append(f'| {base} | {unit} | {r.fault_both} | {r.fault_base_only} | {r.fault_relation_only} | {r.fault_neither} | {r.normal_relation_only} | {r.fault_unique_vins_rescued} |')
    lines += ['', '以上 CH 数量为片段；“补检涉及 VIN”表示该 VIN 至少有一个片段被补检，不等于新增检出车辆。逐样本集合包含分数、阈值、故障类别、工况和 VIN，位于每单元的 `prediction_sets.csv`。', '',
        '| Brand3 折 | 两者均检出 | 仅 A 检出 | 仅 LOO 检出 | 均漏检 | LOO 新增正常误报 |',
        '| --- | --- | --- | --- | --- | --- |']
    for unit in UNITS[2:]:
        r = main[(main.unit == unit) & (main.base == 'A')].iloc[0]
        lines.append(f'| {unit} | {r.fault_both} | {r.fault_base_only} | {r.fault_relation_only} | {r.fault_neither} | {r.normal_relation_only} |')
    unique_rescues = set()
    for unit in UNITS[2:]:
        preds = read(root/'relation'/unit/'prediction_sets.csv')
        subset = preds[(preds.q == .95) & (preds.base == 'A') & (preds.sample_label == 1) &
                       (preds['set'] == 'relation_only')]
        unique_rescues.update(subset.vin)
    lines += ['', f'Brand3 表中单位为车辆，跨折补检共涉及 {len(unique_rescues)} 辆唯一故障车；每折存在的独立检出并未形成稳定的五折收益。故障细分类未提供，不推断类型。', '',
        '| CH 故障类型 | LFP：LOO补检 / A独检 | NCM：LOO补检 / A独检 |',
        '| --- | --- | --- |']
    for fault in ['high_resistance', 'low_capacity', 'self_discharge']:
        cells = []
        for unit in ['LFP', 'NCM']:
            r = overlap[(overlap.unit == unit) & (overlap.q == .95) & (overlap.base == 'A') &
                        (overlap.grouping == 'fault_type') & (overlap.group == fault)].iloc[0]
            cells.append(f'{r.fault_relation_only} / {r.fault_base_only}')
        lines.append(f'| {fault} | '+ ' | '.join(cells)+' |')
    lines += ['', '## 4. 在相同验证报警预算下，互补信息能否转为收益', '',
        '固定 0.5 权重，将正常训练 CDF 后的 A 与 LOO 在窗口层融合，再按原协议聚合；阈值由各方法正常验证 P95 确定。补充 max-CDF 融合，对联合分数重新定正常验证阈值；直接对两个 P95 报警取 OR 只作覆盖诊断，其验证 FPR 与新增误报另存，不能作为同预算提升。', '',
        '| 方法 | LFP AP / AUROC / F1 | NCM AP / AUROC / F1 | Brand3 AP / AUROC / F1 |',
        '| --- | --- | --- | --- |']
    for name in ['A', 'Relation_conditional', 'A_CDF', 'A_Diag_fusion', 'A_Relation_fusion', 'A_Relation_max']:
        lines.append('| '+name+' | '+' | '.join(cell(rel.loc[d, name]) for d in ['LFP', 'NCM', 'Brand3'])+' |')
    lines += ['', '| 数据集 | A→等权关系融合 AUROC差值 [95% CI] | Recall差值 [95% CI] | FPR差值 [95% CI] |',
        '| --- | --- | --- | --- |']
    for d in ['LFP', 'NCM', 'Brand3']:
        lines.append(f'| {d} | {ci_text(d,"A","A_Relation_fusion")} | {ci_text(d,"A","A_Relation_fusion","recall")} | {ci_text(d,"A","A_Relation_fusion","fpr")} |')
    lines += ['', f'NCM 的关系融合相对 A-CDF 的 AUROC 差值为 {ci_text("NCM", "A_CDF", "A_Relation_fusion")}；相对 A+对角边缘融合仍为 {ci_text("NCM", "A_Diag_fusion", "A_Relation_fusion")}。条件 LOO 相对同输入对角评分为 {ci_text("NCM", "Marginal_diag", "Relation_conditional")}，相对无条件 LOO 为 {ci_text("NCM", "Relation_unconditional", "Relation_conditional")}。这些结果支持 NCM 的收益包含条件联合关系信息，不能只由 CDF 或边缘重加权解释。', '',
        'NCM 原 RCR 的 P95 Recall/FPR 为 '+f'{rel.loc["NCM","A"].recall_mean:.4f}/{rel.loc["NCM","A"].fpr_mean:.4f}，等权融合为 {rel.loc["NCM","A_Relation_fusion"].recall_mean:.4f}/{rel.loc["NCM","A_Relation_fusion"].fpr_mean:.4f}。'+
        'LFP 独立补检少于 A 的独检，融合区间跨零；Brand3 的折间差异和误报代价均较大，目前不能确认互补。', '',
        '## 5. 证据边界与产物', '',
        '- 全部指标保持原训练/验证/测试划分、Backbone、Top-5% 聚合及 `score >= threshold` 判定。P95 为主，P99 为预定敏感性检查；`all_metrics.csv`、`paired_VIN_intervals.csv` 含全部结果。',
        '- 500 次标签分层 VIN 成组配对 bootstrap，seed3407；同 VIN 的相关片段共享权重，Brand3 跨折重复车辆也共享权重。Brand3 折均值区间不把折视为独立样本。区间固定训练模型与验证阈值，不包含重训练或阈值估计不确定性；未作多重比较校正。',
        '- `scale/*/fault_normal_rank_pairs.csv` 为车辆级正常/故障排序翻转，`variable_scale_diagnostics.csv` 为尺度与能量贡献，`state_tail_diagnostics.csv` 为工况内 NLL/尾部，`state_support_diagnostics.csv` 为外推与公共尺度摘要。',
        '- `normal_selection_audit.csv` 和各尺度 `selection.json` 保存正常选择记录；原窗口等权选择保存在 `supplement/`，不篡改主任务。',
        '- `relation/*/prediction_sets.csv`、`error_overlap.csv` 保存错检/漏检集合与分组计数，`unbudgeted_OR_diagnostic.csv` 保存未经联合报警率控制的 OR 代价。',
        '- 直接重放能量与原始分数一致性见每单元 `audit.json`；主分数 CSV 均用 `float_precision="round_trip"` 回读以保留并列分数。', '',
        '**设计优先级：Brand3 优先保留全局尺度回退，暂不升级复杂条件分布网络；关系模型优先围绕 NCM 已观察到的高内阻、低容量补检，使用独立数据确认联合关系收益。**', '']
    supported = read(root/'in_support_ranking_diagnostic.csv')
    smean = supported.query('scope=="within_train_P99"').groupby('method').auroc.mean()
    support_text = ('将超出训练 q 半径 P99 的测试窗口剔除、重新按车辆 Top-5% 聚合后，'
        f'全局 RMS AUROC={smean["Global_rms"]:.4f}，完整条件尺度={smean["Log_conditional"]:.4f}，'
        f'只保留公共因子={smean["Log_shared_state"]:.4f}。'
        '这项事后支持域诊断仍使用同一批保留下来的车辆比较各分数，提示退化并非仅由外推窗口造成；'
        '保留车辆数量与逐折结果见 `in_support_ranking_diagnostic.csv`。')
    position = lines.index('## 2. 稳健尺度与全局尺度收缩')
    lines[position:position] = [support_text, '']
    losses = []
    for unit in UNITS[2:]:
        records = json.loads((root/'scale'/unit/'selection.json').read_text())['selection_records']
        losses.append(pd.DataFrame(records))
    losses = pd.concat(losses).groupby('method').mean(numeric_only=True)
    nll_text = ('正常验证 VIN 等权 NLL 的五折均值：全局 RMS '
        f'{losses.loc["log_lambda000", "normal_validation_vin_nll"]:.4f}，'
        f'λ=0.25 为 {losses.loc["log_lambda025", "normal_validation_vin_nll"]:.4f}，'
        f'完整连续条件尺度 {losses.loc["log_lambda100", "normal_validation_vin_nll"]:.4f}（越低越好）。'
        '加入全局 RMS 对照后，完整条件尺度在正常验证本身也不占优；'
        '部分收缩的正常拟合小幅改善仍没有转化为检测收益，不能把正常 NLL 当异常 AP 的替代目标。')
    position = lines.index('## 3. RCR 与 LOO 的错检、漏检和独立检出')
    lines[position:position] = [nll_text, '']
    sensitivity = ('P99 敏感性检查：NCM 融合的排序优势保持；但相对 A 的 Recall 差值为 '
        f'{ci_text("NCM", "A", "A_Relation_fusion", "recall", q=.99)}，'
        f'F1 差值为 {ci_text("NCM", "A", "A_Relation_fusion", "f1", q=.99)}，'
        '均跨零。因此，实际报警收益目前主要在主协议 P95 下获得支持，不能声称对更严格阈值同样稳定。')
    position = lines.index('## 5. 证据边界与产物')
    lines[position:position] = [sensitivity, '']
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(lines))
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for family, title in [('log', 'Conditional log-RMS'), ('robust', 'State robust median')]:
        values = [scale.loc[f'{family}_lambda{k:03d}', 'auroc_mean'] for k in [0,25,50,75,100]]
        ax[0].plot([0,.25,.5,.75,1], values, 'o-', label=title)
    ax[0].axhline(scale.loc['A','auroc_mean'], color='black', ls='--', label='Original RCR')
    ax[0].set(xlabel='Conditional variance weight', ylabel='AUROC', title='Brand3 scale shrinkage (five-fold mean)')
    ax[0].legend(fontsize=8)
    x = np.arange(3)
    for offset, name, title in [(-.25,'A','RCR'), (0,'A_Relation_fusion','RCR + LOO'), (.25,'Relation_conditional','LOO')]:
        ax[1].bar(x+offset, [rel.loc[d,name].auroc_mean for d in ['LFP','NCM','Brand3']], .25, label=title)
    ax[1].set(xticks=x, xticklabels=['LFP','NCM','Brand3'], ylim=(.65,1), ylabel='AUROC', title='Relation complementarity')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ['png','svg']:
        fig.savefig(figures/f'scale_relation_evidence.{suffix}', dpi=180)
    plt.close(fig)
    atomic_json(root/'report_complete.json', dict(report=str(report), figures=str(figures),
        unique_brand3_fault_cars_rescued=len(unique_rescues)))
    print('REPORT COMPLETE', report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs/internal/rcr_scale_relation_evidence_20261025_v1')
    args = parser.parse_args()
    root = Path(args.root)
    state_support(root)
    make_report(root, Path('report/analysis/rcr/SCALE_RELATION_EVIDENCE_20261025.md'),
                Path('report/figures/scale_relation_evidence_20261025'))


if __name__ == '__main__':
    main()
