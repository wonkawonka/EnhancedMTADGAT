"""Read-only score/selection audit and paired VIN analysis for simplified MSCM."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mscm-simple-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.evaluate_rcr import metrics
from src.runners.probe_scale_relation_evidence import read, paired_bootstrap
from src.runners.run_mscm_simple_screen import ARMS, DEFAULT_ROOT, DEFAULT_REPORT, summarize
from src.runners.run_rcr import atomic_json
from src.runners.run_local_residual_models import source_for


def audit(root, units):
    records = []
    for unit in units:
        source, _ = source_for(unit)
        mapping = json.loads((source / 'mapping.json').read_text())
        split = json.loads((root / 'cohorts' / unit / 'normal_split.json').read_text())
        assert not set(split['reconstruction_vins']) & set(split['calibration_vins'])
        assert split['seed'] == 3407
        for arm in ARMS:
            output = root / unit / arm
            assert (output / 'complete.json').exists(), (unit, arm)
            training = json.loads((output / 'training/complete.json').read_text())
            assert training['max_epochs'] == 30 and training['epochs_executed'] <= 30
            assert training['encoder_frozen'] and not training['test_labels_used']
            history = pd.read_csv(output / 'training/history.csv')
            initial = json.loads((output / 'training/initial.json').read_text())
            best_loss = min(initial['normal_validation_huber'], history.normal_validation_huber.min())
            np.testing.assert_allclose(training['normal_validation_huber'], best_loss, rtol=0, atol=1e-12)
            test, validation = read(output / 'test_scores.csv'), read(output / 'validation_scores.csv')
            assert validation.sample_label.eq(0).all()
            error = 0.
            for row in read(output / 'metrics.csv').itertuples():
                np.testing.assert_allclose(row.threshold, validation[row.method].quantile(row.q), rtol=1e-12)
                actual = metrics(test.sample_label, test[row.method], row.threshold)
                for key, value in actual.items():
                    error = max(error, abs(value - getattr(row, key)))
            assert error < 1e-10
            for partition in ['train', 'validation', 'test']:
                windows = np.load(output / f'{partition}_window_scores.npy', mmap_mode='r')
                assert windows.shape == (mapping['windows'][partition], 2)
                assert np.isfinite(windows).all()
            records.append(dict(unit=unit, arm=arm, metric_replay_max_error=error,
                best_epoch=training['best_epoch'], epochs=training['epochs_executed'],
                parameters=training['parameters'], normal_selection_verified=True,
                disjoint_reconstruction_calibration_vins_verified=True, complete_window_counts_verified=True))
        a3 = json.loads((root / unit / 'A3/training/complete.json').read_text())
        a5 = json.loads((root / unit / 'A5/training/complete.json').read_text())
        for key in ['parameters', 'trainable_parameters']:
            assert a3[key] == a5[key]
    pd.DataFrame(records).to_csv(root / 'audit_units.csv', index=False)
    return records


def paired_entries(root, units):
    entries = []
    for unit in units:
        frames, thresholds = {}, {}
        keys = ['vin', 'sample_label'] if unit.startswith('Brand3') else ['sample_id', 'vin', 'sample_label']
        for partition in ['validation', 'test']:
            combined = None
            for arm in ARMS:
                frame = read(root / unit / arm / f'{partition}_scores.csv')
                if combined is None:
                    combined = frame[keys].copy()
                else:
                    pd.testing.assert_frame_equal(combined[keys], frame[keys])
                for score in ['raw', 'cal_abs']:
                    combined[arm + '_' + score] = frame[score].to_numpy()
                t = json.loads((root / unit / arm / 'thresholds.json').read_text())
                thresholds.update({arm + '_' + score: t[score] for score in ['raw', 'cal_abs']})
            frames[partition] = combined
        entries.append((frames, thresholds))
    return entries


def figures(root, units, output):
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(units), figsize=(3.5 * len(units), 6), squeeze=False)
    for col, unit in enumerate(units):
        for kind in ['multi', 'single', 'single_T', 'multi_T']:
            f = pd.read_csv(root / unit / ('pretrain_' + kind) / 'history.csv')
            axes[0, col].plot(f.epoch, f.normal_validation_huber, label=kind)
        for arm in ARMS:
            f = pd.read_csv(root / unit / arm / 'training/history.csv')
            axes[1, col].plot(f.epoch, f.normal_validation_huber, label=arm)
        for row in [0, 1]:
            axes[row, col].set_yscale('log')
            axes[row, col].set_title(unit + (' pretrain' if row == 0 else ' frozen-encoder adaptation'))
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Normal validation Huber')
            axes[row, col].legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(output / ('normal_validation_curves.' + ext), dpi=160)
    plt.close(fig)
    rows = []
    for unit in units:
        for arm in ['A2', 'A2-T', 'A3', 'A3-T']:
            for partition in ['train', 'validation', 'test']:
                d = json.loads((root / unit / arm / f'{partition}_memory_diagnostics.json').read_text())
                rows.append(dict(unit=unit, arm=arm, partition=partition,
                    effective_soft_prototypes=d['effective_soft_prototypes'],
                    hard_used_prototypes=d['hard_used_prototypes'],
                    pairwise_cosine_mean=d['pairwise_cosine_mean'],
                    pairwise_distance_mean=d['pairwise_distance_mean'],
                    median_cosine_distance=d['cosine_distance_quantiles'][1]))
    pd.DataFrame(rows).to_csv(root / 'memory_diagnostics_summary.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT_ROOT)
    parser.add_argument('--report', default='report/analysis/rcr/MSCM_SIMPLE_SCREEN_20261025.md')
    parser.add_argument('--bootstrap', type=int, default=500)
    args = parser.parse_args()
    root, report = Path(args.output), Path(args.report)
    draft = Path(DEFAULT_REPORT)
    if draft.resolve() == report.resolve():
        draft = report.with_name(report.stem + '.generated' + report.suffix)
    protocol = json.loads((root / 'protocol.json').read_text())
    units = protocol['units']
    assert (root / 'complete.json').exists(), 'Wait for all requested training/evaluation units'
    checks = audit(root, units)
    summarize(root, units, draft)
    pairs = [(a + '_' + score, b + '_' + score)
             for score in ['raw', 'cal_abs']
             for a, b in [('A1', 'A3'), ('A2', 'A3'), ('A5', 'A3'),
                          ('A2', 'A2-T'), ('A3', 'A3-T'), ('A2-T', 'A3-T')]]
    intervals = []
    for dataset in ['LFP', 'NCM', 'Brand3']:
        selected = [u for u in units if ('Brand3' if u.startswith('Brand3') else u) == dataset]
        if not selected:
            continue
        entries = paired_entries(root, selected)
        intervals += paired_bootstrap(entries, pairs, args.bootstrap, dataset)
        print('SIMPLE VIN PAIRED COMPLETE', dataset, flush=True)
    ci = pd.DataFrame(intervals)
    ci.to_csv(root / 'paired_VIN_intervals.csv', index=False)
    figure_root = Path('report/figures/mscm_simple_screen_20261025')
    figures(root, units, figure_root)
    extra = ['', '## 配对不确定性与审核', '',
             f'全部 {len(checks)} 个实验单元通过完整窗口计数、正常验证选点、'
             '正常验证阈值和指标复算审核。A3/A5 参数量相等已逐单元核验。'
             '校准 VIN 未参与重构优化和 K-means；输入归一化仍沿用原正常训练集统计。', '',
             f'按 VIN 成组 bootstrap {args.bootstrap} 次，Brand3 同一 VIN 在不同折中共用抽样权重；'
             '区间反映本测试队列的抽样不确定性，不包括训练随机种子变异。完整 AP/AUROC/F1/Recall/FPR '
             '差值和区间见运行目录 `paired_VIN_intervals.csv`。', '',
             '| 比较（左减右） | 评分 | LFP ΔAUROC [95% CI] | NCM ΔAUROC [95% CI] | Brand3 ΔAUROC [95% CI] |',
             '| --- | --- | --- | --- | --- |']
    for a, b in [('A1', 'A3'), ('A2', 'A3'), ('A5', 'A3'),
                 ('A2', 'A2-T'), ('A3', 'A3-T'), ('A2-T', 'A3-T')]:
        for score in ['raw', 'cal_abs']:
            cells = []
            for dataset in ['LFP', 'NCM', 'Brand3']:
                q = ci.query('dataset == @dataset and base == @base and contender == @contender and metric == "auroc"',
                             local_dict=dict(dataset=dataset, base=a + '_' + score,
                                             contender=b + '_' + score))
                if q.empty:
                    cells.append('待补')
                else:
                    d = q.iloc[0]
                    cells.append(f'{d.delta:+.4f} [{d.ci_low:+.4f}, {d.ci_high:+.4f}]')
            extra.append(f'| {b} − {a} | {score} | ' + ' | '.join(cells) + ' |')
    extra += ['', f'正常验证训练曲线：[PNG](../../figures/{figure_root.name}/normal_validation_curves.png)。', '',
              '单种子筛选不得据测试最高 AP 选择训练轮数、尺度或阈值。'
              '后续是否保留 Transformer、是否进行模型二训练，应同时考虑原始残差、校准后分数、'
              'Memory 替换诊断和本轮预算边界，而不能只看某一个最优指标。', '']
    macro = pd.read_csv(root / 'macro_metrics.csv')
    extra += ['## 本轮筛选结论', '']
    for dataset in ['LFP', 'NCM', 'Brand3']:
        current = macro.query('dataset == @dataset and method == "raw"').set_index('arm')
        if current.empty:
            continue
        t = current.loc['A3-T', ['ap', 'auroc', 'f1']] - current.loc['A3', ['ap', 'auroc', 'f1']]
        ts = current.loc['A2-T', ['ap', 'auroc', 'f1']] - current.loc['A2', ['ap', 'auroc', 'f1']]
        scales = current.loc['A3-T', ['ap', 'auroc', 'f1']] - current.loc['A2-T', ['ap', 'auroc', 'f1']]
        m = current.loc['A3', ['ap', 'auroc', 'f1']] - current.loc['A5', ['ap', 'auroc', 'f1']]
        extra.append(f'- {dataset}：A3-T 相对 A3 的原始残差 ΔAP/ΔAUROC/ΔF1 为 '
                     f'{t.ap:+.4f}/{t.auroc:+.4f}/{t.f1:+.4f}；'
                     f'A2-T 相对 A2 为 {ts.ap:+.4f}/{ts.auroc:+.4f}/{ts.f1:+.4f}；'
                     f'A3-T 相对 A2-T 为 {scales.ap:+.4f}/{scales.auroc:+.4f}/{scales.f1:+.4f}；'
                     f'A3 相对同容量 A5 为 {m.ap:+.4f}/{m.auroc:+.4f}/{m.f1:+.4f}。')
    adapt = pd.read_csv(root / 'training_completion.csv')
    pre = pd.read_csv(root / 'pretraining_completion.csv')
    extra += ['', f'预训练最佳点达到20轮预算边界：{int(pre.at_budget_boundary.sum())}/{len(pre)}；'
              f'后续训练最佳点达到30轮预算边界：{int(adapt.at_budget_boundary.sum())}/{len(adapt)}。'
              '本轮仅按用户指定预算筛选，不自动延长。', '',
              '各原型都有占用或有效软原型数接近8，不能单独证明形成了有用的模式分工；'
              '均匀检索也可能出现这一现象。Memory 的必要性应由同容量对照、替换后的评分变化与配对区间共同判断。', '']
    launch = json.loads((root / 'launch.json').read_text())
    started = datetime.datetime.fromisoformat(launch['started_at'])
    finished = datetime.datetime.fromtimestamp((root / 'complete.json').stat().st_mtime, started.tzinfo)
    latest = datetime.datetime.fromisoformat(launch.get('resumed_at', launch['started_at']))
    seconds = (finished - latest).total_seconds()
    recorded_training_seconds = pre.seconds.sum() + adapt.seconds.sum()
    extra += [f'实验最初启动：{started:%Y-%m-%d %H:%M:%S}；本次恢复：{latest:%Y-%m-%d %H:%M:%S}；'
              f'训练/推理完成：{finished:%Y-%m-%d %H:%M:%S}（香港时间）。'
              f'本次恢复至完成约{seconds / 60:.1f}分钟；各独立训练阶段记录的 epoch 耗时累计'
              f'{recorded_training_seconds / 60:.1f}分钟（共享预训练只计一次）。'
              '实验跨越了中断间隔，不能把最初启动至最终完成的日历时间当作计算耗时；'
              '配对区间和最终绘图时间另计。', '',
              '模型二未提交 Kaggle；本轮报告只包含模型一。', '']
    draft.write_text(draft.read_text() + '\n'.join(extra))
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        archive = report.parent / 'archive'
        archive.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy2(report, archive / f'{report.stem}.{stamp}{report.suffix}')
    temporary = report.with_suffix(report.suffix + '.tmp')
    shutil.copyfile(draft, temporary)
    temporary.replace(report)
    atomic_json(root / 'analysis_complete.json', dict(requested_matrix_complete=True,
        audited_units=len(checks), grouped_bootstrap=args.bootstrap, report=str(report),
        weights_changed=False, new_training_started=False))
    print('SIMPLE ANALYSIS COMPLETE', report, flush=True)


if __name__ == '__main__':
    main()
