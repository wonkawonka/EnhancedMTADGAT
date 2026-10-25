"""Compare the complete neural development pipeline with the original RCR reference."""
import argparse
import json
import time
from pathlib import Path
from src.runners.report_metric_format import bold_best_metrics

import pandas as pd

from scripts.evaluate_rcr import metrics
from src.runners.analyze_mscm_development import PRIOR
from src.runners.probe_scale_relation_evidence import paired_bootstrap, read
from src.runners.run_rcr import atomic_json


METHODS = ["B1_original", "B2_gat", "B3_memory_gat", "B4_memory_mlp",
           "B3_q_only_gat", "B5_ECDF_0p5", "B5_MAD_0p5"]


def run(core, relations, repeats):
    assert (core / "model_one_analysis_complete.json").exists()
    assert (relations / "analysis_complete.json").exists()
    audit = json.loads((relations / "artifact_audit.json").read_text())
    assert audit["model_one_unchanged"] and all(audit["code_unchanged"].values())
    protocol = json.loads((relations / "protocol_manifest.json").read_text())
    units = protocol["units"]
    rows, intervals, datasets = [], [], ["LFP", "NCM", "Brand3"]
    for dataset in datasets:
        entries = []
        for unit in units:
            if (unit.startswith("Brand3") if dataset == "Brand3" else unit == dataset):
                keys = ["vin", "sample_label"] if dataset == "Brand3" else ["sample_id", "vin", "sample_label"]
                frames, thresholds = {}, {}
                for partition in ["validation", "test"]:
                    current = read(relations / unit / f"{partition}_scores.csv")
                    original = read(PRIOR / unit / f"{partition}_scores.csv")
                    pd.testing.assert_frame_equal(current[keys], original[keys])
                    current["original_RCR"] = original.A.to_numpy()
                    frames[partition] = current
                thresholds = json.loads((relations / unit / "thresholds.json").read_text())
                thresholds["original_RCR"] = {
                    str(q): float(frames["validation"].original_RCR.quantile(q)) for q in [.95, .99]
                }
                entries.append((frames, thresholds))
                for method in ["original_RCR", "B0"] + METHODS:
                    for q in [.95, .99]:
                        test = frames["test"]
                        rows.append(dict(dataset=dataset, unit=unit, method=method, q=q,
                            **metrics(test.sample_label, test[method], thresholds[method][str(q)])))
        for q in [.95, .99]:
            intervals += paired_bootstrap(entries, [("original_RCR", m) for m in METHODS],
                                          repeats, dataset, q)
        print("ORIGINAL RCR COMPARISON COMPLETE", dataset, flush=True)
    metrics_ = pd.DataFrame(rows)
    metrics_.to_csv(relations / "original_RCR_comparison_metrics.csv", index=False)
    ci = pd.DataFrame(intervals)
    ci.to_csv(relations / "original_RCR_paired_intervals.csv", index=False)
    mean = metrics_.query("q == .95").groupby(["dataset", "method"])[
        ["ap", "auroc", "f1", "recall", "fpr"]].mean()

    def cell(dataset, method):
        row = mean.loc[dataset, method]
        return f"{row.ap:.4f} / {row.auroc:.4f} / {row.f1:.4f}"

    def delta(dataset, method):
        row = ci.query("dataset == @dataset and contender == @method and metric == 'auroc' and q == .95").iloc[0]
        return f"{row.delta:+.4f} [{row.ci_low:+.4f}, {row.ci_high:+.4f}]"

    higher = [d for d in datasets if mean.loc[d, "B3_memory_gat"].auroc > mean.loc[d, "original_RCR"].auroc]
    lower = [d for d in datasets if mean.loc[d, "B3_memory_gat"].auroc < mean.loc[d, "original_RCR"].auroc]
    fusion_wins = [d for d in datasets if mean.loc[d, "B5_ECDF_0p5"].auroc > max(
        mean.loc[d, "B0"].auroc, mean.loc[d, "B3_memory_gat"].auroc)]
    outcome = ("按AUROC点估计，模型二B3高于原RCR的数据集为" + ("、".join(higher) or "无")
        + "，低于原RCR的数据集为" + ("、".join(lower) or "无") + "。预设ECDF等权融合超过两条单支的数据集为"
        + ("、".join(fusion_wins) or "无") + "。NCM的B3相对原RCR差值及95% VIN配对区间为"
        + delta("NCM", "B3_memory_gat") + "；B3、工况GAT、同条件MLP的AUROC分别为"
        + " / ".join(f'{mean.loc["NCM", m].auroc:.4f}' for m in
                     ["B3_memory_gat", "B3_q_only_gat", "B4_memory_mlp"]) + "。")
    lines = ["# MSCM-Net / MCR-GAT：首轮实现与RCR Scale参考对照", "",
        "按附件编号查看完整的[A0–A5、B0–B5结果表](MSCM_MCR_ABLATION_INDEX_20261025.md)。", "",
        "已完成两套新神经模型的首轮实现与训练：模型一35个核心重构单元，模型二35个关系头，覆盖CH的LFP/NCM和Brand3五折。以下为固定配置的开发验证，不是最终收敛或独立留出集结论。", "",
        "**基线来源：本表为RCR Scale校准幅值参考。CH沿用原检查点；Brand3使用RCR-MEM-BRAND3-20261025-v1本地重训检查点，未复用历史Kaggle权重。Brand3本地AP/AUROC/F1为0.8685/0.9156/0.6789；历史Scale为0.8728/0.9240/0.7236，属于不同训练结果。original_RCR字段及配对区间指本表来源，不能解释为与历史Kaggle权重的同检查点比较。历史报告指标顺序为AUROC/AP/F1/Recall/FPR，见[RCR主报告](RCR_REPORT_ZH.md)。**", "",
        outcome, "",
        "模型一：原始时序 → 多尺度卷积 → 各尺度正常Patch Memory → 有界补偿融合 → Transformer与Decoder重构 → 重新拟合的RCR后处理。", "",
        "模型二：冻结模型一 → 有符号校准残差 → 共享TCN变量编码 → 状态条件单层GAT → 严格LOO序列预测 → 关系评分，并单独评价与边缘评分的固定等权融合。", "",
        "本报告补充与上述RCR Scale参考的逐对象配对比较，防止仅超过较弱的A4父模型就被解释为超过对应基线。主模型仍按协议固定为A4、B3和ECDF等权融合，没有根据测试结果替换分支。", "",
        "| 固定方法 | LFP AP / AUROC / F1 | NCM AP / AUROC / F1 | Brand3 AP / AUROC / F1 |",
        "| --- | --- | --- | --- |"]
    labels = [("original_RCR", "RCR Scale参考（CH原检查点；Brand3本地重训）"), ("B0", "模型一A4 / 固定稳健尺度校准"),
        ("B1_original", "原RCR关系头（新残差重训）"), ("B2_gat", "普通GAT"),
        ("B3_memory_gat", "模型二B3 / Memory先验条件GAT"),
        ("B4_memory_mlp", "同条件无图MLP"), ("B3_q_only_gat", "同参数工况条件GAT"),
        ("B5_ECDF_0p5", "预设ECDF等权融合"), ("B5_MAD_0p5", "连续MAD等权融合")]
    for method, label in labels:
        lines.append("| " + label + " | " + " | ".join(cell(d, method) for d in datasets) + " |")
    lines += ["", "Brand3为五折算术均值；F1阈值取各方法正常验证P95，不能据此假设测试FPR相同。", "",
        "| 数据集 | 对应来源RCR Scale参考 → 固定新方法 | AUROC差值 [95% VIN配对区间] |",
        "| --- | --- | --- |"]
    for dataset in datasets:
        for method in METHODS:
            lines.append(f"| {dataset} | {method} | {delta(dataset, method)} |")
    lines += ["", "模型一的证据：LFP的多尺度无Memory版本有明显收益；NCM的Memory相对普通瓶颈有收益，但多尺度叠加与超过完整原RCR的证据不足；Brand3的Memory版本明显退化。原型占用较分散，不能把性能退化简单归因于只剩一个原型。CH的组件干预还显示有界补偿路径对输出很重要，γ=0.1本身不保证Memory主导。", "",
        "模型二是否提供互补信息，应同时看与冻结A4的配对差值、仅关系分支补检的故障与新增正常误报，以及本表相对原RCR的差值。Memory图结构的额外价值需要B3超过普通GAT、同条件MLP和同参数工况GAT；仅超过A4不能证明这一点。", "",
        "边界：重构阶段各最多5+5个epoch，关系头最多10个epoch；正常验证选模型。A4各单元的最佳正常验证点都在阶段B预算上限，不能宣称最终收敛。模型二首版使用由电流/SOC预测正常Memory占用的固定先验；它是工况q的确定函数，不是独立的新观测，完整多变量遮挡Memory条件尚未运行。", "",
        f"不确定性：{repeats}次标签分层VIN成组配对bootstrap，seed3407。跨折重复VIN共享权重；固定模型和正常验证阈值，未包含重训练或阈值估计不确定性，未作多重比较校正。", "",
        "已通过16项新CPU测试及CUDA前向/反向检查；两套全量审计记录指标复算、源码签名、父模型冻结、评分继承和目标遮挡。纯Memory/直接融合、λ、Patch及编码器微调入口已实现但未运行。本轮未修改论文最终四张实验矩阵。", "",
        "详细报告：[模型一](MSCM_MODEL_ONE_CORE_20261025.md)、[模型二](MCR_GAT_MODEL_TWO_20261025.md)；实现协议：[开发协议](../../system_design/MSCM_MCR_DEVELOPMENT_20261025.md)。", ""]
    report = Path("report/analysis/rcr/MSCM_MCR_IMPLEMENTATION_RESULTS_20261025.md")
    report.write_text(bold_best_metrics("\n".join(lines)))
    atomic_json(relations / "original_RCR_comparison_complete.json", dict(
        report=str(report), units=units, repeats=repeats,
        pairing="exact matching decision-object keys; original normal-validation thresholds",
        training_changed=False, test_selection=False))
    print("PIPELINE SUMMARY COMPLETE", report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", default="runs/internal/mscm_mcr_development_20261025_v2")
    parser.add_argument("--relations", default="runs/internal/mcr_gat_development_20261025_v1")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--wait-for-analysis", action="store_true")
    args = parser.parse_args()
    core, relations = Path(args.core), Path(args.relations)
    if args.wait_for_analysis:
        while not (relations / "analysis_complete.json").exists():
            time.sleep(10)
    run(core, relations, args.bootstrap)


if __name__ == "__main__":
    main()
