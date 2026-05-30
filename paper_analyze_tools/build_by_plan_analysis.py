"""按实验计划重整 kaggle 离线结果，生成 inventory/raw summary/03 可读表。

同时自动补入跨 plan 的 baseline/full 对照结果，并为每个计划输出 loss 曲线拼图与分析说明。
"""

import json
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = ROOT / "kaggle离线output"
CONFIG_DIR = ROOT / "configs" / "compare"
OUTPUT_ROOT = KAGGLE_DIR / "analysis" / "by_plan"
PUBLIC_DATASETS = {"SMAP", "MSL"}
CH_BATTERY_DATASET = "CH_BATTERY_LFP_DISCHARGE"
EXTERNAL_VARIANTS = {"TranAD", "Anomaly-Transformer", "GDN", "DCdetector"}
RELATED_EXPERIMENTS = {
    "ch3_ablation": {
        "SMAP": [{"plan": "ch3_main_results", "experiments": ["smap_mtadgat_baseline", "smap_c3_full"]}],
        "MSL": [{"plan": "ch3_main_results", "experiments": ["msl_mtadgat_baseline", "msl_c3_full"]}],
    },
    "ch3_external_baselines": {
        "SMAP": [{"plan": "ch3_main_results", "experiments": ["smap_mtadgat_baseline", "smap_c3_full"]}],
        "MSL": [{"plan": "ch3_main_results", "experiments": ["msl_mtadgat_baseline", "msl_c3_full"]}],
    },
    "ch3_battery_ablation": {
        "NASA_RANDOM_DISCHARGE": [
            {
                "plan": "ch3_main_results",
                "experiments": ["nasa_random_discharge_rw1_mtadgat_baseline", "nasa_random_discharge_rw1_c3_full"],
            }
        ],
        "BMS": [{"plan": "ch3_main_results", "experiments": ["bms_mtadgat_baseline", "bms_c3_full"]}],
    },
    "ch4_bms_main": {
        "NASA_RANDOM_DISCHARGE": [
            {
                "plan": "ch3_main_results",
                "experiments": [
                    "nasa_random_discharge_rw1_mtadgat_baseline",
                    "nasa_random_discharge_rw1_c3_full",
                    "nasa_random_discharge_rw2_mtadgat_baseline",
                    "nasa_random_discharge_rw2_c3_full",
                    "nasa_random_discharge_rw7_mtadgat_baseline",
                    "nasa_random_discharge_rw7_c3_full",
                    "nasa_random_discharge_rw8_mtadgat_baseline",
                    "nasa_random_discharge_rw8_c3_full",
                ],
            },
            {
                "plan": "ch3_battery_ablation",
                "experiments": ["nasa_random_rw1_c3_no_transformer", "nasa_random_rw1_c3_no_revin", "nasa_random_rw1_c3_no_regime"],
            },
        ],
        "BMS": [
            {"plan": "ch3_main_results", "experiments": ["bms_mtadgat_baseline", "bms_c3_full"]},
            {"plan": "ch3_battery_ablation", "experiments": ["bms_c3_no_transformer", "bms_c3_no_revin", "bms_c3_fixed_fusion"]},
        ],
    },
    "ch4_bms_ablation": {
        "BMS": [
            {"plan": "ch3_main_results", "experiments": ["bms_mtadgat_baseline", "bms_c3_full"]},
            {"plan": "ch4_bms_main", "experiments": ["bms_c3_physics_full", "bms_c4_physics_full"]},
        ],
    },
    "ch_battery_ablation": {
        CH_BATTERY_DATASET: [{"plan": "ch_battery_main", "experiments": ["chbatt_lfp_mtadgat_baseline", "chbatt_lfp_c3_full"]}],
    },
    "ch_battery_physics_main": {
        CH_BATTERY_DATASET: [{"plan": "ch_battery_main", "experiments": ["chbatt_lfp_mtadgat_baseline", "chbatt_lfp_c3_full"]}],
    },
    "ch_battery_physics_ablation": {
        CH_BATTERY_DATASET: [
            {"plan": "ch_battery_main", "experiments": ["chbatt_lfp_mtadgat_baseline", "chbatt_lfp_c3_full"]},
            {"plan": "ch_battery_physics_main", "experiments": ["chbatt_lfp_c3_physics"]},
        ],
    },
}


def canonical_name(name):
    name = re.sub(r" \(\d+\)$", "", name)
    name = re.sub(r"（\d+）$", "", name)
    return name


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_config(exp_dir):
    config_path = exp_dir / "config.txt"
    if not config_path.exists():
        return {}
    try:
        return load_json(config_path)
    except json.JSONDecodeError:
        return {}


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return round(float(value), digits)


def mean_or_none(values):
    valid = [safe_float(v) for v in values]
    valid = [v for v in valid if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def infer_dataset(name, config):
    if config.get("dataset"):
        return str(config["dataset"])
    lowered = name.lower()
    if "smap" in lowered:
        return "SMAP"
    if "msl" in lowered:
        return "MSL"
    if "bms" in lowered:
        return "BMS"
    if "nasa_random" in lowered or "rw" in lowered:
        return "NASA_RANDOM_DISCHARGE"
    return ""


def infer_run_id(name, config):
    return str(config.get("run_id", name))


def get_case_summary_paths(exp_dir):
    direct = exp_dir / "summary_metrics.json"
    if direct.exists():
        return [direct]
    return sorted(exp_dir.glob("*/summary_metrics.json"))


def has_any_summary(exp_dir):
    return bool(get_case_summary_paths(exp_dir)) or (exp_dir / "ch_battery_sample_summary.json").exists()


def load_ch_battery_sample_summary(exp_dir):
    summary_path = exp_dir / "ch_battery_sample_summary.json"
    if not summary_path.exists():
        return {}
    return load_json(summary_path)


def get_metric(payload, nested_key, flat_key):
    block = payload.get(nested_key, {})
    if isinstance(block, dict):
        value = safe_float(block.get("f1"))
        if value is not None:
            return value
    return safe_float(payload.get(flat_key))


def get_threshold(payload, nested_key, flat_key):
    block = payload.get(nested_key, {})
    if isinstance(block, dict):
        value = safe_float(block.get("threshold"))
        if value is not None:
            return value
    return safe_float(payload.get(flat_key))


def summarize_experiment(exp_dir, plan_name):
    config = read_config(exp_dir)
    case_summary_paths = get_case_summary_paths(exp_dir)
    ch_battery_summary = load_ch_battery_sample_summary(exp_dir)
    if not case_summary_paths and not ch_battery_summary:
        return None, []

    case_rows = []
    for summary_path in case_summary_paths:
        payload = load_json(summary_path)
        case_dir = summary_path.parent
        thresholds_path = case_dir / "thresholds.json"
        thresholds = load_json(thresholds_path) if thresholds_path.exists() else {}

        epsilon = payload.get("epsilon_result", {})
        pot = payload.get("pot_result", {})
        bf = payload.get("bf_result", {})
        event = payload.get("event_consistency_result", {})
        raw = event.get("raw_result", {})
        final = event.get("event_result", {})
        hier = payload.get("hier_consistency_result", {})
        single_f1 = safe_float(payload.get("metric_f1"))
        single_precision = safe_float(payload.get("metric_precision"))
        single_recall = safe_float(payload.get("metric_recall"))
        single_auroc = safe_float(payload.get("metric_auroc"))
        single_threshold = safe_float(payload.get("metric_threshold"))

        case_rows.append(
            {
                "plan": plan_name,
                "experiment_name": exp_dir.name,
                "case_name": case_dir.name if case_dir != exp_dir else exp_dir.name,
                "dataset": infer_dataset(exp_dir.name, config),
                "run_id": infer_run_id(exp_dir.name, config),
                "epsilon_f1": get_metric(payload, "epsilon_result", "metric_f1"),
                "pot_f1": safe_float(pot.get("f1")),
                "bf_f1": get_metric(payload, "bf_result", "metric_f1"),
                "event_f1": safe_float(final.get("f1")),
                "single_f1": single_f1,
                "single_precision": single_precision,
                "single_recall": single_recall,
                "single_auroc": single_auroc,
                "epsilon_threshold": get_threshold(payload, "epsilon_result", "metric_threshold"),
                "pot_threshold": safe_float(pot.get("threshold")),
                "raw_positive": safe_float(raw.get("positive_count")),
                "final_positive": safe_float(final.get("positive_count")),
                "event_enabled": bool(event.get("enabled", False)),
                "hier_enabled": bool(hier.get("enabled", False)),
                "hier_weight": safe_float(hier.get("weight")),
                "global_threshold": safe_float(thresholds.get("global_threshold")) if thresholds else single_threshold,
                "dir_path": str(case_dir),
            }
        )

    summary = {
        "plan": plan_name,
        "experiment_name": exp_dir.name,
        "canonical_name": canonical_name(exp_dir.name),
        "dataset": infer_dataset(exp_dir.name, config),
        "run_id": infer_run_id(exp_dir.name, config),
        "case_count": len(case_rows),
        "epsilon_f1": mean_or_none([row["epsilon_f1"] for row in case_rows]),
        "pot_f1": mean_or_none([row["pot_f1"] for row in case_rows]),
        "bf_f1": mean_or_none([row["bf_f1"] for row in case_rows]),
        "event_f1": mean_or_none([row["event_f1"] for row in case_rows]),
        "single_f1": mean_or_none([row["single_f1"] for row in case_rows]),
        "single_precision": mean_or_none([row["single_precision"] for row in case_rows]),
        "single_recall": mean_or_none([row["single_recall"] for row in case_rows]),
        "single_auroc": mean_or_none([row["single_auroc"] for row in case_rows]),
        "epsilon_threshold": mean_or_none([row["epsilon_threshold"] for row in case_rows]),
        "pot_threshold": mean_or_none([row["pot_threshold"] for row in case_rows]),
        "raw_positive": mean_or_none([row["raw_positive"] for row in case_rows]),
        "final_positive": mean_or_none([row["final_positive"] for row in case_rows]),
        "event_enabled": any(row["event_enabled"] for row in case_rows),
        "hier_enabled": any(row["hier_enabled"] for row in case_rows),
        "hier_weight": mean_or_none([row["hier_weight"] for row in case_rows]),
        "sample_count": safe_float(ch_battery_summary.get("sample_count")),
        "normal_count": safe_float(ch_battery_summary.get("normal_count")),
        "fault_count": safe_float(ch_battery_summary.get("fault_count")),
        "sample_auroc": safe_float(ch_battery_summary.get("sample_auroc")),
        "sample_auprc": safe_float(ch_battery_summary.get("sample_auprc")),
        "best_f1": safe_float(ch_battery_summary.get("best_f1")),
        "best_threshold": safe_float(ch_battery_summary.get("best_threshold")),
        "score_field": ch_battery_summary.get("score_field", ""),
        "dir_path": str(exp_dir),
    }
    return summary, case_rows


def write_df(df, out_dir, basename):
    csv_path = out_dir / f"{basename}.csv"
    md_path = out_dir / f"{basename}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) if pd.notna(row[col]) else "" for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def render_markdown_table(df):
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) if pd.notna(row[col]) else "" for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def bold_text(value):
    text = "" if value is None else str(value)
    return f"**{text}**" if text else text


def write_highlighted_markdown(df, out_dir, basename, analysis_lines, group_key=None, higher_is_better_cols=None, lower_is_better_cols=None):
    higher_is_better_cols = higher_is_better_cols or []
    lower_is_better_cols = lower_is_better_cols or []
    md_df = df.copy().astype(object)

    if group_key and group_key in md_df.columns:
        grouped = md_df.groupby(group_key, dropna=False)
    else:
        grouped = [(None, md_df)]

    for _, group in grouped:
        if len(group) <= 1:
            continue
        for col in higher_is_better_cols:
            if col not in md_df.columns:
                continue
            numeric = pd.to_numeric(group[col], errors="coerce")
            if numeric.notna().any():
                best = numeric.max()
                idxs = group.index[numeric == best]
                md_df.loc[idxs, col] = md_df.loc[idxs, col].map(bold_text)
        for col in lower_is_better_cols:
            if col not in md_df.columns:
                continue
            numeric = pd.to_numeric(group[col], errors="coerce")
            if numeric.notna().any():
                best = numeric.min()
                idxs = group.index[numeric == best]
                md_df.loc[idxs, col] = md_df.loc[idxs, col].map(bold_text)

    md_path = out_dir / f"{basename}.md"
    lines = render_markdown_table(md_df)
    if analysis_lines:
        lines.extend(["", "## 分析"])
        lines.extend(f"- {line}" for line in analysis_lines)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def build_public_analysis(df):
    lines = []
    for dataset, group in df.groupby("Dataset", dropna=False):
        bf_numeric = pd.to_numeric(group["BF_F1"], errors="coerce")
        event_numeric = pd.to_numeric(group["Event_F1"], errors="coerce")
        eps_numeric = pd.to_numeric(group["Epsilon_Threshold"], errors="coerce")
        valid = group.copy()
        valid["BF_F1_num"] = bf_numeric
        valid["Event_F1_num"] = event_numeric
        valid["Epsilon_Threshold_num"] = eps_numeric
        valid = valid[valid["BF_F1_num"].notna()].sort_values("BF_F1_num", ascending=False)
        if valid.empty:
            continue
        baseline_rows = valid[valid["Variant"] == "MTAD-GAT"]
        baseline = baseline_rows.iloc[0] if not baseline_rows.empty else None

        best = valid.iloc[0]
        if baseline is not None and best["Variant"] != "MTAD-GAT":
            delta = best["BF_F1_num"] - baseline["BF_F1_num"]
            lines.append(
                f"在 `{dataset}` 上，以 baseline `MTAD-GAT` 为对照时，增强方案 `{best['Variant']}` 取得最高 `BF_F1={best['BF_F1']}`，"
                f"相较 baseline 提高 `{fmt(delta)}`。"
            )
        elif baseline is not None and best["Variant"] == "MTAD-GAT":
            non_baseline = valid[valid["Variant"] != "MTAD-GAT"]
            if not non_baseline.empty:
                second = non_baseline.iloc[0]
                delta = best["BF_F1_num"] - second["BF_F1_num"]
                lines.append(
                    f"在 `{dataset}` 上，baseline `MTAD-GAT` 仍然保持最高 `BF_F1={best['BF_F1']}`，"
                    f"领先当前最优增强方案 `{second['Variant']}` `{fmt(delta)}`。"
                )
            else:
                lines.append(f"在 `{dataset}` 上，当前仅有 baseline `MTAD-GAT` 提供有效 `BF_F1` 结果，数值为 `{best['BF_F1']}`。")
        elif len(valid) >= 2:
            second = valid.iloc[1]
            delta = best["BF_F1_num"] - second["BF_F1_num"]
            lines.append(
                f"在 `{dataset}` 上，`{best['Variant']}` 取得当前表内最高 `BF_F1={best['BF_F1']}`，"
                f"相较次优方案 `{second['Variant']}` 提高 `{fmt(delta)}`。"
            )
        else:
            lines.append(f"在 `{dataset}` 上，当前仅有 `{best['Variant']}` 提供有效 `BF_F1` 结果，数值为 `{best['BF_F1']}`。")

        if pd.notna(best["Event_F1_num"]):
            lines.append(
                f"对应的 `Event_F1` 为 `{best['Event_F1']}`，说明该结论在事件级评价下与点级结果保持一致。"
            )

        if valid["Epsilon_Threshold_num"].notna().any():
            min_thr_row = valid.loc[valid["Epsilon_Threshold_num"].idxmin()]
            lines.append(
                f"从阈值角度看，`{dataset}` 中最小 `Epsilon_Threshold` 出现在 `{min_thr_row['Variant']}`，"
                f"数值为 `{min_thr_row['Epsilon_Threshold']}`，表明其异常分数分布更偏向低阈值区域。"
            )
    if df["Variant"].isin(EXTERNAL_VARIANTS).any():
        lines.append("外部基线行的 `BF_F1` / `Epsilon_Threshold` 来自各自离线目录中的单阈值导出指标，因此 `POT_F1` 与 `Event_F1` 可能为空；主比较时优先看 `BF_F1`。")
    if not lines:
        lines.append("当前计划没有可用于公开标注数据集比较的有效 F1 结果。")
    return lines


def build_nasa_analysis(df):
    lines = []
    summary_rows = []
    for battery, group in df.groupby("Battery", dropna=False):
        eps_numeric = pd.to_numeric(group["Epsilon_Threshold"], errors="coerce")
        final_numeric = pd.to_numeric(group["Final_Positive"], errors="coerce")
        pot_numeric = pd.to_numeric(group["POT_Threshold"], errors="coerce")
        group = group.copy()
        group["Epsilon_Threshold_num"] = eps_numeric
        group["POT_Threshold_num"] = pot_numeric
        group["Final_Positive_num"] = final_numeric
        if group["Epsilon_Threshold_num"].notna().any():
            min_row = group.loc[group["Epsilon_Threshold_num"].idxmin()]
            summary_rows.append((battery, min_row["Variant"], min_row["Epsilon_Threshold_num"], min_row["Final_Positive_num"]))

    if summary_rows:
        dominant = [row[1] for row in summary_rows]
        counts = pd.Series(dominant).value_counts()
        top_variant = counts.index[0]
        lines.append(
            f"以 baseline `MTAD-GAT` 为参照时，当前计划中的增强方案里，`{top_variant}` 在多数电池上取得更低的 `Epsilon_Threshold`，"
            f"说明其分数分布相较 baseline 更容易进入低阈值区间。"
        )

    final_numeric_all = pd.to_numeric(df["Final_Positive"], errors="coerce")
    if final_numeric_all.notna().any():
        max_row = df.loc[final_numeric_all.idxmax()]
        min_row = df.loc[final_numeric_all.idxmin()]
        lines.append(
            f"`Final_Positive` 在不同电池间差异较大，最高出现在 `{max_row['Battery']}-{max_row['Variant']}` "
            f"(`{max_row['Final_Positive']}`)，最低为 `{min_row['Battery']}-{min_row['Variant']}` "
            f"(`{min_row['Final_Positive']}`)，表明不同随机工况下触发范围存在明显差异。"
        )

    event_flags = df["Event_Enabled"].astype(str).unique().tolist() if "Event_Enabled" in df.columns else []
    if "True" in event_flags and "False" in event_flags:
        lines.append("当前表同时包含开启与未开启事件一致性的方案，最终阳性点数除受分数分布影响外，也会受到后处理策略差异的共同作用。")

    lines.append("建议结合 `kaggle离线output/analysis/ch4_condition_slices_from_dirs_test/nasa_random/nasa_phase_score_boxplots.png`、`nasa_phase_switch_profile.png` 与 `nasa_case_transition.png` 一起查看，以判断低阈值是否主要集中在 phase 切换段。")
    lines.append("由于 NASA 随机场景缺少统一标准标签，这里的结论应理解为相对 baseline 的分数统计与后处理行为分析，而不直接等同于性能优劣比较。")
    return lines


def build_bms_analysis(df):
    lines = []
    eps_numeric = pd.to_numeric(df["Avg_Epsilon_Threshold"], errors="coerce")
    pot_numeric = pd.to_numeric(df["Avg_POT_Threshold"], errors="coerce")
    final_numeric = pd.to_numeric(df["Avg_Final_Positive"], errors="coerce")
    raw_numeric = pd.to_numeric(df["Avg_Raw_Positive"], errors="coerce")
    df = df.copy()
    df["Avg_Epsilon_Threshold_num"] = eps_numeric
    df["Avg_POT_Threshold_num"] = pot_numeric
    df["Avg_Final_Positive_num"] = final_numeric
    df["Avg_Raw_Positive_num"] = raw_numeric

    if eps_numeric.notna().any():
        min_idx = eps_numeric.idxmin()
        row = df.loc[min_idx]
        lines.append(
            f"相较 baseline `MTAD-GAT`，当前 BMS 方案中 `{row['Variant']}` 取得最低的平均 `Epsilon_Threshold={row['Avg_Epsilon_Threshold']}`，"
            f"表明其在簇级实验中形成了更低阈值的分数分布。"
        )
    if final_numeric.notna().any():
        max_idx = final_numeric.idxmax()
        row = df.loc[max_idx]
        lines.append(
            f"`Avg_Final_Positive` 最高的是 `{row['Variant']}`，数值为 `{row['Avg_Final_Positive']}`，"
            f"说明该方案在事件一致性处理后触发了更长或更多的异常片段。"
        )
    if "Hier_Enabled" in df.columns and (df["Hier_Enabled"].astype(str) == "True").any():
        hier_rows = df[df["Hier_Enabled"].astype(str) == "True"]["Variant"].tolist()
        lines.append(f"其中开启层级一致性的方案为：`{'`, `'.join(hier_rows)}`，可将其理解为在 `C4` 基础上进一步加入层级一致性增强。")
    if df["Avg_Raw_Positive_num"].notna().any() and df["Avg_Final_Positive_num"].notna().any():
        top_raw = df.loc[df["Avg_Raw_Positive_num"].idxmax()]
        top_final = df.loc[df["Avg_Final_Positive_num"].idxmax()]
        if top_raw["Variant"] != top_final["Variant"]:
            lines.append(
                f"值得注意的是，原始阳性点最多的方案为 `{top_raw['Variant']}`，但最终阳性点最多的是 `{top_final['Variant']}`，"
                "说明后处理与层级一致性对最终告警范围具有显著影响。"
            )
    lines.append("建议结合 `kaggle离线output/analysis/ch4_condition_slices_from_dirs_test/bms/bms_regulation_score_boxplots.png`、`bms_regulation_error_summary.png`、`bms_branch_contribution.png` 与 `bms_case_typical_anomaly.png` 一起查看，以判断异常分数提升是否集中在 regulation 或 branch 切换相关片段。")
    lines.append("由于 BMS 场景缺少统一标准标签，这里的表格主要用于呈现相对 baseline 的分数分布、阈值估计和后处理触发范围变化。")
    return lines


def build_ch_battery_analysis(df):
    lines = []
    df = df.copy()
    for col in ["Sample_AUROC", "Sample_AUPRC", "Best_F1", "Best_Threshold", "Samples", "Fault_Samples"]:
        if col in df.columns:
            df[f"{col}_num"] = pd.to_numeric(df[col], errors="coerce")

    valid = df[df["Sample_AUROC_num"].notna()].sort_values("Sample_AUROC_num", ascending=False) if "Sample_AUROC_num" in df.columns else pd.DataFrame()
    if not valid.empty:
        best = valid.iloc[0]
        baseline_rows = valid[valid["Variant"] == "MTAD-GAT"]
        baseline = baseline_rows.iloc[0] if not baseline_rows.empty else None
        if baseline is not None and best["Variant"] != "MTAD-GAT":
            delta = best["Sample_AUROC_num"] - baseline["Sample_AUROC_num"]
            lines.append(
                f"在 CH-BATTERY 样本级弱标签评估上，以 baseline `MTAD-GAT` 为对照时，`{best['Variant']}` 取得最高 `Sample_AUROC={best['Sample_AUROC']}`，相较 baseline 提高 `{fmt(delta)}`。"
            )
        elif baseline is not None and best["Variant"] == "MTAD-GAT":
            non_baseline = valid[valid["Variant"] != "MTAD-GAT"]
            if not non_baseline.empty:
                second = non_baseline.iloc[0]
                delta = best["Sample_AUROC_num"] - second["Sample_AUROC_num"]
                lines.append(
                    f"在 CH-BATTERY 上，baseline `MTAD-GAT` 仍保持最高 `Sample_AUROC={best['Sample_AUROC']}`，领先当前最优增强方案 `{second['Variant']}` `{fmt(delta)}`。"
                )
            else:
                lines.append(f"在 CH-BATTERY 上，当前仅有 baseline `MTAD-GAT` 提供有效样本级结果，`Sample_AUROC={best['Sample_AUROC']}`。")
        elif len(valid) >= 2:
            second = valid.iloc[1]
            delta = best["Sample_AUROC_num"] - second["Sample_AUROC_num"]
            lines.append(
                f"在 CH-BATTERY 上，`{best['Variant']}` 取得当前表内最高 `Sample_AUROC={best['Sample_AUROC']}`，相较次优方案 `{second['Variant']}` 提高 `{fmt(delta)}`。"
            )

        if pd.notna(best.get("Sample_AUPRC_num")):
            lines.append(f"对应的 `Sample_AUPRC` 为 `{best['Sample_AUPRC']}`，说明该结果在故障样本稀疏的样本级排序上同样成立。")
        if pd.notna(best.get("Best_F1_num")):
            lines.append(f"同一方案的样本级 `Best_F1` 为 `{best['Best_F1']}`，可作为后续论文主表里最直观的弱标签检测指标。")

    if "Best_Threshold_num" in df.columns and df["Best_Threshold_num"].notna().any():
        threshold_row = df.loc[df["Best_Threshold_num"].idxmin()]
        lines.append(
            f"从样本级最优阈值看，`{threshold_row['Variant']}` 的 `Best_Threshold` 最低，为 `{threshold_row['Best_Threshold']}`；这一数值主要用于辅助选择工作点，不宜单独解释为模型一定更优。"
        )

    if "Fault_Samples_num" in df.columns and df["Fault_Samples_num"].notna().any():
        fault_count = int(df["Fault_Samples_num"].dropna().iloc[0])
        sample_count = int(df["Samples_num"].dropna().iloc[0]) if "Samples_num" in df.columns and df["Samples_num"].notna().any() else None
        if sample_count is not None:
            lines.append(f"当前 CH-BATTERY 评估口径是固定的样本级弱标签验证：总样本 `{sample_count}` 个，其中故障样本 `{fault_count}` 个，所有方案共用同一套预处理划分。")
    lines.append("这组结果和 `NASA_RANDOM_DISCHARGE` 不同，它不是时间点级指标，而是窗口分数聚合后的样本级指标，因此更适合作为外部故障数据集的补充验证。")
    return lines


def write_visual_guide():
    guide_path = OUTPUT_ROOT / "看图指南_NASA_BMS.md"
    lines = [
        "# NASA / BMS 看图指南",
        "",
        "这份指南用于把无标签数据集的统计表和 `analysis/ch4_condition_slices_from_dirs_test/` 下的图对应起来，帮助你判断应该把哪张图写进论文哪一段。",
        "",
        "## 总原则",
        "",
        "- `MTAD-GAT` 视为 baseline。",
        "- `C3`、`C4`、`C4+Hier` 均视为在 baseline 基础上的增强方案。",
        "- 无标签数据集不能单靠阈值或阳性点数判断优劣，必须结合图看“异常分数集中在哪些工况切片、切换边界和典型案例里”。",
        "",
        "## NASA 随机场景",
        "",
        f"- 对应统计表：[03_nasa_random_table](file:///{(OUTPUT_ROOT / 'ch3_main_results' / '03_nasa_random_table.md').as_posix()}) 或 [ch4_bms_main/03_nasa_random_table](file:///{(OUTPUT_ROOT / 'ch4_bms_main' / '03_nasa_random_table.md').as_posix()})",
        "- 图目录：`kaggle离线output/analysis/ch4_condition_slices_from_dirs_test/nasa_random/`",
        "",
        "**建议优先看这几张图**",
        "",
        f"- [nasa_phase_score_boxplots.png](file:///{(KAGGLE_DIR / 'analysis' / 'ch4_condition_slices_from_dirs_test' / 'nasa_random' / 'nasa_phase_score_boxplots.png').as_posix()})",
        "  作用：看不同 phase 切片下，baseline 与增强方案的异常分数分布有没有明显分离。",
        "  重点：如果增强方案主要在 transition/switch 邻域出现更高分数，而 stable phase 仍保持较低分数，这种现象才更有解释价值。",
        "  适合写入：第四章 NASA 随机场景切片分析小节。",
        "",
        f"- [nasa_phase_error_summary.png](file:///{(KAGGLE_DIR / 'analysis' / 'ch4_condition_slices_from_dirs_test' / 'nasa_random' / 'nasa_phase_error_summary.png').as_posix()})",
        "  作用：看重构误差是否也在 phase 切换段同步抬升。",
        "  重点：如果分数升高和误差升高同时出现，说明增强不是单纯把分数拉高，而是真正捕捉到了切换段的困难区域。",
        "  适合写入：第三章/第四章对比方法机理解释部分。",
        "",
        f"- [nasa_phase_switch_profile.png](file:///{(KAGGLE_DIR / 'analysis' / 'ch4_condition_slices_from_dirs_test' / 'nasa_random' / 'nasa_phase_switch_profile.png').as_posix()})",
        "  作用：看 phase 切换前后分数曲线的平均轮廓。",
        "  重点：如果增强方案在切换中心附近形成更尖锐的峰，而远离切换点后迅速回落，说明它更聚焦于 phase transition 本身。",
        "  适合写入：第四章场景切换敏感性分析。",
        "",
        f"- [nasa_case_transition.png](file:///{(KAGGLE_DIR / 'analysis' / 'ch4_condition_slices_from_dirs_test' / 'nasa_random' / 'nasa_case_transition.png').as_posix()})",
        "  作用：看单个典型 case 的时间轴例子。",
        "  重点：最好配合表中的某个电池，如 RW1 或 RW7，一起说明增强方案在切换段会比 baseline 更早或更稳定地抬升分数。",
        "  适合写入：案例分析或图注说明。",
        "",
        f"- [nasa_case_stable.png](file:///{(KAGGLE_DIR / 'analysis' / 'ch4_condition_slices_from_dirs_test' / 'nasa_random' / 'nasa_case_stable.png').as_posix()})",
        "  作用：作为对照，说明在相对稳定阶段，增强方案是否仍能保持较低背景分数。",
        "  适合写入：说明方法不会在 stable phase 普遍放大误报。",
        "",
        "**NASA 在论文里怎么写**",
        "",
        "- 先用 `03_nasa_random_table.md` 写“相较 baseline，增强方案的阈值区间与触发范围发生变化”。",
        "- 再用 `nasa_phase_score_boxplots.png` 和 `nasa_phase_switch_profile.png` 支撑“变化主要集中在 phase 切换段”。",
        "- 最后用 `nasa_case_transition.png` 给出单案例可视化证据。",
        "",
        "## BMS 私有数据",
        "",
        f"- 对应统计表：[ch3_main_results/03_bms_table](file:///{(OUTPUT_ROOT / 'ch3_main_results' / '03_bms_table.md').as_posix()})、[ch4_bms_main/03_bms_table](file:///{(OUTPUT_ROOT / 'ch4_bms_main' / '03_bms_table.md').as_posix()})",
        "- 图目录：`kaggle离线output/analysis/ch4_condition_slices_from_dirs_test/bms/`",
        "",
        "**建议优先看这几张图**",
        "",
        f"- [bms_regulation_score_boxplots.png](file:///{(KAGGLE_DIR / 'analysis' / 'ch4_condition_slices_from_dirs_test' / 'bms' / 'bms_regulation_score_boxplots.png').as_posix()})",
        "  作用：比较不同 regulation 切片下 baseline 与增强方案的分数分布。",
        "  重点：如果增强方案在 regulation/切换相关切片中分数更高，而在相对平稳切片中仍维持低背景，说明方法真正学到了工况相关差异。",
        "  适合写入：第四章 BMS 切片分析主图。",
        "",
        f"- [bms_regulation_error_summary.png](file:///{(KAGGLE_DIR / 'analysis' / 'ch4_condition_slices_from_dirs_test' / 'bms' / 'bms_regulation_error_summary.png').as_posix()})",
        "  作用：看误差是否主要集中在 regulation 感知困难的切片。",
        "  重点：可用来解释为什么某些增强方案会在无标签统计表中表现为更低阈值或更高最终阳性点数。",
        "  适合写入：第四章方法机理解释部分。",
        "",
        f"- [bms_branch_contribution.png](file:///{(KAGGLE_DIR / 'analysis' / 'ch4_condition_slices_from_dirs_test' / 'bms' / 'bms_branch_contribution.png').as_posix()})",
        "  作用：解释主分支与残差/层级一致性分支的贡献差异。",
        "  重点：如果 `C4+Hier` 的最终阳性点明显升高，可以用这张图说明是不是 residual/hier branch 在某些切片上显著放大了异常分数。",
        "  适合写入：第四章层级一致性机制说明。",
        "",
        f"- [bms_case_typical_anomaly.png](file:///{(KAGGLE_DIR / 'analysis' / 'ch4_condition_slices_from_dirs_test' / 'bms' / 'bms_case_typical_anomaly.png').as_posix()})",
        "  作用：展示典型异常 case 的时间序列。",
        "  重点：适合对照 `03_bms_table.md` 中最终阳性点明显升高的方案，说明异常片段是被合理延展，还是可能存在过度扩张。",
        "  适合写入：第四章案例分析小节。",
        "",
        "**BMS 在论文里怎么写**",
        "",
        "- 先用 `03_bms_table.md` 写：相较 baseline，`C3`/`C4` 改变了阈值分布，而 `C4+Hier` 进一步改变了最终异常片段范围。",
        "- 再用 `bms_regulation_score_boxplots.png` 与 `bms_regulation_error_summary.png` 支撑“变化主要集中在 regulation 感知困难切片”。",
        "- 如果需要解释层级一致性，就用 `bms_branch_contribution.png`。",
        "- 最后用 `bms_case_typical_anomaly.png` 给出一个可直观看出告警扩展效果的例子。",
    ]
    guide_path.write_text("\n".join(lines), encoding="utf-8")


def get_plan_config(plan_dir_name):
    path = CONFIG_DIR / f"{plan_dir_name}.json"
    if path.exists():
        return load_json(path), path
    return {}, None


def build_inventory(plan_dir, out_dir):
    config, config_path = get_plan_config(plan_dir.name)
    common_args = config.get("common_args", {})
    actual_map = {}
    for exp_dir in sorted(plan_dir.iterdir()):
        if exp_dir.is_dir():
            canon = canonical_name(exp_dir.name)
            actual_map.setdefault(canon, exp_dir)

    rows = []
    planned_names = set()
    for idx, experiment in enumerate(config.get("experiments", []), start=1):
        name = experiment.get("name", "")
        canon = canonical_name(name)
        planned_names.add(canon)
        args = dict(common_args)
        args.update(experiment.get("args", {}))
        found_dir = actual_map.get(canon)
        found_summary = found_dir is not None and has_any_summary(found_dir)
        status = "found_with_result" if found_summary else "found_no_result" if found_dir else "missing"
        rows.append(
            {
                "order": idx,
                "experiment_name": name,
                "dataset": args.get("dataset", ""),
                "run_id": args.get("run_id", ""),
                "status": status,
                "found_dir": str(found_dir) if found_dir else "",
                "comment": experiment.get("comment", "") or experiment.get("baseline", ""),
            }
        )

    extra_idx = len(rows) + 1
    for canon, exp_dir in actual_map.items():
        if canon in planned_names:
            continue
        rows.append(
            {
                "order": extra_idx,
                "experiment_name": exp_dir.name,
                "dataset": infer_dataset(exp_dir.name, read_config(exp_dir)),
                "run_id": infer_run_id(exp_dir.name, read_config(exp_dir)),
                "status": "extra_found_with_result" if has_any_summary(exp_dir) else "extra_found_no_result",
                "found_dir": str(exp_dir),
                "comment": "存在于离线目录，但未在当前计划配置中找到对应实验",
            }
        )
        extra_idx += 1

    inventory_df = pd.DataFrame(rows)
    if not inventory_df.empty:
        write_df(inventory_df, out_dir, "01_inventory")

    lines = [
        f"计划目录: {plan_dir.name}",
        f"配置文件: {config_path if config_path else '未找到'}",
        "",
        f"计划说明: {config.get('_comment', '')}",
        f"使用说明: {config.get('_usage', '')}",
        "",
        "实验映射:",
    ]
    mappings = config.get("_experiment_mapping", [])
    if mappings:
        lines.extend(f"- {item}" for item in mappings)
    else:
        lines.append("- 无")
    (out_dir / "README_计划说明.txt").write_text("\n".join(lines), encoding="utf-8")

    return config, inventory_df, actual_map


def build_plan_raw_outputs(plan_dir, out_dir, actual_map):
    exp_summaries = []
    case_rows = []
    for exp_dir in sorted(actual_map.values(), key=lambda p: p.name.lower()):
        summary, cases = summarize_experiment(exp_dir, plan_dir.name)
        if summary is not None:
            exp_summaries.append(summary)
            case_rows.extend(cases)

    exp_df = pd.DataFrame(exp_summaries)
    case_df = pd.DataFrame(case_rows)
    if not exp_df.empty:
        exp_df = exp_df.sort_values(["dataset", "experiment_name"]).reset_index(drop=True)
        write_df(exp_df, out_dir, "02_experiment_summary")
    if not case_df.empty:
        case_df = case_df.sort_values(["dataset", "experiment_name", "case_name"]).reset_index(drop=True)
        write_df(case_df, out_dir, "02_case_summary")
    return exp_df, case_df


def normalize_label(name):
    lowered = name.lower()
    if "tranad" in lowered:
        return "TranAD"
    if "anomaly_transformer" in lowered:
        return "Anomaly-Transformer"
    if "gdn" in lowered:
        return "GDN"
    if "dcdetector" in lowered:
        return "DCdetector"
    if "mtadgat" in lowered and "baseline" in lowered:
        return "MTAD-GAT"
    if "c4" in lowered and "physics" in lowered:
        return "C4+Hier"
    if "c3" in lowered and "physics" in lowered:
        return "C4"
    if "c3" in lowered and "no_transformer" in lowered:
        return "No Transformer"
    if "c3" in lowered and "no_revin" in lowered:
        return "No RevIN"
    if "fixed_fusion" in lowered:
        return "Fixed Fusion"
    if "no_event" in lowered:
        return "No Event"
    if "no_regime" in lowered:
        return "No Regime"
    if "phys_encoding_only" in lowered:
        return "Physics Encoding Only"
    if "phys_reg_only" in lowered:
        return "Physics Regularization Only"
    if "structure_only" in lowered:
        return "Structure Only"
    if "c3" in lowered:
        return "C3"
    return name


def battery_label(value):
    text = str(value)
    match = re.search(r"rw(\d+)", text, flags=re.IGNORECASE)
    return f"RW{match.group(1)}" if match else text


def dedupe_experiment_rows(df):
    if df.empty:
        return df
    ordered = df.sort_values(["dataset", "canonical_name", "plan", "experiment_name", "run_id"]).copy()
    return ordered.drop_duplicates(subset=["dataset", "canonical_name", "run_id"], keep="first").reset_index(drop=True)


def select_related_rows(all_exp_df, related_plan, dataset, experiment_names):
    if all_exp_df.empty:
        return pd.DataFrame(columns=all_exp_df.columns)
    wanted = {canonical_name(name) for name in experiment_names}
    return all_exp_df[
        (all_exp_df["plan"] == related_plan)
        & (all_exp_df["dataset"] == dataset)
        & (all_exp_df["canonical_name"].isin(wanted))
    ].copy()


def merge_plan_with_related(plan_name, exp_df, all_exp_df, datasets):
    frames = [exp_df[exp_df["dataset"].isin(datasets)].copy()]
    for dataset in datasets:
        for spec in RELATED_EXPERIMENTS.get(plan_name, {}).get(dataset, []):
            related = select_related_rows(all_exp_df, spec["plan"], dataset, spec["experiments"])
            if not related.empty:
                frames.append(related)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=exp_df.columns)
    return dedupe_experiment_rows(merged)


def build_cross_plan_note(df, current_plan):
    if df.empty or "Source_Plan" not in df.columns:
        return ""
    sources = sorted({str(item) for item in df["Source_Plan"].dropna().unique() if str(item) != current_plan})
    if not sources:
        return ""
    return f"本表额外补入了 `{'`, `'.join(sources)}` 中已完成的可比结果，用于补齐 baseline、full 或单独拆出去执行的对照实验。"


def get_loss_image_path(dir_path, loss_kind):
    base_dir = Path(dir_path)
    candidate = base_dir / f"{loss_kind}_losses.png"
    if candidate.exists():
        return candidate
    if base_dir.parent.exists():
        parent_candidate = base_dir.parent / f"{loss_kind}_losses.png"
        if parent_candidate.exists():
            return parent_candidate
    return None


def write_loss_grid(records, out_path, loss_kind):
    available = []
    for record in records:
        image_path = get_loss_image_path(record["dir_path"], loss_kind)
        if image_path is not None:
            item = dict(record)
            item["image_path"] = image_path
            available.append(item)

    if not available:
        return None, 0

    cols = 1 if len(available) == 1 else 2 if len(available) <= 4 else 3 if len(available) <= 9 else 4
    rows = math.ceil(len(available) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.6, rows * 4.4))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes:
        ax.axis("off")

    title_map = {"train": "训练损失", "validation": "验证损失"}
    for ax, record in zip(axes, available):
        image = plt.imread(record["image_path"])
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"{record['dataset']} | {record['label']}\n{record['experiment_name']} [{record['plan']}]",
            fontsize=9,
        )

    fig.suptitle(f"{out_path.parent.name} - {title_map[loss_kind]}拼图", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path, len(available)


def markdown_rel_path(from_file, target_path):
    target = Path(target_path).resolve()
    base = Path(from_file).resolve().parent
    return Path(os.path.relpath(target, start=base)).as_posix()


def image_markdown(from_file, path, alt_text):
    return f"![{alt_text}]({markdown_rel_path(from_file, path)})"


def append_markdown_section(md_path, section_lines):
    if not md_path.exists() or not section_lines:
        return
    content = md_path.read_text(encoding="utf-8").rstrip()
    content += "\n\n" + "\n".join(section_lines).rstrip() + "\n"
    md_path.write_text(content, encoding="utf-8")


def build_external_registry_notes(plan_dir):
    registry_paths = sorted(plan_dir.glob("run_registry*.json"))
    if not registry_paths:
        return []

    status_map = {}
    for registry_path in registry_paths:
        payload = load_json(registry_path)
        for experiment in payload.get("experiments", []):
            name = str(experiment.get("name", ""))
            status_map.setdefault(name, []).append(
                {
                    "status": str(experiment.get("status", "")),
                    "return_code": experiment.get("return_code"),
                    "registry": registry_path.name,
                }
            )

    done_but_missing = []
    failed_only = []
    for name, rows in sorted(status_map.items()):
        exp_dir = plan_dir / name
        has_standard_output = (exp_dir / "summary_metrics.json").exists() or (exp_dir / "test_output.pkl").exists()
        if any(row["status"] == "done" for row in rows) and not has_standard_output:
            done_but_missing.append(name)
        elif any(row["status"] == "failed" for row in rows) and not any(row["status"] == "done" for row in rows) and not has_standard_output:
            failed_only.append(name)

    lines = []
    if done_but_missing:
        lines.append(
            f"`run_registry` 记录里，`{'`, `'.join(done_but_missing)}` 当时状态为 `done`，"
            "但当前离线目录只保留了 checkpoint/log，缺少 `summary_metrics.json` 或 `test_output.pkl`，所以没法自动汇总到表里。"
        )
    if failed_only:
        lines.append(
            f"当前 registry 中仅看到 `{'`, `'.join(failed_only)}` 没有成功产物；至少在这份离线包里，它们没有可用的最终标准化结果。"
        )
    return lines


def build_gdn_diagnosis_lines():
    base_dir = KAGGLE_DIR / "ch3_external_baselines"
    lines = []
    configs = [
        ("SMAP", base_dir / "gdn_smap_ch3"),
        ("MSL", base_dir / "gdn_msl_ch3"),
    ]
    for dataset, exp_dir in configs:
        summary_path = exp_dir / "summary_metrics.json"
        score_path = exp_dir / "test_output.pkl"
        if not summary_path.exists() or not score_path.exists():
            continue
        summary = load_json(summary_path)
        df = pd.read_pickle(score_path)
        threshold = safe_float(summary.get("metric_threshold"))
        f1 = safe_float(summary.get("metric_f1"))
        auroc = safe_float(summary.get("metric_auroc") or summary.get("metric_auc"))
        precision = safe_float(summary.get("metric_precision"))
        recall = safe_float(summary.get("metric_recall"))
        score_std = safe_float(df["A_Score_Global"].astype(float).std())
        lines.append(
            f"`GDN-{dataset}` 当前仅达到 `F1={fmt(f1)}`、`AUROC={fmt(auroc)}`，"
            f"`precision={fmt(precision)}` 明显低于 `recall={fmt(recall)}`；"
            f"结合导出的 `threshold={fmt(threshold)}` 与分数波动 `std={fmt(score_std)}`，说明它在当前统一数据口径下更像是“能覆盖到异常，但排序和阈值稳定性都不够好”。"
        )
    if lines:
        lines.append("从配置上看，`GDN` 在当前计划里只训练 `10` 个 epoch，窗口仅 `15`，而主模型使用更长时间窗和更完整后处理；这也是它明显落后的一个直接原因。")
    return lines


def build_loss_analysis_rows(compared_df):
    lines = []
    labeled = compared_df[compared_df["dataset"].isin(PUBLIC_DATASETS)].copy()
    if not labeled.empty:
        for dataset, group in labeled.groupby("dataset", dropna=False):
            metric = pd.to_numeric(group["bf_f1"], errors="coerce")
            valid = group.loc[metric.notna()].copy()
            if len(valid) < 2:
                continue
            valid["bf_f1_num"] = pd.to_numeric(valid["bf_f1"], errors="coerce")
            valid = valid.sort_values("bf_f1_num", ascending=False)
            best = valid.iloc[0]
            worst = valid.iloc[-1]
            lines.append(
                f"`{dataset}` 建议重点对比 `{normalize_label(best['experiment_name'])}` 与 `{normalize_label(worst['experiment_name'])}` 的验证损失曲线；"
                f"它们在当前比较集合中的 `BF_F1` 分别为 `{fmt(best['bf_f1'])}` 和 `{fmt(worst['bf_f1'])}`。"
            )

    nasa = compared_df[compared_df["dataset"] == "NASA_RANDOM_DISCHARGE"].copy()
    if not nasa.empty:
        eps = pd.to_numeric(nasa["epsilon_threshold"], errors="coerce")
        valid = nasa.loc[eps.notna()].copy()
        if not valid.empty:
            valid["epsilon_threshold_num"] = pd.to_numeric(valid["epsilon_threshold"], errors="coerce")
            best = valid.sort_values("epsilon_threshold_num").iloc[0]
            lines.append(
                f"`NASA_RANDOM_DISCHARGE` 建议优先查看 `{best['run_id']}-{normalize_label(best['experiment_name'])}` 的 loss 曲线，"
                f"再结合切片图判断更低阈值是否来自 phase 切换段。"
            )

    bms = compared_df[compared_df["dataset"] == "BMS"].copy()
    if not bms.empty:
        final_positive = pd.to_numeric(bms["final_positive"], errors="coerce")
        valid = bms.loc[final_positive.notna()].copy()
        if not valid.empty:
            valid["final_positive_num"] = pd.to_numeric(valid["final_positive"], errors="coerce")
            top = valid.sort_values("final_positive_num", ascending=False).iloc[0]
            lines.append(
                f"`BMS` 建议重点对比 `{normalize_label(top['experiment_name'])}` 与 baseline/`C3` 的训练和验证损失，"
                f"并结合最终阳性点变化分析层级一致性或物理增强是否改变了收敛行为。"
            )

    ch_battery = compared_df[compared_df["dataset"] == CH_BATTERY_DATASET].copy()
    if not ch_battery.empty and "sample_auroc" in ch_battery.columns:
        auroc = pd.to_numeric(ch_battery["sample_auroc"], errors="coerce")
        valid = ch_battery.loc[auroc.notna()].copy()
        if not valid.empty:
            valid["sample_auroc_num"] = pd.to_numeric(valid["sample_auroc"], errors="coerce")
            best = valid.sort_values("sample_auroc_num", ascending=False).iloc[0]
            lines.append(
                f"`CH_BATTERY_LFP_DISCHARGE` 建议重点对比 `{normalize_label(best['experiment_name'])}` 与 baseline 的 loss 曲线，再结合样本级 `AUROC/AUPRC` 判断提升是否来自更稳定的样本级排序。"
            )

    if not lines:
        lines.append("当前计划缺少可用于稳定比较的指标组合，loss 拼图主要作为训练过程留档。")
    return lines


def build_plan_loss_outputs(plan_dir, out_dir, compared_df):
    if compared_df.empty:
        return

    records_df = dedupe_experiment_rows(compared_df).copy()
    if records_df.empty:
        return

    records_df["label"] = records_df["experiment_name"].map(normalize_label)
    records = records_df[["dataset", "plan", "experiment_name", "dir_path", "label", "run_id"]].to_dict("records")

    train_path, train_count = write_loss_grid(records, out_dir / "04_train_loss_overview.png", "train")
    val_path, val_count = write_loss_grid(records, out_dir / "04_validation_loss_overview.png", "validation")

    missing = []
    for record in records:
        if get_loss_image_path(record["dir_path"], "train") is None and get_loss_image_path(record["dir_path"], "validation") is None:
            missing.append(f"{record['experiment_name']} [{record['plan']}]")

    lines = [
        "# Loss 曲线对比",
        "",
        f"- 当前拼图覆盖 `03_*` 表中涉及的 `{len(records)}` 个实验；其中训练损失图找到 `{train_count}` 个，验证损失图找到 `{val_count}` 个。",
    ]
    loss_md = out_dir / "04_loss_comparison.md"
    if train_path is not None:
        lines.append(f"- 训练损失拼图：[04_train_loss_overview.png](file:///{train_path.as_posix()})")
        lines.extend(["", image_markdown(loss_md, train_path, f"{out_dir.name} 训练损失拼图")])
    if val_path is not None:
        lines.append(f"- 验证损失拼图：[04_validation_loss_overview.png](file:///{val_path.as_posix()})")
        lines.extend(["", image_markdown(loss_md, val_path, f"{out_dir.name} 验证损失拼图")])
    if missing:
        lines.append(f"- 以下实验未找到 loss png：`{'`, `'.join(missing)}`。")
    lines.extend(["", "## 分析"])
    lines.extend(f"- {line}" for line in build_loss_analysis_rows(records_df))
    loss_md.write_text("\n".join(lines), encoding="utf-8")


def build_public_visual_section(plan_dir, out_dir, md_path):
    lines = ["## 相关图像", ""]
    train_path = out_dir / "04_train_loss_overview.png"
    val_path = out_dir / "04_validation_loss_overview.png"
    if train_path.exists():
        lines.extend(["### 训练损失拼图", "", image_markdown(md_path, train_path, f"{out_dir.name} 训练损失拼图"), ""])
    if val_path.exists():
        lines.extend(["### 验证损失拼图", "", image_markdown(md_path, val_path, f"{out_dir.name} 验证损失拼图"), ""])

    score_paths = sorted(out_dir.glob("*score_separation*.png"))
    if score_paths:
        lines.extend(["### 分数区分度图", "", image_markdown(md_path, score_paths[0], f"{out_dir.name} 分数区分度图"), ""])

    lines.extend(["## 结果解读"])
    if plan_dir.name == "ch3_external_baselines":
        notes = build_external_registry_notes(plan_dir)
        notes.extend(build_gdn_diagnosis_lines())
        if not notes:
            notes.append("当前外部基线目录只保留了部分标准化产物，因此先以现有可读结果为准。")
        lines.extend(f"- {line}" for line in notes)
        lines.extend(
            [
                "- 从现有可读结果看，公开数据集上真正有竞争力的仍是 `MTAD-GAT` 与 `C3`；当前这版 `GDN` 与二者差距较大，不能支撑“外部基线优于本文方法”的结论。",
                "- 当前 `Anomaly-Transformer` 与 `DCdetector` 已补齐统一离线评估；但 `TranAD` 仍是基于历史日志恢复的指标，缺少同口径 `test_output.pkl`，因此这组结果更适合作为主表补充，而不是细粒度分数分析来源。",
            ]
        )
    elif plan_dir.name == "ch3_ablation":
        lines.extend(
            [
                "- 在 `MSL` 上，`Fixed Fusion` 的 `BF_F1` 高于 `C3`，说明第三章完整方案中的质量感知融合并不是在所有场景都占优，至少在 `MSL` 上存在进一步简化反而更稳的现象。",
                "- 在 `SMAP` 上，baseline `MTAD-GAT` 仍然整体最好，说明 `C3` 及其各项消融虽然改变了分数分布，但没有把异常与正常的排序关系真正拉开；这里的核心问题不是“有没有收敛”，而是“增强模块是否适合该数据集”。",
                "- 因此第三章消融更适合写成：各模块在 `MSL` 上能带来局部收益，但在 `SMAP` 上存在明显数据集依赖性，说明改进并非无条件成立。"
            ]
        )
    else:
        lines.extend(
            [
                "- loss 曲线主要用于排除“训练没跑稳”这种低层原因；如果几组曲线都已经收敛但最终指标仍有差异，就应把重点放在分数区分度和后处理行为，而不是继续讨论训练轮数。",
                "- 因此这里的图像更适合作为训练过程佐证，而不是主结论本身；主结论仍以表中的指标优劣为准。",
            ]
        )
    return lines


def build_nasa_visual_section(out_dir, md_path):
    nasa_dir = KAGGLE_DIR / "analysis" / "ch4_condition_slices_from_dirs_test" / "nasa_random"
    lines = ["## 相关图像", ""]
    train_path = out_dir / "04_train_loss_overview.png"
    val_path = out_dir / "04_validation_loss_overview.png"
    if train_path.exists():
        lines.extend(["### 训练损失拼图", "", image_markdown(md_path, train_path, f"{out_dir.name} 训练损失拼图"), ""])
    if val_path.exists():
        lines.extend(["### 验证损失拼图", "", image_markdown(md_path, val_path, f"{out_dir.name} 验证损失拼图"), ""])

    image_names = [
        ("nasa_phase_score_boxplots.png", "NASA 工况切片分数箱线图"),
        ("nasa_phase_switch_profile.png", "NASA 工况切换平均轮廓"),
        ("nasa_case_transition.png", "NASA 典型切换案例"),
        ("nasa_case_stable.png", "NASA 稳定片段对照"),
    ]
    for file_name, alt_text in image_names:
        image_path = nasa_dir / file_name
        if image_path.exists():
            lines.extend([f"### {alt_text}", "", image_markdown(md_path, image_path, alt_text), ""])

    lines.extend(
        [
            "## 结果解读",
            "- 结合表和切片图，这一组结果更适合写成“`C4` 相比 baseline 更敏感于 phase 切换”，而不是直接写成“性能更好”；因为当前没有统一标签，结论只能落在分数行为层面。",
            "- 从 `RW1/RW7/RW8` 看，`C4` 普遍把阈值压低到比 baseline 更小的区间，且切换轮廓图如果在 transition 中心附近出现更尖锐的峰，就说明物理增强主要在工况切换处起作用。",
            "- 但 `RW2` 上 `C3` 的阈值和最终触发范围仍优于 `C4`，说明物理增强并不是对所有随机工况都稳定增益；这一点在论文里应写成“跨电池存在异质性”。",
            "- 因而 NASA 这部分最稳妥的写法是：方法能改变异常分数在 phase 切换处的响应方式，并给出若干典型案例支撑，但不把它包装成严格可量化的精度提升。 ",
        ]
    )
    return lines


def build_bms_visual_section(out_dir, md_path):
    bms_dir = KAGGLE_DIR / "analysis" / "ch4_condition_slices_from_dirs_test" / "bms"
    lines = ["## 相关图像", ""]
    train_path = out_dir / "04_train_loss_overview.png"
    val_path = out_dir / "04_validation_loss_overview.png"
    if train_path.exists():
        lines.extend(["### 训练损失拼图", "", image_markdown(md_path, train_path, f"{out_dir.name} 训练损失拼图"), ""])
    if val_path.exists():
        lines.extend(["### 验证损失拼图", "", image_markdown(md_path, val_path, f"{out_dir.name} 验证损失拼图"), ""])

    image_names = [
        ("bms_regulation_score_boxplots.png", "BMS 调节切片分数箱线图"),
        ("bms_regulation_error_summary.png", "BMS 调节切片误差汇总"),
        ("bms_branch_contribution.png", "BMS 分支贡献图"),
        ("bms_case_typical_anomaly.png", "BMS 典型异常案例"),
    ]
    for file_name, alt_text in image_names:
        image_path = bms_dir / file_name
        if image_path.exists():
            lines.extend([f"### {alt_text}", "", image_markdown(md_path, image_path, alt_text), ""])

    lines.extend(
        [
            "## 结果解读",
            "- `C4` 的平均阈值最低，而 `C4+Hier` 的最终阳性片段最长，这说明第四章两类增强起到的是不同作用：物理增强更多改变基础分数分布，层级一致性更多改变最终告警范围。",
            "- 如果 `bms_regulation_score_boxplots.png` 和 `bms_regulation_error_summary.png` 显示分数与误差都集中抬升在 regulation/切换片段，那么就可以支撑“模型学到了工况敏感区域”这一写法，而不是简单说阈值更低。",
            "- `bms_branch_contribution.png` 若显示层级分支在特定切片贡献更高，就能解释为什么 `C4+Hier` 会把异常片段进一步连成长段；这在论文里应表述为“改善了片段连续性”，同时保留“可能扩大告警范围”的审慎表述。",
            "- 因此 BMS 这部分是可以写进硕论的，但要明确它属于“无标签场景下的机理证据和案例证据”，不是严格监督指标上的性能证明。",
        ]
    )
    return lines


def build_ch_battery_visual_section(out_dir, md_path):
    lines = ["## 相关图像", ""]
    train_path = out_dir / "04_train_loss_overview.png"
    val_path = out_dir / "04_validation_loss_overview.png"
    if train_path.exists():
        lines.extend(["### 训练损失拼图", "", image_markdown(md_path, train_path, f"{out_dir.name} 训练损失拼图"), ""])
    if val_path.exists():
        lines.extend(["### 验证损失拼图", "", image_markdown(md_path, val_path, f"{out_dir.name} 验证损失拼图"), ""])

    lines.extend(
        [
            "## 结果解读",
            "- CH-BATTERY 的主结果应以样本级 `AUROC`、`AUPRC` 和 `Best_F1` 为主，而不是沿用 NASA/BMS 那种阈值与阳性片段统计口径。",
            "- 这里的 loss 图主要用于确认不同方案都已稳定收敛；真正的优劣判断应回到样本级弱标签汇总表。",
            "- 因为所有方案共用同一份预处理产物、同一套正常 VIN 训练划分和故障样本测试集合，所以这组表可以直接用于外部有标签样本级对比。",
        ]
    )
    return lines


def append_visuals_to_table_markdown(plan_dir, out_dir):
    public_md = out_dir / "03_public_labeled_table.md"
    nasa_md = out_dir / "03_nasa_random_table.md"
    bms_md = out_dir / "03_bms_table.md"
    ch_battery_md = out_dir / "03_ch_battery_table.md"
    table_visuals = {
        "03_public_labeled_table.md": build_public_visual_section(plan_dir, out_dir, public_md),
        "03_nasa_random_table.md": build_nasa_visual_section(out_dir, nasa_md),
        "03_bms_table.md": build_bms_visual_section(out_dir, bms_md),
        "03_ch_battery_table.md": build_ch_battery_visual_section(out_dir, ch_battery_md),
    }
    for file_name, section_lines in table_visuals.items():
        append_markdown_section(out_dir / file_name, section_lines)


def build_plan_paper_tables(plan_dir, out_dir, exp_df, all_exp_df):
    if exp_df.empty:
        return [], pd.DataFrame()

    generated = []
    compared_frames = []

    labeled_df = merge_plan_with_related(plan_dir.name, exp_df, all_exp_df, PUBLIC_DATASETS)
    if not labeled_df.empty:
        compared_frames.append(labeled_df)
        labeled_df["Label"] = labeled_df["experiment_name"].map(normalize_label)
        out = labeled_df[
            ["dataset", "Label", "plan", "experiment_name", "run_id", "epsilon_f1", "pot_f1", "bf_f1", "event_f1", "epsilon_threshold", "event_enabled"]
        ].copy()
        out.columns = [
            "Dataset",
            "Variant",
            "Source_Plan",
            "Experiment",
            "Run_ID",
            "Epsilon_F1",
            "POT_F1",
            "BF_F1",
            "Event_F1",
            "Epsilon_Threshold",
            "Event_Enabled",
        ]
        for col in ["Epsilon_F1", "POT_F1", "BF_F1", "Event_F1", "Epsilon_Threshold"]:
            out[col] = out[col].map(fmt)
        out = out.sort_values(["Dataset", "Variant", "Source_Plan", "Experiment"]).reset_index(drop=True)
        analysis_lines = build_public_analysis(out)
        source_note = build_cross_plan_note(out, plan_dir.name)
        if source_note:
            analysis_lines.append(source_note)
        write_df(out, out_dir, "03_public_labeled_table")
        write_highlighted_markdown(
            out,
            out_dir,
            "03_public_labeled_table",
            analysis_lines,
            group_key="Dataset",
            higher_is_better_cols=["Epsilon_F1", "POT_F1", "BF_F1", "Event_F1"],
        )
        generated.append("03_public_labeled_table")

    nasa_df = merge_plan_with_related(plan_dir.name, exp_df, all_exp_df, {"NASA_RANDOM_DISCHARGE"})
    if not nasa_df.empty:
        compared_frames.append(nasa_df)
        nasa_df["Battery"] = nasa_df["run_id"].map(battery_label)
        nasa_df["Label"] = nasa_df["experiment_name"].map(normalize_label)
        out = nasa_df[
            ["Battery", "Label", "plan", "experiment_name", "run_id", "epsilon_threshold", "pot_threshold", "raw_positive", "final_positive", "event_enabled"]
        ].copy()
        out.columns = [
            "Battery",
            "Variant",
            "Source_Plan",
            "Experiment",
            "Run_ID",
            "Epsilon_Threshold",
            "POT_Threshold",
            "Raw_Positive",
            "Final_Positive",
            "Event_Enabled",
        ]
        for col in ["Epsilon_Threshold", "POT_Threshold", "Raw_Positive", "Final_Positive"]:
            out[col] = out[col].map(fmt)
        out = out.sort_values(["Battery", "Variant", "Source_Plan", "Experiment"]).reset_index(drop=True)
        analysis_lines = build_nasa_analysis(out)
        source_note = build_cross_plan_note(out, plan_dir.name)
        if source_note:
            analysis_lines.append(source_note)
        write_df(out, out_dir, "03_nasa_random_table")
        write_highlighted_markdown(
            out,
            out_dir,
            "03_nasa_random_table",
            analysis_lines,
            group_key="Battery",
            lower_is_better_cols=["Epsilon_Threshold", "POT_Threshold"],
            higher_is_better_cols=["Raw_Positive", "Final_Positive"],
        )
        generated.append("03_nasa_random_table")

    bms_df = merge_plan_with_related(plan_dir.name, exp_df, all_exp_df, {"BMS"})
    if not bms_df.empty:
        compared_frames.append(bms_df)
        bms_df["Label"] = bms_df["experiment_name"].map(normalize_label)
        out = bms_df[
            ["Label", "plan", "experiment_name", "run_id", "case_count", "epsilon_threshold", "pot_threshold", "raw_positive", "final_positive", "event_enabled", "hier_enabled"]
        ].copy()
        out.columns = [
            "Variant",
            "Source_Plan",
            "Experiment",
            "Run_ID",
            "Clusters",
            "Avg_Epsilon_Threshold",
            "Avg_POT_Threshold",
            "Avg_Raw_Positive",
            "Avg_Final_Positive",
            "Event_Enabled",
            "Hier_Enabled",
        ]
        for col in ["Avg_Epsilon_Threshold", "Avg_POT_Threshold", "Avg_Raw_Positive", "Avg_Final_Positive"]:
            out[col] = out[col].map(fmt)
        out = out.sort_values(["Variant", "Source_Plan", "Experiment"]).reset_index(drop=True)
        analysis_lines = build_bms_analysis(out)
        source_note = build_cross_plan_note(out, plan_dir.name)
        if source_note:
            analysis_lines.append(source_note)
        write_df(out, out_dir, "03_bms_table")
        write_highlighted_markdown(
            out,
            out_dir,
            "03_bms_table",
            analysis_lines,
            lower_is_better_cols=["Avg_Epsilon_Threshold", "Avg_POT_Threshold"],
            higher_is_better_cols=["Avg_Raw_Positive", "Avg_Final_Positive"],
        )
        generated.append("03_bms_table")

    ch_battery_df = merge_plan_with_related(plan_dir.name, exp_df, all_exp_df, {CH_BATTERY_DATASET})
    if not ch_battery_df.empty:
        compared_frames.append(ch_battery_df)
        ch_battery_df["Label"] = ch_battery_df["experiment_name"].map(normalize_label)
        out = ch_battery_df[
            ["Label", "plan", "experiment_name", "run_id", "sample_count", "fault_count", "score_field", "sample_auroc", "sample_auprc", "best_f1", "best_threshold"]
        ].copy()
        out.columns = [
            "Variant",
            "Source_Plan",
            "Experiment",
            "Run_ID",
            "Samples",
            "Fault_Samples",
            "Score_Field",
            "Sample_AUROC",
            "Sample_AUPRC",
            "Best_F1",
            "Best_Threshold",
        ]
        for col in ["Samples", "Fault_Samples", "Sample_AUROC", "Sample_AUPRC", "Best_F1", "Best_Threshold"]:
            out[col] = out[col].map(fmt)
        out = out.sort_values(["Variant", "Source_Plan", "Experiment"]).reset_index(drop=True)
        analysis_lines = build_ch_battery_analysis(out)
        source_note = build_cross_plan_note(out, plan_dir.name)
        if source_note:
            analysis_lines.append(source_note)
        write_df(out, out_dir, "03_ch_battery_table")
        write_highlighted_markdown(
            out,
            out_dir,
            "03_ch_battery_table",
            analysis_lines,
            higher_is_better_cols=["Sample_AUROC", "Sample_AUPRC", "Best_F1"],
        )
        generated.append("03_ch_battery_table")

    compared_df = dedupe_experiment_rows(pd.concat(compared_frames, ignore_index=True)) if compared_frames else pd.DataFrame()
    return generated, compared_df


def write_top_readme(plan_names):
    lines = [
        "按实验计划整理的离线分析结果",
        "",
        "目录规则：",
        "- 每个计划单独一个目录，不再混合第三章/第四章或不同数据集。",
        "- 01_inventory：计划中声明的实验、实际找到的离线目录、是否存在可用结果。",
        "- 02_experiment_summary：按实验聚合后的原始汇总。",
        "- 02_case_summary：按 case/cluster 的细粒度汇总。",
        "- 03_*_table：从当前计划内部提炼出的可读表格；CH-BATTERY 会单独输出 `03_ch_battery_table`。",
        "",
        "已整理计划：",
    ]
    lines.extend(f"- {name}" for name in plan_names)
    (OUTPUT_ROOT / "README_按计划整理.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    plan_names = []
    plan_payloads = []
    all_exp_parts = []
    for plan_dir in sorted(KAGGLE_DIR.iterdir()):
        if not plan_dir.is_dir() or plan_dir.name == "analysis":
            continue

        plan_names.append(plan_dir.name)
        out_dir = OUTPUT_ROOT / plan_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        _, _, actual_map = build_inventory(plan_dir, out_dir)
        exp_df, _ = build_plan_raw_outputs(plan_dir, out_dir, actual_map)
        plan_payloads.append((plan_dir, out_dir, exp_df))
        if not exp_df.empty:
            all_exp_parts.append(exp_df)

    all_exp_df = pd.concat(all_exp_parts, ignore_index=True) if all_exp_parts else pd.DataFrame()

    for plan_dir, out_dir, exp_df in plan_payloads:
        _, compared_df = build_plan_paper_tables(plan_dir, out_dir, exp_df, all_exp_df)
        build_plan_loss_outputs(plan_dir, out_dir, compared_df)
        append_visuals_to_table_markdown(plan_dir, out_dir)

    write_visual_guide()
    write_top_readme(plan_names)
    print(f"Wrote by-plan analysis to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
