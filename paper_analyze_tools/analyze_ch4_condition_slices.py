"""按实验目录或 registry 生成第四章切片分析图，用于 NASA/BMS 无标签结果的条件切片、切换轮廓和典型案例可视化。"""

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import get_bms_feature_names


ROOT = PROJECT_ROOT
NASA_RANDOM_FEATURE_NAMES = ["step_type_code", "voltage", "current", "temperature"]
NASA_SLICE_ORDER = ["stable_charge", "stable_discharge", "stable_rest", "transition"]
BMS_SLICE_ORDER = ["relative_steady", "high_frequency_regulation"]

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SLICE_LABEL_ZH = {
    "stable_charge": "稳定充电",
    "stable_discharge": "稳定放电",
    "stable_rest": "稳定静置",
    "transition": "工况切换",
    "relative_steady": "相对平稳",
    "high_frequency_regulation": "高频调节",
}

METRIC_LABEL_ZH = {
    "A_Score_Global_mean": "全局异常分数均值",
    "Voltage_Delta_Error_mean": "电压差分误差均值",
    "Temperature_Delta_Error_mean": "温度差分误差均值",
    "Current_Cum_Error_mean": "电流累积误差均值",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Condition-aware slice analysis for Chapter 4 battery experiments."
    )
    parser.add_argument(
        "--registries",
        nargs="+",
        default=[],
        help="One or more run_registry.json files produced by compare_experiments.py",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[],
        help="One or more extracted experiment directories, plan directories, or kaggle离线output root directories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to analysis/ch4_condition_slices",
    )
    parser.add_argument(
        "--transition-margin",
        type=int,
        default=5,
        help="Rows near a NASA phase boundary that are marked as transition",
    )
    parser.add_argument(
        "--transition-radius",
        type=int,
        default=40,
        help="Radius for phase-switch score profile aggregation",
    )
    parser.add_argument(
        "--reg-high-quantile",
        type=float,
        default=0.7,
        help="Quantile threshold for BMS high-frequency regulation slices",
    )
    parser.add_argument(
        "--reg-low-quantile",
        type=float,
        default=0.3,
        help="Quantile threshold for BMS relative-steady slices",
    )
    parser.add_argument(
        "--case-radius",
        type=int,
        default=80,
        help="Half window size for local case plots",
    )
    return parser.parse_args()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def zh_slice_label(name):
    return SLICE_LABEL_ZH.get(name, name)


def zh_metric_label(name):
    return METRIC_LABEL_ZH.get(name, name)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def is_sequence_container(obj):
    return isinstance(obj, (list, tuple))


def normalize_experiment_label(experiment):
    name = experiment.get("name", "")
    args = experiment.get("args", {})
    dataset = str(experiment.get("dataset", args.get("dataset", ""))).upper()
    model_name = str(args.get("model_name", "")).lower()
    use_phys_enc = bool(args.get("use_physical_state_encoding", False))
    use_phys_reg = bool(args.get("use_physical_regularization", False))

    if dataset == "NASA_RANDOM_DISCHARGE":
        if use_phys_enc and use_phys_reg:
            return "C4"
        if use_phys_enc:
            return "C3+PhysEnc"
        if use_phys_reg:
            return "C3+PhysReg"
        return "C3"

    if dataset == "BMS":
        if model_name == "mtad_gat_c4" and use_phys_enc and use_phys_reg:
            return "C4+Hier"
        if model_name == "mtad_gat_c3" and use_phys_enc and use_phys_reg:
            return "C4"
        if model_name == "mtad_gat_c4" and not use_phys_enc and not use_phys_reg:
            return "C4-StructureOnly"
        if model_name == "mtad_gat_c3" and use_phys_enc:
            return "C3+PhysEnc"
        if model_name == "mtad_gat_c3" and use_phys_reg:
            return "C3+PhysReg"
        if model_name == "mtad_gat_c4":
            return "C4"
        return "C3"

    return name or model_name or dataset or "exp"


def canonical_experiment_name(name):
    name = re.sub(r" \(\d+\)$", "", str(name))
    name = re.sub(r"（\d+）$", "", name)
    return name


def infer_dataset_from_name(name):
    lowered = str(name).lower()
    if "nasa_random" in lowered or "_rw" in lowered:
        return "NASA_RANDOM_DISCHARGE"
    if "bms" in lowered:
        return "BMS"
    if "smap" in lowered:
        return "SMAP"
    if "msl" in lowered:
        return "MSL"
    return ""


def build_aligned_raw_dataframe(raw_data, window_size, feature_names):
    if is_sequence_container(raw_data):
        parts = []
        global_pos = 0
        for segment_id, seq in enumerate(raw_data):
            seq = np.asarray(seq, dtype=np.float32)
            if len(seq) <= window_size:
                continue
            aligned = seq[window_size:]
            part = pd.DataFrame(aligned, columns=feature_names)
            part["Segment_ID"] = int(segment_id)
            part["Segment_Pos"] = np.arange(len(aligned), dtype=np.int32)
            part["Global_Pos"] = np.arange(global_pos, global_pos + len(aligned), dtype=np.int32)
            parts.append(part)
            global_pos += len(aligned)
        if not parts:
            return pd.DataFrame(columns=list(feature_names) + ["Segment_ID", "Segment_Pos", "Global_Pos"])
        return pd.concat(parts, axis=0, ignore_index=True)

    raw_data = np.asarray(raw_data, dtype=np.float32)
    if len(raw_data) <= window_size:
        return pd.DataFrame(columns=list(feature_names) + ["Segment_ID", "Segment_Pos", "Global_Pos"])
    aligned = raw_data[window_size:]
    df = pd.DataFrame(aligned, columns=feature_names)
    df["Segment_ID"] = 0
    df["Segment_Pos"] = np.arange(len(aligned), dtype=np.int32)
    df["Global_Pos"] = np.arange(len(aligned), dtype=np.int32)
    return df


def merge_score_and_raw(score_df, raw_df, feature_prefix="Raw_"):
    score_df = score_df.reset_index(drop=True).copy()
    raw_df = raw_df.reset_index(drop=True).copy()
    if len(score_df) != len(raw_df):
        min_len = min(len(score_df), len(raw_df))
        score_df = score_df.iloc[:min_len].reset_index(drop=True)
        raw_df = raw_df.iloc[:min_len].reset_index(drop=True)

    metadata_cols = ["Segment_ID", "Segment_Pos", "Global_Pos"]
    raw_feature_cols = [c for c in raw_df.columns if c not in metadata_cols]
    missing_metadata_cols = [c for c in metadata_cols if c in raw_df.columns and c not in score_df.columns]
    raw_metadata_df = raw_df[missing_metadata_cols] if missing_metadata_cols else pd.DataFrame(index=raw_df.index)
    raw_feature_df = raw_df[raw_feature_cols].rename(columns={c: f"{feature_prefix}{c}" for c in raw_feature_cols})
    return pd.concat([score_df, raw_metadata_df, raw_feature_df], axis=1)


def safe_diff_abs_error(pred_series, true_series):
    pred = np.asarray(pred_series, dtype=np.float32)
    true = np.asarray(true_series, dtype=np.float32)
    if len(pred) == 0:
        return np.array([], dtype=np.float32)
    out = np.zeros(len(pred), dtype=np.float32)
    if len(pred) > 1:
        out[1:] = np.abs(np.diff(pred) - np.diff(true))
        out[0] = out[1]
    return out


def safe_normalized_cumsum_abs_error(pred_series, true_series):
    pred = np.asarray(pred_series, dtype=np.float32)
    true = np.asarray(true_series, dtype=np.float32)
    if len(pred) == 0:
        return np.array([], dtype=np.float32)
    pred_cum = np.cumsum(pred)
    true_cum = np.cumsum(true)
    pred_scale = max(float(np.max(np.abs(pred_cum))), 1e-6)
    true_scale = max(float(np.max(np.abs(true_cum))), 1e-6)
    return np.abs(pred_cum / pred_scale - true_cum / true_scale)


def add_common_metrics(df, voltage_idx, current_idx, temperature_idx):
    if voltage_idx is not None:
        df["Voltage_Delta_Error"] = safe_diff_abs_error(df[f"Recon_{voltage_idx}"], df[f"True_{voltage_idx}"])
    if temperature_idx is not None:
        df["Temperature_Delta_Error"] = safe_diff_abs_error(df[f"Recon_{temperature_idx}"], df[f"True_{temperature_idx}"])
    if current_idx is not None:
        df["Current_Cum_Error"] = safe_normalized_cumsum_abs_error(df[f"Recon_{current_idx}"], df[f"True_{current_idx}"])
    return df


def classify_phase_from_code(value):
    if value > 0:
        return "charge"
    if value < 0:
        return "discharge"
    return "rest"


def add_nasa_slice_labels(df, transition_margin):
    phase = df["Raw_step_type_code"].astype(float).map(classify_phase_from_code)
    slice_labels = np.array([f"stable_{item}" for item in phase], dtype=object)

    for _, segment_df in df.groupby("Segment_ID", sort=True):
        idxs = segment_df.index.to_numpy()
        phase_values = phase.loc[idxs].to_numpy()
        if len(idxs) <= 1:
            continue
        change_points = np.where(phase_values[1:] != phase_values[:-1])[0] + 1
        if len(change_points) == 0:
            continue
        for cp in change_points:
            start = max(0, cp - transition_margin)
            end = min(len(idxs), cp + transition_margin + 1)
            slice_labels[idxs[start:end]] = "transition"

    df["phase_label"] = phase
    df["slice_label"] = slice_labels
    return df


def compute_transition_profile(df, radius):
    rows = []
    for (experiment_label, entity_name), entity_df in df.groupby(["experiment_label", "entity_name"], sort=False):
        if "phase_label" not in entity_df.columns:
            continue
        for _, segment_df in entity_df.groupby("Segment_ID", sort=True):
            phase_values = segment_df["phase_label"].to_numpy()
            scores = segment_df["A_Score_Global"].astype(float).to_numpy()
            if len(phase_values) <= 1:
                continue
            change_points = np.where(phase_values[1:] != phase_values[:-1])[0] + 1
            for cp in change_points:
                left = max(0, cp - radius)
                right = min(len(segment_df), cp + radius + 1)
                base_positions = np.arange(left, right) - cp
                for rel_pos, score in zip(base_positions, scores[left:right]):
                    rows.append({
                        "experiment_label": experiment_label,
                        "entity_name": entity_name,
                        "rel_pos": int(rel_pos),
                        "score": float(score),
                    })
    if not rows:
        return pd.DataFrame(columns=["experiment_label", "rel_pos", "mean_score", "std_score", "count"])
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["experiment_label", "rel_pos"], as_index=False).agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        count=("score", "size"),
    )
    summary["std_score"] = summary["std_score"].fillna(0.0)
    return summary


def add_bms_slice_labels(df, current_idx, high_quantile, low_quantile):
    current_values = df[f"True_{current_idx}"].astype(float).to_numpy()
    delta = np.zeros(len(current_values), dtype=np.float32)
    if len(current_values) > 1:
        delta[1:] = np.abs(np.diff(current_values))
        delta[0] = delta[1]
    if len(delta) == 0:
        df["slice_label"] = []
        return df

    high_thr = float(np.quantile(delta, high_quantile))
    low_thr = float(np.quantile(delta, low_quantile))
    labels = np.full(len(delta), "middle", dtype=object)
    labels[delta >= high_thr] = "high_frequency_regulation"
    labels[delta <= low_thr] = "relative_steady"

    df["delta_current_abs"] = delta
    df["slice_label"] = labels
    return df


def iqr(series):
    if len(series) == 0:
        return math.nan
    q75, q25 = np.percentile(np.asarray(series, dtype=np.float32), [75, 25])
    return float(q75 - q25)


def summarize_slices(df, group_cols, metrics):
    rows = []
    for keys, group_df in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["count"] = int(len(group_df))
        for metric in metrics:
            values = group_df[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
            row[f"{metric}_median"] = float(np.median(values)) if len(values) else math.nan
            row[f"{metric}_iqr"] = iqr(values)
        rows.append(row)
    return pd.DataFrame(rows)


def write_table(df, csv_path):
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path = csv_path.with_suffix(".md")
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def plot_slice_boxpanels(df, slice_order, output_path, title):
    if df.empty:
        return
    experiments = list(dict.fromkeys(df["experiment_label"].tolist()))
    fig, axes = plt.subplots(1, len(slice_order), figsize=(4.8 * len(slice_order), 4.8), sharey=True)
    if len(slice_order) == 1:
        axes = [axes]

    for ax, slice_name in zip(axes, slice_order):
        slice_df = df[df["slice_label"] == slice_name]
        data = []
        labels = []
        for exp in experiments:
            exp_values = slice_df.loc[slice_df["experiment_label"] == exp, "A_Score_Global"].astype(float).to_numpy()
            if len(exp_values) == 0:
                continue
            data.append(exp_values)
            labels.append(exp)
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(zh_slice_label(slice_name))
        ax.set_xlabel("实验方案")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    axes[0].set_ylabel("全局异常分数")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_transition_profiles(profile_df, output_path):
    if profile_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for experiment_label, exp_df in profile_df.groupby("experiment_label", sort=False):
        exp_df = exp_df.sort_values("rel_pos")
        ax.plot(exp_df["rel_pos"], exp_df["mean_score"], linewidth=1.5, label=experiment_label)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("相对切换位置")
    ax.set_ylabel("平均全局异常分数")
    ax.set_title("NASA 工况切换分数轮廓")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_metric(summary_df, output_path, title, metric_col, slice_order):
    if summary_df.empty:
        return
    pivot = (
        summary_df.groupby(["experiment_label", "slice_label"], as_index=False)[metric_col]
        .mean()
        .pivot(index="experiment_label", columns="slice_label", values=metric_col)
        .reindex(columns=slice_order)
        .fillna(0.0)
    )
    experiments = list(pivot.index)
    x = np.arange(len(experiments), dtype=np.float32)
    width = 0.35 if len(slice_order) <= 2 else 0.18

    fig, ax = plt.subplots(figsize=(max(8, len(experiments) * 1.6), 4.8))
    for idx, slice_name in enumerate(slice_order):
        if slice_name not in pivot.columns:
            continue
        offset = (idx - (len(slice_order) - 1) / 2.0) * width
        ax.bar(x + offset, pivot[slice_name].to_numpy(), width=width, label=slice_name)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=20, ha="right")
    ax.set_ylabel(zh_metric_label(metric_col))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bms_branch_contribution(summary_df, output_path):
    if summary_df.empty:
        return
    target = summary_df[summary_df["experiment_label"].isin(["C4+Hier", "C4-StructureOnly"])].copy()
    if target.empty:
        return

    metrics = [
        "A_Score_Main_Branch_mean",
        "A_Score_Residual_Branch_mean",
        "A_Score_Global_mean",
    ]
    fig, axes = plt.subplots(1, len(BMS_SLICE_ORDER), figsize=(5.5 * len(BMS_SLICE_ORDER), 4.8), sharey=True)
    if len(BMS_SLICE_ORDER) == 1:
        axes = [axes]
    for ax, slice_name in zip(axes, BMS_SLICE_ORDER):
        slice_df = target[target["slice_label"] == slice_name]
        x = np.arange(len(slice_df), dtype=np.float32)
        width = 0.22
        for idx, metric in enumerate(metrics):
            if metric not in slice_df.columns:
                continue
            offset = (idx - 1) * width
            label_map = {
                "A_Score_Main_Branch_mean": "主分支",
                "A_Score_Residual_Branch_mean": "残差分支",
                "A_Score_Global_mean": "融合后全局分数",
            }
            ax.bar(x + offset, slice_df[metric].to_numpy(), width=width, label=label_map.get(metric, metric.replace("_mean", "")))
        ax.set_xticks(x)
        ax.set_xticklabels(slice_df["experiment_label"].tolist(), rotation=20, ha="right")
        ax.set_title(zh_slice_label(slice_name))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("平均分数")
    axes[0].legend(frameon=False)
    fig.suptitle("BMS 各切片分支贡献对比")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def preferred_nasa_experiments(df):
    preferred = ["C3", "C3+PhysEnc", "C3+PhysReg", "C4"]
    available = set(df["experiment_label"].unique())
    ordered = [item for item in preferred if item in available]
    if ordered:
        return ordered
    return sorted(available)


def preferred_bms_experiments(df):
    preferred = ["C3", "C3+PhysEnc", "C3+PhysReg", "C4", "C4-StructureOnly", "C4+Hier"]
    available = set(df["experiment_label"].unique())
    ordered = [item for item in preferred if item in available]
    if ordered:
        return ordered
    return sorted(available)


def choose_nasa_case_center(df, target_slice, metric):
    target_df = df[df["slice_label"] == target_slice].copy()
    if target_df.empty:
        return None

    if {"C3", "C4"}.issubset(set(target_df["experiment_label"].unique())):
        base_df = target_df[target_df["experiment_label"] == "C3"][
            ["entity_name", "Global_Pos", metric]
        ].rename(columns={metric: f"{metric}_base"})
        phys_df = target_df[target_df["experiment_label"] == "C4"][
            ["entity_name", "Global_Pos", metric]
        ].rename(columns={metric: f"{metric}_phys"})
        merged = base_df.merge(phys_df, on=["entity_name", "Global_Pos"], how="inner")
        if not merged.empty:
            merged["improvement"] = merged[f"{metric}_base"] - merged[f"{metric}_phys"]
            best = merged.sort_values(["improvement", f"{metric}_base"], ascending=[False, False]).iloc[0]
            return best["entity_name"], int(best["Global_Pos"])

    fallback_label = "C3" if "C3" in set(target_df["experiment_label"].unique()) else target_df["experiment_label"].iloc[0]
    best = (
        target_df[target_df["experiment_label"] == fallback_label]
        .sort_values(metric, ascending=False)
        .iloc[0]
    )
    return best["entity_name"], int(best["Global_Pos"])


def choose_bms_case_center(df):
    candidates = df.copy()
    if "C4+Hier" in set(candidates["experiment_label"].unique()):
        candidates = candidates[candidates["experiment_label"] == "C4+Hier"].copy()
    elif "C4" in set(candidates["experiment_label"].unique()):
        candidates = candidates[candidates["experiment_label"] == "C4"].copy()
    if "high_frequency_regulation" in set(candidates["slice_label"].unique()):
        high_df = candidates[candidates["slice_label"] == "high_frequency_regulation"]
        if not high_df.empty:
            candidates = high_df
    metric = "A_Score_Residual_Branch" if "A_Score_Residual_Branch" in candidates.columns else "A_Score_Global"
    best = candidates.sort_values(metric, ascending=False).iloc[0]
    return best["entity_name"], int(best["Global_Pos"])


def plot_nasa_case(df, output_path, title, entity_name, center_pos, radius):
    entity_df = df[df["entity_name"] == entity_name].copy()
    if entity_df.empty:
        return
    local_df = entity_df[
        (entity_df["Global_Pos"] >= center_pos - radius) & (entity_df["Global_Pos"] <= center_pos + radius)
    ].copy()
    if local_df.empty:
        return

    experiments = preferred_nasa_experiments(local_df)
    reference_df = local_df[local_df["experiment_label"] == experiments[0]].sort_values("Global_Pos")
    x = reference_df["Global_Pos"].to_numpy()

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(x, reference_df["Raw_step_type_code"].to_numpy(), color="black", linewidth=1.2)
    axes[0].set_ylabel("工况编码")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)

    for exp in experiments:
        exp_df = local_df[local_df["experiment_label"] == exp].sort_values("Global_Pos")
        axes[1].plot(exp_df["Global_Pos"], exp_df["A_Score_Global"], linewidth=1.4, label=exp)
    axes[1].set_ylabel("异常分数")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)

    axes[2].plot(x, reference_df["Raw_voltage"].to_numpy(), color="black", linewidth=1.2, label="真实电压")
    for exp in experiments:
        exp_df = local_df[local_df["experiment_label"] == exp].sort_values("Global_Pos")
        axes[2].plot(exp_df["Global_Pos"], exp_df["Recon_1"], linewidth=1.1, label=f"{exp} 重构")
    axes[2].set_ylabel("电压")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, ncol=2)

    axes[3].plot(x, reference_df["Raw_temperature"].to_numpy(), color="black", linewidth=1.2, label="真实温度")
    for exp in experiments:
        exp_df = local_df[local_df["experiment_label"] == exp].sort_values("Global_Pos")
        axes[3].plot(exp_df["Global_Pos"], exp_df["Recon_3"], linewidth=1.1, label=f"{exp} 重构")
    axes[3].set_ylabel("温度")
    axes[3].set_xlabel("全局位置")
    axes[3].grid(alpha=0.25)
    axes[3].legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bms_case(df, output_path, title, entity_name, center_pos, radius):
    entity_df = df[df["entity_name"] == entity_name].copy()
    if entity_df.empty:
        return
    local_df = entity_df[
        (entity_df["Global_Pos"] >= center_pos - radius) & (entity_df["Global_Pos"] <= center_pos + radius)
    ].copy()
    if local_df.empty:
        return

    feature_names = get_bms_feature_names()
    voltage_idx = feature_names.index("BMSnVmean")
    temperature_idx = feature_names.index("BMSnTmean")
    experiments = preferred_bms_experiments(local_df)
    reference_df = local_df[local_df["experiment_label"] == experiments[0]].sort_values("Global_Pos")
    x = reference_df["Global_Pos"].to_numpy()

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, reference_df["Raw_BMSnI"].to_numpy(), color="black", linewidth=1.2)
    axes[0].set_ylabel("电流")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)

    for exp in experiments:
        exp_df = local_df[local_df["experiment_label"] == exp].sort_values("Global_Pos")
        if exp in {"C4", "C4+Hier", "C4-StructureOnly"}:
            axes[1].plot(exp_df["Global_Pos"], exp_df["A_Score_Global"], linewidth=1.5, label=f"{exp} 全局")
    c4_df = local_df[local_df["experiment_label"] == "C4"].sort_values("Global_Pos")
    if not c4_df.empty:
        axes[1].plot(c4_df["Global_Pos"], c4_df["A_Score_Main_Branch"], linewidth=1.2, linestyle="--", label="C4 主分支")
        axes[1].plot(c4_df["Global_Pos"], c4_df["A_Score_Residual_Branch"], linewidth=1.2, linestyle=":", label="C4 残差分支")
    axes[1].set_ylabel("异常分数")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)

    axes[2].plot(x, reference_df["Raw_BMSnVmean"].to_numpy(), color="black", linewidth=1.2, label="真实电压")
    for exp in experiments:
        if exp not in {"C4", "C4+Hier", "C4-StructureOnly"}:
            continue
        exp_df = local_df[local_df["experiment_label"] == exp].sort_values("Global_Pos")
        axes[2].plot(exp_df["Global_Pos"], exp_df[f"Recon_{voltage_idx}"], linewidth=1.1, label=f"{exp} 重构")
    axes[2].set_ylabel("电压")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, ncol=2)

    axes[3].plot(x, reference_df["Raw_BMSnTmean"].to_numpy(), color="black", linewidth=1.2, label="真实温度")
    for exp in experiments:
        if exp not in {"C4", "C4+Hier", "C4-StructureOnly"}:
            continue
        exp_df = local_df[local_df["experiment_label"] == exp].sort_values("Global_Pos")
        axes[3].plot(exp_df["Global_Pos"], exp_df[f"Recon_{temperature_idx}"], linewidth=1.1, label=f"{exp} 重构")
    axes[3].set_ylabel("温度")
    axes[3].set_xlabel("全局位置")
    axes[3].grid(alpha=0.25)
    axes[3].legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_score_dataframe(score_path):
    return pd.read_pickle(score_path)


def load_nasa_cases(experiment):
    dataset = str(experiment["dataset"]).upper()
    args = experiment.get("args", {})
    battery_id = args.get("nasa_battery_id", "")
    if not battery_id:
        return []

    output_dir = Path(experiment["output_dir"])
    score_path = output_dir / "test_output.pkl"
    if not score_path.exists():
        return []

    config_path = output_dir / "config.txt"
    config = load_json(config_path) if config_path.exists() else args
    window_size = int(config.get("lookback", args.get("lookback", 100)))

    raw_path = ROOT / "datasets" / dataset / "processed" / f"{dataset}_{battery_id}_test.pkl"
    if not raw_path.exists():
        return []

    score_df = load_score_dataframe(score_path)
    raw_df = build_aligned_raw_dataframe(load_pickle(raw_path), window_size, NASA_RANDOM_FEATURE_NAMES)
    merged = merge_score_and_raw(score_df, raw_df)
    merged["experiment_name"] = experiment["name"]
    merged["experiment_label"] = normalize_experiment_label(experiment)
    merged["dataset"] = dataset
    merged["entity_name"] = battery_id
    merged["case_path"] = str(output_dir)
    merged = add_common_metrics(merged, voltage_idx=1, current_idx=2, temperature_idx=3)
    merged = add_nasa_slice_labels(merged, transition_margin=args_namespace.transition_margin)
    return [merged]


def load_bms_cases(experiment):
    args = experiment.get("args", {})
    output_dir = Path(experiment["output_dir"])
    config_path = output_dir / "config.txt"
    config = load_json(config_path) if config_path.exists() else args
    window_size = int(config.get("lookback", args.get("lookback", 100)))

    feature_names = get_bms_feature_names()
    voltage_idx = feature_names.index("BMSnVmean")
    current_idx = feature_names.index("BMSnI")
    temperature_idx = feature_names.index("BMSnTmean")

    cases = []
    for score_path in sorted(output_dir.rglob("test_output.pkl")):
        cluster_dir = score_path.parent
        cluster_name = cluster_dir.name
        raw_path = ROOT / "datasets" / "BMS" / "processed" / f"{cluster_name}_test.pkl"
        if not raw_path.exists():
            continue
        score_df = load_score_dataframe(score_path)
        raw_df = build_aligned_raw_dataframe(load_pickle(raw_path), window_size, feature_names)
        merged = merge_score_and_raw(score_df, raw_df)
        merged["experiment_name"] = experiment["name"]
        merged["experiment_label"] = normalize_experiment_label(experiment)
        merged["dataset"] = "BMS"
        merged["entity_name"] = cluster_name
        merged["case_path"] = str(cluster_dir)
        merged = add_common_metrics(merged, voltage_idx=voltage_idx, current_idx=current_idx, temperature_idx=temperature_idx)
        merged = add_bms_slice_labels(
            merged,
            current_idx=current_idx,
            high_quantile=args_namespace.reg_high_quantile,
            low_quantile=args_namespace.reg_low_quantile,
        )
        cases.append(merged)
    return cases


def collect_experiments(registries):
    experiments = []
    for registry_path in registries:
        payload = load_json(registry_path)
        for experiment in payload.get("experiments", []):
            if experiment.get("status") not in {"done", "skipped_existing"}:
                continue
            experiments.append(experiment)
    return experiments


def is_experiment_dir(path):
    if not path.is_dir():
        return False
    if not (path / "config.txt").exists():
        return False
    if (path / "test_output.pkl").exists():
        return True
    return any(path.glob("*/test_output.pkl"))


def build_experiment_from_dir(exp_dir):
    config_path = exp_dir / "config.txt"
    args = load_json(config_path) if config_path.exists() else {}
    dataset = str(args.get("dataset") or infer_dataset_from_name(exp_dir.name)).upper()
    return {
        "name": canonical_experiment_name(exp_dir.name),
        "dataset": dataset,
        "run_id": args.get("run_id", canonical_experiment_name(exp_dir.name)),
        "output_dir": str(exp_dir.resolve()),
        "args": args,
        "status": "done",
    }


def collect_experiments_from_sources(source_paths):
    experiments = []
    seen_dirs = set()
    seen_names = set()

    for source_item in source_paths:
        source_path = Path(source_item).resolve()
        if not source_path.exists():
            continue

        if source_path.is_file() and source_path.suffix.lower() == ".json":
            experiments.extend(collect_experiments([str(source_path)]))
            continue

        if source_path.is_dir() and is_experiment_dir(source_path):
            key = str(source_path)
            if key not in seen_dirs:
                experiments.append(build_experiment_from_dir(source_path))
                seen_dirs.add(key)
            continue

        if source_path.is_dir():
            config_paths = sorted(source_path.rglob("config.txt"))
            for config_path in config_paths:
                exp_dir = config_path.parent
                if "analysis" in exp_dir.parts:
                    continue
                if not is_experiment_dir(exp_dir):
                    continue
                canonical_name = canonical_experiment_name(exp_dir.name)
                dir_key = str(exp_dir.resolve())
                if canonical_name in seen_names or dir_key in seen_dirs:
                    continue
                experiments.append(build_experiment_from_dir(exp_dir))
                seen_names.add(canonical_name)
                seen_dirs.add(dir_key)

    return experiments


def analyze_nasa(experiments, output_dir):
    nasa_frames = []
    for experiment in experiments:
        if str(experiment.get("dataset", "")).upper() != "NASA_RANDOM_DISCHARGE":
            continue
        nasa_frames.extend(load_nasa_cases(experiment))

    nasa_dir = output_dir / "nasa_random"
    ensure_dir(nasa_dir)
    if not nasa_frames:
        return {"rows": 0, "dir": str(nasa_dir)}

    nasa_df = pd.concat(nasa_frames, axis=0, ignore_index=True)
    nasa_df = nasa_df[nasa_df["slice_label"].isin(NASA_SLICE_ORDER)].copy()

    metrics = [
        "A_Score_Global",
        "Pred_Error_Global",
        "Recon_Error_Global",
        "Voltage_Delta_Error",
        "Temperature_Delta_Error",
        "Current_Cum_Error",
    ]
    entity_summary = summarize_slices(
        nasa_df,
        ["experiment_label", "entity_name", "slice_label"],
        metrics,
    )
    overall_summary = summarize_slices(
        nasa_df,
        ["experiment_label", "slice_label"],
        metrics,
    )
    transition_profile = compute_transition_profile(nasa_df, radius=args_namespace.transition_radius)

    write_table(entity_summary, nasa_dir / "nasa_phase_slice_entity_summary.csv")
    write_table(overall_summary, nasa_dir / "nasa_phase_slice_overall_summary.csv")
    write_table(transition_profile, nasa_dir / "nasa_phase_switch_profile.csv")

    plot_slice_boxpanels(
        nasa_df[["experiment_label", "slice_label", "A_Score_Global"]].copy(),
        NASA_SLICE_ORDER,
        nasa_dir / "nasa_phase_score_boxplots.png",
        "NASA 各工况切片异常分数分布",
    )
    plot_grouped_metric(
        overall_summary,
        nasa_dir / "nasa_phase_error_summary.png",
        "NASA 各工况切片重构误差汇总",
        "Recon_Error_Global_mean",
        NASA_SLICE_ORDER,
    )
    plot_transition_profiles(
        transition_profile,
        nasa_dir / "nasa_phase_switch_profile.png",
    )

    transition_case = choose_nasa_case_center(nasa_df, target_slice="transition", metric="A_Score_Global")
    if transition_case is not None:
        plot_nasa_case(
            nasa_df,
            nasa_dir / "nasa_case_transition.png",
            "NASA 典型工况切换案例",
            entity_name=transition_case[0],
            center_pos=transition_case[1],
            radius=args_namespace.case_radius,
        )

    stable_case = choose_nasa_case_center(nasa_df, target_slice="stable_discharge", metric="Recon_Error_Global")
    if stable_case is not None:
        plot_nasa_case(
            nasa_df,
            nasa_dir / "nasa_case_stable.png",
            "NASA 典型稳定工况案例",
            entity_name=stable_case[0],
            center_pos=stable_case[1],
            radius=args_namespace.case_radius,
        )

    return {
        "rows": int(len(nasa_df)),
        "dir": str(nasa_dir),
        "experiments": sorted(nasa_df["experiment_label"].unique().tolist()),
    }


def analyze_bms(experiments, output_dir):
    bms_frames = []
    for experiment in experiments:
        if str(experiment.get("dataset", "")).upper() != "BMS":
            continue
        bms_frames.extend(load_bms_cases(experiment))

    bms_dir = output_dir / "bms"
    ensure_dir(bms_dir)
    if not bms_frames:
        return {"rows": 0, "dir": str(bms_dir)}

    bms_df = pd.concat(bms_frames, axis=0, ignore_index=True)
    bms_df = bms_df[bms_df["slice_label"].isin(BMS_SLICE_ORDER)].copy()

    metrics = [
        "A_Score_Global",
        "Pred_Error_Global",
        "Recon_Error_Global",
        "Voltage_Delta_Error",
        "Temperature_Delta_Error",
        "Current_Cum_Error",
        "A_Score_Main_Branch",
        "A_Score_Residual_Branch",
    ]
    entity_summary = summarize_slices(
        bms_df,
        ["experiment_label", "entity_name", "slice_label"],
        metrics,
    )
    overall_summary = summarize_slices(
        bms_df,
        ["experiment_label", "slice_label"],
        metrics,
    )

    write_table(entity_summary, bms_dir / "bms_regulation_slice_entity_summary.csv")
    write_table(overall_summary, bms_dir / "bms_regulation_slice_overall_summary.csv")

    plot_slice_boxpanels(
        bms_df[["experiment_label", "slice_label", "A_Score_Global"]].copy(),
        BMS_SLICE_ORDER,
        bms_dir / "bms_regulation_score_boxplots.png",
        "BMS 各调节切片异常分数分布",
    )
    plot_grouped_metric(
        overall_summary,
        bms_dir / "bms_regulation_error_summary.png",
        "BMS 各调节切片重构误差汇总",
        "Recon_Error_Global_mean",
        BMS_SLICE_ORDER,
    )
    plot_bms_branch_contribution(
        overall_summary,
        bms_dir / "bms_branch_contribution.png",
    )

    bms_case = choose_bms_case_center(bms_df)
    if bms_case is not None:
        plot_bms_case(
            bms_df,
            bms_dir / "bms_case_typical_anomaly.png",
            "BMS 典型异常案例",
            entity_name=bms_case[0],
            center_pos=bms_case[1],
            radius=args_namespace.case_radius,
        )

    return {
        "rows": int(len(bms_df)),
        "dir": str(bms_dir),
        "experiments": sorted(bms_df["experiment_label"].unique().tolist()),
    }


def build_manifest(output_dir, registries, sources, nasa_result, bms_result):
    manifest = {
        "registries": [str(Path(p).resolve()) for p in registries],
        "sources": [str(Path(p).resolve()) for p in sources],
        "output_dir": str(output_dir),
        "nasa_random": nasa_result,
        "bms": bms_result,
        "generated_files": sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()),
    }
    with open(output_dir / "analysis_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


args_namespace = None


def main():
    global args_namespace
    args_namespace = parse_args()

    if not args_namespace.registries and not args_namespace.sources:
        raise ValueError("At least one of --registries or --sources must be provided")

    registry_paths = [str(Path(item).resolve()) for item in args_namespace.registries]
    source_paths = [str(Path(item).resolve()) for item in args_namespace.sources]
    if args_namespace.output_dir:
        output_dir = Path(args_namespace.output_dir).resolve()
    else:
        output_dir = ROOT / "analysis" / "ch4_condition_slices"
    ensure_dir(output_dir)

    experiments = []
    if registry_paths:
        experiments.extend(collect_experiments(registry_paths))
    if source_paths:
        experiments.extend(collect_experiments_from_sources(source_paths))

    deduped = []
    seen = set()
    for experiment in experiments:
        output_dir_key = str(Path(experiment["output_dir"]).resolve())
        if output_dir_key in seen:
            continue
        seen.add(output_dir_key)
        deduped.append(experiment)
    experiments = deduped

    nasa_result = analyze_nasa(experiments, output_dir)
    bms_result = analyze_bms(experiments, output_dir)
    build_manifest(output_dir, registry_paths, source_paths, nasa_result, bms_result)

    print(f"[DONE] analysis output: {output_dir}")
    print(f"[NASA_RANDOM] rows={nasa_result['rows']} experiments={nasa_result.get('experiments', [])}")
    print(f"[BMS] rows={bms_result['rows']} experiments={bms_result.get('experiments', [])}")


if __name__ == "__main__":
    main()
