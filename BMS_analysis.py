# BMS 数据集可视化分析

import glob
import json
import os
import pickle
import re
import sys

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 设置中文字体以避免警告
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "")
from utils import get_bms_feature_names, get_data, get_data_dim


PROCESSED_DIR = os.path.join("datasets", "BMS", "processed")
OUTPUT_DIR = os.path.join("output", "BMS_analysis")

# 这是第一版“建议删减”的候选，不会自动改预处理，只用于分析图和文字提示。
RECOMMENDED_DROP_CANDIDATES = [
    "BMSnETmax",
    "BMSnETmean",
    "BMSnSOH",
    "SYS_SOH",
    "BMSnVol_T",
    "BMSnVol_B",
    "SYS_Vol",
    "SYS_I",
    "BMSnTmin",
]


def create_feature_axes(n_features, n_cols=4, panel_height=3.1, width=18):
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, panel_height * n_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    return fig, axes


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_and_show(fig, filename):
    ensure_output_dir()
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"图已保存: {save_path}")
    plt.close(fig)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def cluster_sort_key(train_path):
    basename = os.path.basename(train_path)
    match = re.search(r"cluster(\d+)_train\.pkl$", basename)
    cluster_id = int(match.group(1)) if match else 9999
    bundle_name = basename.split("_cluster")[0]
    return bundle_name, cluster_id, basename


def build_cluster_label(train_path):
    basename = os.path.basename(train_path).replace("_train.pkl", "")
    match = re.search(r"(cluster\d+)$", basename)
    return match.group(1) if match else basename


def load_cluster_splits(processed_dir):
    cluster_train_files = sorted(
        glob.glob(os.path.join(processed_dir, "BMS_*_cluster*_train.pkl")),
        key=cluster_sort_key,
    )
    cluster_splits = []
    for train_path in cluster_train_files:
        base_path = train_path[: -len("_train.pkl")]
        test_path = base_path + "_test.pkl"
        label_path = base_path + "_test_label.pkl"
        cluster_splits.append(
            {
                "name": build_cluster_label(train_path),
                "train": load_pickle(train_path),
                "test": load_pickle(test_path) if os.path.exists(test_path) else None,
                "test_label": load_pickle(label_path) if os.path.exists(label_path) else None,
                "train_path": train_path,
            }
        )
    return cluster_splits


def print_dataset_overview(train_data, test_data, test_labels, cluster_splits, feature_names):
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    print(f"测试标签形状: {None if test_labels is None else test_labels.shape}")
    print(f"特征数量: {len(feature_names)}")
    print(f"特征名称: {feature_names}")
    print(f"按簇读取到的序列数: {len(cluster_splits)}")
    if cluster_splits:
        for cluster_info in cluster_splits:
            train_shape = None if cluster_info["train"] is None else cluster_info["train"].shape
            test_shape = None if cluster_info["test"] is None else cluster_info["test"].shape
            print(f"  - {cluster_info['name']}: train={train_shape}, test={test_shape}")


def plot_cluster_feature_trends(cluster_splits, feature_names, split_name, max_points=1500):
    if not cluster_splits:
        return

    fig, axes = create_feature_axes(len(feature_names), panel_height=3.2)
    fig.suptitle(f"BMS {split_name} 按簇分开趋势图", fontsize=16)
    colors = plt.get_cmap("tab10", len(cluster_splits))

    for feature_idx, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        for cluster_idx, cluster_info in enumerate(cluster_splits):
            data = cluster_info[split_name]
            if data is None:
                continue
            plot_len = min(len(data), max_points)
            x_axis = np.arange(plot_len)
            ax.plot(
                x_axis,
                data[:plot_len, feature_idx],
                linewidth=1.0,
                color=colors(cluster_idx),
                alpha=0.9,
                label=cluster_info["name"],
            )
        ax.set_title(f"{feature_name}")
        ax.set_xlabel("簇内时间点")
        ax.set_ylabel(feature_name)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(feature_names):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False)
        fig.subplots_adjust(top=0.92)
    save_and_show(fig, f"cluster_{split_name}_trends.png")


def plot_distribution_comparison(train_data, test_data, feature_names):
    fig, axes = create_feature_axes(len(feature_names), panel_height=3.0)
    fig.suptitle("BMS 训练集与测试集分布对比", fontsize=16)

    for feature_idx, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        ax.hist(train_data[:, feature_idx], bins=40, alpha=0.55, label="train", color="#4C72B0")
        ax.hist(test_data[:, feature_idx], bins=40, alpha=0.45, label="test", color="#55A868")
        ax.set_title(feature_name)
        ax.set_xlabel(feature_name)
        ax.set_ylabel("频数")
        ax.grid(True, alpha=0.25)

    for ax in axes[len(feature_names):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.subplots_adjust(top=0.92)
    save_and_show(fig, "train_test_distribution.png")


def plot_test_anomalies(test_data, test_labels, feature_names):
    if test_labels is None:
        print("测试标签不存在，跳过异常点散点图。")
        return

    anomaly_indices = np.where(test_labels == 1)[0]
    normal_indices = np.where(test_labels == 0)[0]

    print(f"测试集中异常点数量: {len(anomaly_indices)}")
    print(f"测试集中正常点数量: {len(normal_indices)}")
    print(f"异常点占比: {len(anomaly_indices) / len(test_labels) * 100:.2f}%")

    if len(anomaly_indices) == 0:
        print("当前 BMS test_label 全为 0，占位标签，不绘制异常点对比图。")
        return

    fig, axes = create_feature_axes(len(feature_names), panel_height=3.5)
    fig.suptitle("异常点在各特征上的值分布", fontsize=16)

    for feature_idx, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        ax.scatter(
            normal_indices,
            test_data[normal_indices, feature_idx],
            c="#4C72B0",
            s=2,
            alpha=0.45,
            label="正常点",
        )
        ax.scatter(
            anomaly_indices,
            test_data[anomaly_indices, feature_idx],
            c="#C44E52",
            s=8,
            alpha=0.8,
            label="异常点",
        )
        ax.set_title(feature_name)
        ax.set_xlabel("时间点")
        ax.set_ylabel(feature_name)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(feature_names):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.subplots_adjust(top=0.92)
    save_and_show(fig, "test_anomaly_scatter.png")


def plot_correlation_heatmap(data, feature_names, title, filename):
    corr = pd.DataFrame(data, columns=feature_names).corr()
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title)
    save_and_show(fig, filename)
    return corr


def summarize_feature_redundancy(train_data, feature_names):
    train_df = pd.DataFrame(train_data, columns=feature_names)
    std_series = train_df.std().sort_values()
    corr_matrix = train_df.corr().abs()
    np.fill_diagonal(corr_matrix.values, np.nan)

    high_corr_pairs = []
    for i, left_name in enumerate(feature_names):
        for j in range(i + 1, len(feature_names)):
            right_name = feature_names[j]
            corr_value = corr_matrix.iloc[i, j]
            if pd.notna(corr_value) and corr_value >= 0.98:
                high_corr_pairs.append((corr_value, left_name, right_name))
    high_corr_pairs.sort(reverse=True)

    core_features = [name for name in feature_names if name not in RECOMMENDED_DROP_CANDIDATES]

    print("\n=== 建议优先删除的量（第一版）===")
    for feature_name in RECOMMENDED_DROP_CANDIDATES:
        if feature_name in std_series.index:
            print(
                f"- {feature_name}: std={std_series[feature_name]:.6f}, "
                f"缺少新增信息或与其他变量高度重复"
            )

    print("\n=== 方差最小的前 10 个特征 ===")
    for feature_name, std_value in std_series.head(10).items():
        print(f"- {feature_name}: std={std_value:.6f}")

    print("\n=== 绝对相关系数 >= 0.98 的前 15 个特征对 ===")
    for corr_value, left_name, right_name in high_corr_pairs[:15]:
        print(f"- {left_name} <-> {right_name}: |corr|={corr_value:.4f}")

    print("\n=== 建议保留的核心特征 ===")
    print(core_features)

    return core_features


def plot_cluster_mean_heatmap(cluster_splits, feature_names, split_name, filename):
    if not cluster_splits:
        return

    mean_rows = []
    row_names = []
    for cluster_info in cluster_splits:
        data = cluster_info[split_name]
        if data is None:
            continue
        mean_rows.append(np.mean(data, axis=0))
        row_names.append(cluster_info["name"])

    if not mean_rows:
        return

    mean_df = pd.DataFrame(mean_rows, index=row_names, columns=feature_names)
    fig, ax = plt.subplots(figsize=(16, 4 + 0.35 * len(row_names)))
    sns.heatmap(mean_df, cmap="viridis", linewidths=0.3, ax=ax)
    ax.set_title(f"BMS {split_name} 各簇均值热力图")
    ax.set_xlabel("特征")
    ax.set_ylabel("簇")
    save_and_show(fig, filename)


def load_bms_data():
    try:
        (train_data, _), (test_data, test_labels) = get_data("BMS", normalize=False)
    except FileNotFoundError as exc:
        print(f"数据文件未找到: {exc}")
        print("请先运行 `python preprocess.py --dataset BMS` 生成数据。")
        sys.exit(1)
    return train_data, test_data, test_labels


def get_latest_bms_run_dir():
    output_path = os.path.join("output", "BMS")
    if not os.path.exists(output_path):
        return None

    run_dirs = []
    for entry in os.listdir(output_path):
        run_path = os.path.join(output_path, entry)
        if not os.path.isdir(run_path):
            continue
        if entry == "logs":
            continue
        run_dirs.append(run_path)

    if not run_dirs:
        return None

    run_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return run_dirs[0]


def find_cluster_output_dirs(run_dir):
    if run_dir is None or not os.path.exists(run_dir):
        return []

    cluster_dirs = []
    for entry in sorted(os.listdir(run_dir)):
        entry_path = os.path.join(run_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if os.path.exists(os.path.join(entry_path, "test_output.pkl")):
            cluster_dirs.append(entry_path)
    return cluster_dirs


def plot_cluster_output_trends(cluster_output_dirs):
    if not cluster_output_dirs:
        print("未检测到按簇输出目录，跳过簇级模型结果分析。")
        return

    summary_rows = []
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)
    metric_configs = [
        ("A_Score_Global", "Global Anomaly Score", axes[0], "cluster_test_anomaly_scores.png"),
        ("Pred_Error_Global", "Global Prediction Error", axes[1], None),
        ("Recon_Error_Global", "Global Reconstruction Error", axes[2], None),
    ]

    for cluster_idx, cluster_dir in enumerate(cluster_output_dirs):
        cluster_name = os.path.basename(cluster_dir)
        test_output_path = os.path.join(cluster_dir, "test_output.pkl")
        summary_path = os.path.join(cluster_dir, "summary_metrics.json")
        threshold_path = os.path.join(cluster_dir, "thresholds.json")

        if not os.path.exists(test_output_path):
            continue

        test_output = pd.read_pickle(test_output_path)
        threshold_value = None
        if os.path.exists(threshold_path):
            with open(threshold_path, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
            threshold_value = thresholds.get("global_threshold")

        summary_metrics = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_metrics = json.load(f)

        for metric_name, title, ax, _ in metric_configs:
            if metric_name not in test_output.columns:
                continue
            ax.plot(
                np.arange(len(test_output)),
                test_output[metric_name].values,
                linewidth=1.0,
                alpha=0.9,
                label=cluster_name,
            )
            ax.set_title(title)
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.25)

            if metric_name == "A_Score_Global" and threshold_value is not None:
                ax.axhline(float(threshold_value), color="black", linestyle="--", linewidth=0.8, alpha=0.35)

        summary_rows.append(
            {
                "cluster": cluster_name,
                "test_points": int(len(test_output)),
                "mean_global_score": float(test_output["A_Score_Global"].mean()) if "A_Score_Global" in test_output else np.nan,
                "max_global_score": float(test_output["A_Score_Global"].max()) if "A_Score_Global" in test_output else np.nan,
                "mean_pred_error": float(test_output["Pred_Error_Global"].mean()) if "Pred_Error_Global" in test_output else np.nan,
                "mean_recon_error": float(test_output["Recon_Error_Global"].mean()) if "Recon_Error_Global" in test_output else np.nan,
                "global_threshold": float(threshold_value) if threshold_value is not None else np.nan,
                "epsilon_threshold": (
                    summary_metrics.get("epsilon_result", {}).get("threshold")
                    if isinstance(summary_metrics, dict) else np.nan
                ),
            }
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False)
        fig.subplots_adjust(top=0.90)
    save_and_show(fig, "cluster_model_output_trends.png")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("cluster")
        summary_csv_path = os.path.join(OUTPUT_DIR, "cluster_model_output_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
        print(f"模型输出汇总已保存: {summary_csv_path}")
        print("\n=== BMS 按簇模型输出摘要 ===")
        print(summary_df.to_string(index=False))


def plot_model_outputs_if_available():
    latest_run_dir = get_latest_bms_run_dir()
    if latest_run_dir is None:
        print("未找到 BMS 训练结果目录，跳过模型输出分析。")
        return

    print(f"检测到最新 BMS 运行目录: {latest_run_dir}")
    cluster_output_dirs = find_cluster_output_dirs(latest_run_dir)
    plot_cluster_output_trends(cluster_output_dirs)


def main():
    ensure_output_dir()
    feature_names = get_bms_feature_names()
    train_data, test_data, test_labels = load_bms_data()
    cluster_splits = load_cluster_splits(PROCESSED_DIR)

    print_dataset_overview(train_data, test_data, test_labels, cluster_splits, feature_names)
    plot_cluster_feature_trends(cluster_splits, feature_names, split_name="train")
    plot_cluster_feature_trends(cluster_splits, feature_names, split_name="test")
    plot_distribution_comparison(train_data, test_data, feature_names)
    plot_test_anomalies(test_data, test_labels, feature_names)

    core_features = summarize_feature_redundancy(train_data, feature_names)
    plot_correlation_heatmap(train_data, feature_names, "BMS 训练集全特征相关性热力图", "corr_train_all.png")
    plot_correlation_heatmap(test_data, feature_names, "BMS 测试集全特征相关性热力图", "corr_test_all.png")
    plot_correlation_heatmap(
        train_data[:, [feature_names.index(name) for name in core_features]],
        core_features,
        "BMS 训练集核心特征相关性热力图",
        "corr_train_core.png",
    )
    plot_cluster_mean_heatmap(cluster_splits, feature_names, split_name="train", filename="cluster_train_mean_heatmap.png")
    plot_cluster_mean_heatmap(cluster_splits, feature_names, split_name="test", filename="cluster_test_mean_heatmap.png")

    plot_model_outputs_if_available()


if __name__ == "__main__":
    main()
