import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "kaggle离线output" / "analysis"
OUTPUT_DIR = ANALYSIS_DIR / "paper_artifacts"


PUBLIC_NAME_MAP = {
    "main_smap_c3": "SMAP / C3",
    "main_smap_mtadgat": "SMAP / MTAD-GAT",
    "smap_mtadgat_plus_c3_common_no_regime": "SMAP / C3+ common",
    "main_msl_c3": "MSL / C3",
    "main_msl_mtadgat": "MSL / MTAD-GAT",
    "msl_mtadgat_plus_c3_common_no_regime": "MSL / C3+ common",
}

BMS_NAME_MAP = {
    "main_bms_c3": "BMS / C3",
    "main_bms_mtadgat": "BMS / MTAD-GAT",
    "ch4abl_bms_no_hier": "BMS / No-Hier",
    "ch4abl_bms_c4_w03": "BMS / C4 w=0.3",
    "c4cmp_bms_c4": "BMS / C4",
    "bms_mtadgat_plus_c3_common_no_regime": "BMS / C3+ common",
}


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    results_df = pd.read_csv(ANALYSIS_DIR / "local_output_results_table.csv")
    event_df = pd.read_csv(ANALYSIS_DIR / "internal_event_segment_stats.csv")
    return results_df, event_df


def format_float(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return round(float(value), digits)


def write_markdown_table(df, output_path):
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) if pd.notna(row[col]) else "" for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_internal_public_table(results_df):
    target = results_df[results_df["leaf"].isin(PUBLIC_NAME_MAP.keys())].copy()
    target["display_name"] = target["leaf"].map(PUBLIC_NAME_MAP)
    out = target[[
        "display_name", "epsilon_f1", "pot_f1", "bf_f1", "event_result_f1"
    ]].copy()
    out.columns = ["Experiment", "Epsilon_F1", "POT_F1", "BF_F1", "Event_F1"]
    for col in ["Epsilon_F1", "POT_F1", "BF_F1", "Event_F1"]:
        out[col] = out[col].map(format_float)
    out = out.sort_values("Experiment")
    out.to_csv(OUTPUT_DIR / "table_internal_public_results.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(out, OUTPUT_DIR / "table_internal_public_results.md")
    return out


def build_bms_table(results_df, event_df):
    bms_results = results_df[results_df["family"].isin(BMS_NAME_MAP.keys())].copy()
    bms_summary = (
        bms_results.groupby("family", as_index=False)
        .agg(
            avg_epsilon_threshold=("epsilon_threshold", "mean"),
            avg_pot_threshold=("pot_threshold", "mean"),
            avg_event_raw_positive_count=("event_raw_positive_count", "mean"),
            avg_event_result_positive_count=("event_result_positive_count", "mean"),
        )
    )

    event_df["family"] = event_df["experiment_dir"].map(lambda x: Path(x).parts[-2])
    bms_events = event_df[event_df["family"].isin(BMS_NAME_MAP.keys())].copy()
    event_summary = (
        bms_events.groupby("family", as_index=False)
        .agg(
            avg_raw_event_count=("raw_event_count", "mean"),
            avg_final_event_count=("final_event_count", "mean"),
            avg_delta_event_count=("delta_event_count_event_minus_raw", "mean"),
            avg_raw_positive_points=("raw_positive_points", "mean"),
            avg_final_positive_points=("final_positive_points", "mean"),
        )
    )

    out = bms_summary.merge(event_summary, on="family", how="left")
    out["Experiment"] = out["family"].map(BMS_NAME_MAP)
    out = out[[
        "Experiment",
        "avg_epsilon_threshold",
        "avg_pot_threshold",
        "avg_raw_event_count",
        "avg_final_event_count",
        "avg_delta_event_count",
        "avg_raw_positive_points",
        "avg_final_positive_points",
    ]].copy()
    out.columns = [
        "Experiment",
        "Avg_Epsilon_Thresh",
        "Avg_POT_Thresh",
        "Avg_Raw_Event_Count",
        "Avg_Final_Event_Count",
        "Avg_Event_Count_Delta",
        "Avg_Raw_Positive_Points",
        "Avg_Final_Positive_Points",
    ]
    for col in out.columns[1:]:
        out[col] = out[col].map(format_float)
    out = out.sort_values("Experiment")
    out.to_csv(OUTPUT_DIR / "table_bms_ablation_summary.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(out, OUTPUT_DIR / "table_bms_ablation_summary.md")
    return out


def build_external_table(results_df):
    ext = results_df[results_df["kind"] == "external"].copy()
    ext["Dataset"] = ext["leaf"].map(infer_external_dataset)
    ext["Experiment"] = ext["leaf"]
    out = ext[["Experiment", "Dataset", "baseline", "metric_f1", "metric_precision", "metric_recall", "metric_auroc"]].copy()
    out.columns = ["Experiment", "Dataset", "Baseline", "F1", "Precision", "Recall", "AUROC"]
    for col in ["F1", "Precision", "Recall", "AUROC"]:
        out[col] = out[col].map(format_float)
    out = out.sort_values(["Baseline", "Dataset", "Experiment"])
    out.to_csv(OUTPUT_DIR / "table_external_results.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(out, OUTPUT_DIR / "table_external_results.md")
    return out


def infer_external_dataset(name):
    if "smap" in name:
        return "SMAP"
    if "msl" in name:
        return "MSL"
    if "bms" in name:
        return "BMS"
    if "nasa_random" in name:
        return "NASA_RANDOM"
    return "UNKNOWN"


def plot_internal_public_results(table_df):
    df = pd.read_csv(OUTPUT_DIR / "table_internal_public_results.csv")
    labels = df["Experiment"].tolist()
    bf = df["BF_F1"].fillna(0).astype(float).tolist()
    event = df["Event_F1"].fillna(0).astype(float).tolist()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = range(len(labels))
    width = 0.38
    ax.bar([i - width / 2 for i in x], bf, width=width, label="BF-F1")
    ax.bar([i + width / 2 for i in x], event, width=width, label="Event-F1")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Internal Results on Public Datasets")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_internal_public_results.png", dpi=160)
    plt.close(fig)


def plot_bms_ablation(table_df):
    df = pd.read_csv(OUTPUT_DIR / "table_bms_ablation_summary.csv")
    labels = df["Experiment"].tolist()
    eps = df["Avg_Epsilon_Thresh"].astype(float).tolist()
    final_events = df["Avg_Final_Event_Count"].astype(float).tolist()
    x = list(range(len(labels)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(x, eps, color="#4C72B0")
    axes[0].set_ylabel("Threshold")
    axes[0].set_title("BMS Ablation Summary")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, final_events, color="#55A868")
    axes[1].set_ylabel("Final Event Count")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_bms_ablation_summary.png", dpi=160)
    plt.close(fig)


def pick_window(df, label_col="A_True_Global", score_col="A_Score_Global", radius=1500):
    if label_col in df.columns and df[label_col].sum() > 0:
        anomaly_idx = int(df.index[df[label_col].astype(int) == 1][0])
        start = max(0, anomaly_idx - radius)
        end = min(len(df), anomaly_idx + radius)
        return df.iloc[start:end].copy()
    peak_idx = int(df[score_col].astype(float).idxmax())
    start = max(0, peak_idx - radius)
    end = min(len(df), peak_idx + radius)
    return df.iloc[start:end].copy()


def plot_representative_cases():
    smap_df = pd.read_pickle(ROOT / "kaggle离线output" / "5.12内部实验" / "main_smap_mtadgat" / "test_output.pkl")
    bms_df = pd.read_pickle(ROOT / "kaggle离线output" / "5.12内部实验" / "main_bms_c3" / "BMS_B14_3_2_cluster1" / "test_output.pkl")
    gdn_df = pd.read_pickle(ROOT / "kaggle离线output" / "5.13外部实验" / "ch3_external_baselines_20260517_162719" / "output" / "gdn_smap_ch3" / "test_output.pkl")

    smap_case = pick_window(smap_df, radius=2500)
    bms_case = pick_window(bms_df, radius=1800)
    gdn_case = pick_window(gdn_df, radius=2500)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    _plot_case(
        axes[0],
        smap_case,
        title="Representative Case: Internal Model on SMAP",
        show_truth=True,
    )
    _plot_case(
        axes[1],
        bms_case,
        title="Representative Case: Internal Model on BMS Cluster 1",
        show_truth=False,
    )
    _plot_case(
        axes[2],
        gdn_case,
        title="Representative Case: GDN on SMAP",
        show_truth=True,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_representative_cases.png", dpi=160)
    plt.close(fig)


def _plot_case(ax, df, title, show_truth):
    x = range(len(df))
    score = df["A_Score_Global"].astype(float).to_numpy()
    ax.plot(x, score, label="Anomaly score", linewidth=1.0)

    if "Thresh_Global" in df.columns:
        thr = float(pd.Series(df["Thresh_Global"]).dropna().iloc[0])
        ax.axhline(thr, color="black", linestyle="--", linewidth=1.0, label="Threshold")
    if "Thresh_Global_Low" in df.columns:
        low = pd.Series(df["Thresh_Global_Low"]).dropna()
        if len(low) > 0:
            ax.axhline(float(low.iloc[0]), color="gray", linestyle=":", linewidth=0.9, label="Low threshold")

    if show_truth and "A_True_Global" in df.columns:
        truth = df["A_True_Global"].astype(int).to_numpy()
        in_event = False
        start = 0
        for idx, flag in enumerate(truth):
            if flag and not in_event:
                start = idx
                in_event = True
            elif not flag and in_event:
                ax.axvspan(start, idx, color="red", alpha=0.12)
                in_event = False
        if in_event:
            ax.axvspan(start, len(truth) - 1, color="red", alpha=0.12)

    if "A_Pred_Global_Event" in df.columns:
        pred = df["A_Pred_Global_Event"].astype(int).to_numpy()
        pred_idx = [i for i, value in enumerate(pred) if value == 1]
        if pred_idx:
            ax.scatter(pred_idx, score[pred_idx], s=5, alpha=0.4, label="Predicted event")

    ax.set_title(title)
    ax.set_ylabel("Global score")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")


def write_summary_note():
    note = [
        "Generated artifacts:",
        "- table_internal_public_results.csv/.md",
        "- table_bms_ablation_summary.csv/.md",
        "- table_external_results.csv/.md",
        "- fig_internal_public_results.png",
        "- fig_bms_ablation_summary.png",
        "- fig_representative_cases.png",
    ]
    (OUTPUT_DIR / "README.txt").write_text("\n".join(note), encoding="utf-8")


def main():
    ensure_output_dir()
    results_df, event_df = load_inputs()
    internal_public = build_internal_public_table(results_df)
    bms_table = build_bms_table(results_df, event_df)
    external_table = build_external_table(results_df)
    plot_internal_public_results(internal_public)
    plot_bms_ablation(bms_table)
    plot_representative_cases()
    write_summary_note()
    print(f"Artifacts written to: {OUTPUT_DIR}")
    print(f"Internal public rows: {len(internal_public)}")
    print(f"BMS ablation rows: {len(bms_table)}")
    print(f"External rows: {len(external_table)}")


if __name__ == "__main__":
    main()
