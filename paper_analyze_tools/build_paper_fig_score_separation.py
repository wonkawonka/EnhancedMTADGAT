"""直接画原始分数时间序列片段，标注真异常区域，直观对比 Baseline 和 C3 的分离能力。

输出: analysis/paper_figures/score_separation.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(
    r"D:\作业\论文复现\基线模型\mtad-gat-pytorch\kaggle离线output\ch3_main_results"
)

CONFIG = {
    "MSL": {
        "baseline_dir": "msl_mtadgat_baseline",
        "c3_dir": "msl_c3_full",
        "seg_start": 5000,
        "seg_len": 3000,
        "title": "MSL（好：C3 能明显分离正常和异常）",
    },
    "SMAP": {
        "baseline_dir": "smap_mtadgat_baseline",
        "c3_dir": "smap_c3_full",
        "seg_start": 10000,
        "seg_len": 3000,
        "title": "SMAP（差：C3 把异常分数也过度压低了）",
    },
}

SAMPLE_SIZE = 50000
OUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(exp_dir):
    p = BASE / exp_dir / "test_output.pkl"
    df = pd.read_pickle(p)
    return df["A_Score_Global"].to_numpy(dtype=np.float64), df["A_True_Global"].to_numpy(dtype=bool)


fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=False)

for col_idx, (ds, cfg) in enumerate(CONFIG.items()):
    print(f"\n=== {ds} ===")
    baseline_scores, baseline_labels = load_data(cfg["baseline_dir"])
    c3_scores, c3_labels = load_data(cfg["c3_dir"])

    start = cfg["seg_start"]
    end = start + cfg["seg_len"]

    x = np.arange(end - start, dtype=int)

    # 上一行：Baseline
    ax = axes[0][col_idx]
    # 用灰色标异常区域
    ax.fill_between(x, 0, 1, where=baseline_labels[start:end],
                    color="#ffcccc", alpha=0.5, transform=ax.get_xaxis_transform(),
                    label="真异常")
    ax.plot(x, baseline_scores[start:end], color="#666666", linewidth=0.8, label="Baseline 分数")
    # 阈值线
    if ds == "MSL":
        ax.axhline(0.088, color="#888888", linestyle="--", linewidth=1.0, label="阈值=0.088")
        ax.set_ylim(-0.01, 0.5)
    else:
        ax.axhline(0.1543, color="#888888", linestyle="--", linewidth=1.0, label="阈值=0.154")
        ax.set_ylim(-0.01, 0.4)
    ax.set_ylabel("异常分数")
    ax.set_title(f"[{ds}] Baseline (MTAD-GAT)")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # 下一行：C3
    ax = axes[1][col_idx]
    ax.fill_between(x, 0, 1, where=c3_labels[start:end],
                    color="#ffcccc", alpha=0.5, transform=ax.get_xaxis_transform(),
                    label="真异常")
    ax.plot(x, c3_scores[start:end], color="#4a8fc1", linewidth=0.8, label="C3 分数")
    if ds == "MSL":
        ax.axhline(0.0344, color="#2b5f8a", linestyle="--", linewidth=1.0, label="阈值=0.0344")
        ax.set_ylim(-0.01, 0.5)
    else:
        ax.axhline(0.0696, color="#2b5f8a", linestyle="--", linewidth=1.0, label="阈值=0.0696")
        ax.set_ylim(-0.01, 0.4)
    ax.set_xlabel("时间步")
    ax.set_ylabel("异常分数")
    ax.set_title(f"[{ds}] C3")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # 计算异常区域的平均分离度
    base_anomaly_avg = np.mean(baseline_scores[baseline_labels])
    base_normal_avg = np.mean(baseline_scores[~baseline_labels])
    c3_anomaly_avg = np.mean(c3_scores[c3_labels])
    c3_normal_avg = np.mean(c3_scores[~c3_labels])
    print(f"  Baseline: 正常平均={base_normal_avg:.4f}, 异常平均={base_anomaly_avg:.4f}")
    print(f"  C3: 正常平均={c3_normal_avg:.4f}, 异常平均={c3_anomaly_avg:.4f}")

fig.suptitle("异常分数时间序列片段对比（灰色底色=真异常区域）", fontsize=15, y=1.02)
fig.tight_layout()
out_path = OUT_DIR / "score_separation.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\n[DONE] saved to {out_path}")
