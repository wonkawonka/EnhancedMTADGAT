"""Complete read-only diagnostics for the frozen residual router probe.

Thresholds use train-normal scores only. Best-head rows use test labels and are
explicitly diagnostic upper bounds, not deployable model results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


HEADS = ("slow", "fast", "relation")
SCORES = HEADS + ("avg", "routed")
GROUPS = tuple(f"{cycle}/{dynamic}" for cycle in ("charge", "discharge")
               for dynamic in ("low", "medium", "high"))
SCORE_COLUMN = {head: head + "_z" for head in HEADS} | {"avg": "avg", "routed": "routed"}


def distribution_plot(test: pd.DataFrame, target: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    colors = ("#2878b5", "#d95f02")
    for ax, group in zip(axes.flat, GROUPS):
        subset = test.loc[test.group == group]
        arrays = []
        positions = []
        for index, head in enumerate(HEADS):
            for label in (0, 1):
                values = subset.loc[subset.label == label, SCORE_COLUMN[head]].to_numpy()
                arrays.append(values)
                positions.append(index * 3 + label)
        finite = np.concatenate(arrays)
        lower, upper = np.quantile(finite, [0.01, 0.99])
        if upper <= lower:
            upper = lower + 1e-6
        clipped = [np.clip(values, lower, upper) for values in arrays]
        boxes = ax.boxplot(clipped, positions=positions, widths=0.7, patch_artist=True,
                           showfliers=False, medianprops={"color": "black"})
        for index, patch in enumerate(boxes["boxes"]):
            patch.set_facecolor(colors[index % 2])
            patch.set_alpha(0.75)
        ax.set_xticks([0.5, 3.5, 6.5], HEADS)
        ax.set_title(group.replace("/", " / "))
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel("Normal-calibrated score (1%-99% clipped)")
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=colors[0], label="Normal"),
                        Patch(color=colors[1], label="Fault")],
               loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(target, dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize(method: str, root: Path) -> None:
    folder = root / method
    train = pd.read_csv(folder / "train_sample_scores.csv")
    test = pd.read_csv(folder / "test_sample_scores.csv")
    if train.label.nunique() != 1 or int(train.label.iloc[0]) != 0:
        raise ValueError("Threshold calibration must use train-normal samples only")
    if set(test.group) != set(GROUPS):
        raise ValueError("Unexpected test condition groups")

    thresholds = {name: float(np.quantile(train[SCORE_COLUMN[name]], 0.95))
                  for name in SCORES}
    rows = []
    for group in ("all",) + GROUPS:
        subset = test if group == "all" else test.loc[test.group == group]
        labels = subset.label.to_numpy(dtype=int)
        normal = labels == 0
        fault = labels == 1
        for name in SCORES:
            values = subset[SCORE_COLUMN[name]].to_numpy(dtype=float)
            predicted = values > thresholds[name]
            rows.append({
                "group": group, "score": name, "normal": int(normal.sum()),
                "fault": int(fault.sum()), "auroc": float(roc_auc_score(labels, values)),
                "ap": float(average_precision_score(labels, values)),
                "fpr_at_train_normal_p95": float(predicted[normal].mean()),
                "recall_at_train_normal_p95": float(predicted[fault].mean()),
                "train_normal_p95_threshold": thresholds[name],
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(folder / "complete_diagnostics.csv", index=False)

    oracle_rows = []
    for group in GROUPS:
        subset = metrics.loc[(metrics.group == group) & metrics.score.isin(HEADS)]
        best = subset.loc[subset.auroc.idxmax()]
        relation = subset.loc[subset.score == "relation"].iloc[0]
        oracle_rows.append({
            "group": group, "normal": int(best.normal), "fault": int(best.fault),
            "best_head_selected_with_test_labels": best.score,
            "best_head_auroc": float(best.auroc),
            "relation_auroc": float(relation.auroc),
            "best_minus_relation_auroc": float(best.auroc - relation.auroc),
            "best_head_ap": float(best.ap), "relation_ap": float(relation.ap),
        })
    oracle = pd.DataFrame(oracle_rows)
    oracle.to_csv(folder / "best_head_oracle_diagnostic.csv", index=False)
    weights = oracle.normal + oracle.fault
    summary = {
        "method": method,
        "threshold_policy": "global train-normal sample-score P95 per score; no test labels",
        "best_head_policy": "test-label post hoc per group; diagnostic only",
        "macro_group_auroc_best_head": float(oracle.best_head_auroc.mean()),
        "macro_group_auroc_relation": float(oracle.relation_auroc.mean()),
        "macro_group_auroc_gap": float(oracle.best_minus_relation_auroc.mean()),
        "sample_weighted_group_auroc_gap": float(np.average(oracle.best_minus_relation_auroc, weights=weights)),
        "warning": "Groupwise AUROCs are not a pooled test AUROC; do not report as deployable performance.",
    }
    (folder / "diagnostic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    distribution_plot(test, folder / "condition_score_distributions.png", method.upper())
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/internal/residual_router_probe_lfp_seed3407"))
    args = parser.parse_args()
    for method in ("mtad_gat", "tranad"):
        summarize(method, args.root)


if __name__ == "__main__":
    main()
