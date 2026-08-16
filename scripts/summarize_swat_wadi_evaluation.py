#!/usr/bin/env python3
"""Summarize completed SWaT/WADI continuation evaluations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    PROJECT_ROOT / "runs/kaggle_downloads/swat_wadi_v8/evaluation"
)


def recall_at_fpr(labels, scores, limit):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eligible = np.flatnonzero(fpr <= limit)
    if not len(eligible):
        return 0.0, float("inf"), 0.0
    index = eligible[np.argmax(tpr[eligible])]
    return float(tpr[index]), float(thresholds[index]), float(fpr[index])


def main():
    rows = []
    manual_root = EVALUATION_ROOT / "runs/manual"
    for summary_path in sorted(manual_root.rglob("summary_metrics.json")):
        model_dir = summary_path.parent
        dataset = model_dir.parent.name
        experiment = model_dir.name.replace(f"_dataset{dataset}", "")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        raw = summary["raw_point_result"]
        epsilon = summary["epsilon_result"]
        pot = summary["pot_result"]
        best_f1 = summary["bf_result"]
        event = summary["event_consistency_result"]["raw_event_result"]
        score_frame = pd.read_pickle(model_dir / "test_output.pkl")
        labels = score_frame["A_True_Global"].to_numpy(dtype=np.int64)
        scores = score_frame["A_Score_Global"].to_numpy(dtype=np.float64)
        row = {
            "dataset": dataset,
            "experiment": experiment,
            "auroc": raw["auroc"],
            "average_precision": raw["average_precision"],
            "raw_point_f1": raw["point_f1"],
            "raw_point_precision": raw["point_precision"],
            "raw_point_recall": raw["point_recall"],
            "epsilon_f1_point_adjusted": epsilon["f1"],
            "pot_f1_point_adjusted": pot["f1"],
            "best_f1_oracle_point_adjusted": best_f1["f1"],
            "raw_event_f1": event["event_f1"],
            "raw_event_recall": event["event_recall"],
        }
        for limit, label in ((0.01, "1pct"), (0.005, "0_5pct"), (0.001, "0_1pct")):
            recall, threshold, actual_fpr = recall_at_fpr(labels, scores, limit)
            row[f"recall_at_fpr_{label}"] = recall
            row[f"actual_fpr_{label}"] = actual_fpr
            row[f"threshold_at_fpr_{label}"] = threshold
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(["dataset", "experiment"])
    csv_path = EVALUATION_ROOT / "swat_wadi_metrics.csv"
    json_path = EVALUATION_ROOT / "swat_wadi_metrics.json"
    result.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(result.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
