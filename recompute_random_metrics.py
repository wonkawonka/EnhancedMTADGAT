import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["A_Score_Global", "Pred_Error_Global", "Recon_Error_Global"]


def safe_fpr(result):
    if not isinstance(result, dict):
        return None

    fp = result.get("FP")
    tn = result.get("TN")
    if fp is None or tn is None:
        return None

    denom = fp + tn
    if denom == 0:
        return None
    return float(fp / denom)


def get_stage_means(scores):
    if len(scores) == 0:
        return {"early": None, "middle": None, "late": None}

    boundaries = np.linspace(0, len(scores), 4, dtype=int)
    stage_names = ["early", "middle", "late"]
    stage_means = {}
    for idx, stage_name in enumerate(stage_names):
        start, end = boundaries[idx], boundaries[idx + 1]
        if end <= start:
            stage_means[stage_name] = None
        else:
            stage_means[stage_name] = float(np.mean(scores[start:end]))
    return stage_means


def get_score_rising_stage(scores):
    stage_means = get_stage_means(scores)
    valid_stage_means = {k: v for k, v in stage_means.items() if v is not None}
    if not valid_stage_means:
        return "unknown"
    return max(valid_stage_means, key=valid_stage_means.get)


def parse_experiment_metadata(exp_dir):
    name = exp_dir.name
    battery_match = re.search(r"(?:fold_)?(rw\d+)", name, flags=re.IGNORECASE)
    battery_id = battery_match.group(1).upper() if battery_match else None
    feattrans = "feattrans-on" in name
    multi_scale = "_basic_" in name or "_ms_" in name

    parent_name = exp_dir.parent.name
    if parent_name == "开了transformer":
        transformer_group = "on"
    elif parent_name == "没开transformer":
        transformer_group = "off"
    else:
        transformer_group = "unknown"

    return {
        "experiment_name": name,
        "battery_id": battery_id,
        "transformer_group": transformer_group,
        "feattrans_enabled": feattrans,
        "multi_scale_enabled": multi_scale,
    }


def load_optional_json(file_path):
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_summary(exp_dir):
    test_output_path = exp_dir / "test_output.pkl"
    if not test_output_path.exists():
        raise FileNotFoundError(f"Missing file: {test_output_path}")

    df = pd.read_pickle(test_output_path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {test_output_path}: {missing_columns}")

    thresholds = load_optional_json(exp_dir / "thresholds.json")
    summary_metrics = load_optional_json(exp_dir / "summary_metrics.json")

    anomaly_scores = df["A_Score_Global"].to_numpy(dtype=np.float32)
    pred_error = df["Pred_Error_Global"].to_numpy(dtype=np.float32)
    recon_error = df["Recon_Error_Global"].to_numpy(dtype=np.float32)
    stage_means = get_stage_means(anomaly_scores)
    epsilon_result = summary_metrics.get("epsilon_result")
    pot_result = summary_metrics.get("pot_result")
    bf_result = summary_metrics.get("bf_result")

    summary = {
        **parse_experiment_metadata(exp_dir),
        "experiment_dir": str(exp_dir),
        "test_output_path": str(test_output_path),
        "num_points": int(len(df)),
        "global_threshold": float(thresholds["global_threshold"]) if "global_threshold" in thresholds else None,
        "max_score": float(np.max(anomaly_scores)) if len(anomaly_scores) > 0 else None,
        "mean_score": float(np.mean(anomaly_scores)) if len(anomaly_scores) > 0 else None,
        "score_std": float(np.std(anomaly_scores)) if len(anomaly_scores) > 0 else None,
        "early_mean_score": stage_means["early"],
        "middle_mean_score": stage_means["middle"],
        "late_mean_score": stage_means["late"],
        "score_rising_stage": get_score_rising_stage(anomaly_scores),
        "pred_error_mean": float(np.mean(pred_error)) if len(pred_error) > 0 else None,
        "recon_error_mean": float(np.mean(recon_error)) if len(recon_error) > 0 else None,
        "topk_score_indices": [int(idx) for idx in np.argsort(anomaly_scores)[-5:][::-1].tolist()] if len(anomaly_scores) > 0 else [],
        "epsilon_fpr": safe_fpr(epsilon_result),
        "pot_fpr": safe_fpr(pot_result),
        "bf_fpr": safe_fpr(bf_result),
        "epsilon_result": epsilon_result,
        "pot_result": pot_result,
        "bf_result": bf_result,
    }
    return summary


def find_random_experiment_dirs(root_dir):
    return sorted(
        [
            path
            for path in root_dir.glob("**/ch3_rw_discharge*")
            if path.is_dir() and (path / "test_output.pkl").exists()
        ]
    )


def write_json(file_path, data):
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    if sys.version_info < (3, 8):
        raise RuntimeError("This script requires Python 3.8+ to read pickle protocol 5 files.")

    parser = argparse.ArgumentParser(description="Recompute summary metrics for NASA random discharge experiments.")
    parser.add_argument(
        "--root",
        default="kaggle离线output",
        help="Root directory that contains random discharge experiment folders.",
    )
    parser.add_argument(
        "--summary-name",
        default="random_case_summary.json",
        help="Per-experiment summary file name to write.",
    )
    parser.add_argument(
        "--skip-write-per-case",
        action="store_true",
        help="Only write the aggregated files and skip per-experiment summary json.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    exp_dirs = find_random_experiment_dirs(root_dir)
    if not exp_dirs:
        raise FileNotFoundError(f"No random discharge experiment with test_output.pkl found under: {root_dir}")

    summaries = []
    failures = []
    for exp_dir in exp_dirs:
        try:
            summary = build_summary(exp_dir)
            summaries.append(summary)
            if not args.skip_write_per_case:
                write_json(exp_dir / args.summary_name, summary)
        except Exception as exc:
            failures.append({"experiment_dir": str(exp_dir), "error": str(exc)})

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["transformer_group", "battery_id", "multi_scale_enabled", "experiment_name"],
            na_position="last",
        )
        summary_df.to_csv(root_dir / "random_case_comparison.csv", index=False, encoding="utf-8-sig")
        write_json(root_dir / "random_case_comparison.json", summaries)

    if failures:
        write_json(root_dir / "random_case_failures.json", failures)
        print(f"Completed with {len(failures)} failures. See random_case_failures.json for details.")
    else:
        print(f"Completed successfully for {len(summaries)} experiments.")


if __name__ == "__main__":
    main()
