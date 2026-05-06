import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_VARIANTS = {
    "raw": "A_Pred_Global_Raw",
    "event": "A_Pred_Global_Event",
    "final": "A_Pred_Global",
}


def iter_segment_slices(df):
    if "Segment_ID" not in df.columns or len(df) == 0:
        yield slice(0, len(df))
        return

    segment_ids = df["Segment_ID"].to_numpy()
    boundaries = np.where(segment_ids[1:] != segment_ids[:-1])[0] + 1
    start = 0
    for boundary in boundaries:
        yield slice(start, boundary)
        start = boundary
    yield slice(start, len(df))


def extract_event_segments(binary_pred, segment_ids=None):
    binary_pred = np.asarray(binary_pred).astype(np.int32)
    if len(binary_pred) == 0:
        return []

    if segment_ids is None:
        segment_ids = np.zeros(len(binary_pred), dtype=np.int32)
    else:
        segment_ids = np.asarray(segment_ids)

    segments = []
    start = None
    current_segment_id = None
    for idx, flag in enumerate(binary_pred):
        row_segment_id = int(segment_ids[idx])

        if start is not None and row_segment_id != current_segment_id:
            segments.append((start, idx - 1, current_segment_id))
            start = None
            current_segment_id = None

        if flag and start is None:
            start = idx
            current_segment_id = row_segment_id
        elif not flag and start is not None:
            segments.append((start, idx - 1, current_segment_id))
            start = None
            current_segment_id = None

    if start is not None:
        segments.append((start, len(binary_pred) - 1, current_segment_id))

    return segments


def summarize_segments(segments, total_length, short_event_threshold):
    lengths = np.asarray([end - start + 1 for start, end, _ in segments], dtype=np.int32)
    summary = {
        "event_count": int(len(segments)),
        "positive_points": int(lengths.sum()) if len(lengths) > 0 else 0,
        "coverage_ratio": float(lengths.sum() / total_length) if total_length > 0 else 0.0,
        "short_event_threshold": int(short_event_threshold),
        "short_event_count": int(np.sum(lengths < short_event_threshold)) if len(lengths) > 0 else 0,
        "singleton_event_count": int(np.sum(lengths == 1)) if len(lengths) > 0 else 0,
        "mean_event_length": float(np.mean(lengths)) if len(lengths) > 0 else 0.0,
        "median_event_length": float(np.median(lengths)) if len(lengths) > 0 else 0.0,
        "max_event_length": int(np.max(lengths)) if len(lengths) > 0 else 0,
        "min_event_length": int(np.min(lengths)) if len(lengths) > 0 else 0,
    }
    summary["short_event_ratio"] = (
        float(summary["short_event_count"] / summary["event_count"])
        if summary["event_count"] > 0
        else 0.0
    )
    return summary


def summarize_overlap(pred_segments, true_segments):
    if true_segments is None:
        return {}

    overlapped_pred = 0
    for pred_start, pred_end, pred_seg_id in pred_segments:
        has_overlap = any(
            pred_seg_id == true_seg_id and pred_start <= true_end and pred_end >= true_start
            for true_start, true_end, true_seg_id in true_segments
        )
        if has_overlap:
            overlapped_pred += 1

    detected_true = 0
    for true_start, true_end, true_seg_id in true_segments:
        has_detection = any(
            pred_seg_id == true_seg_id and pred_start <= true_end and pred_end >= true_start
            for pred_start, pred_end, pred_seg_id in pred_segments
        )
        if has_detection:
            detected_true += 1

    pred_count = len(pred_segments)
    true_count = len(true_segments)
    return {
        "overlap_pred_event_count": int(overlapped_pred),
        "non_overlap_pred_event_count": int(pred_count - overlapped_pred),
        "pred_event_overlap_ratio": float(overlapped_pred / pred_count) if pred_count > 0 else 0.0,
        "true_event_count": int(true_count),
        "detected_true_event_count": int(detected_true),
        "true_event_detection_ratio": float(detected_true / true_count) if true_count > 0 else 0.0,
    }


def flatten_metrics(prefix, payload, row):
    for key, value in payload.items():
        next_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flatten_metrics(next_key, value, row)
        else:
            row[next_key] = value


def build_case_summary(exp_dir, short_event_threshold):
    test_output_path = exp_dir / "test_output.pkl"
    if not test_output_path.exists():
        raise FileNotFoundError(f"Missing file: {test_output_path}")

    df = pd.read_pickle(test_output_path)
    segment_ids = df["Segment_ID"].to_numpy() if "Segment_ID" in df.columns else None
    total_length = len(df)

    row = {
        "experiment_dir": str(exp_dir),
        "test_output_path": str(test_output_path),
        "num_points": int(total_length),
        "short_event_threshold": int(short_event_threshold),
    }

    true_segments = None
    if "A_True_Global" in df.columns:
        true_segments = extract_event_segments(df["A_True_Global"].to_numpy(dtype=np.int32), segment_ids)
        flatten_metrics(
            "true",
            summarize_segments(true_segments, total_length, short_event_threshold),
            row,
        )

    for variant_name, column_name in PREDICTION_VARIANTS.items():
        if column_name not in df.columns:
            continue
        pred_segments = extract_event_segments(df[column_name].to_numpy(dtype=np.int32), segment_ids)
        flatten_metrics(
            variant_name,
            summarize_segments(pred_segments, total_length, short_event_threshold),
            row,
        )
        if true_segments is not None:
            flatten_metrics(
                f"{variant_name}_overlap",
                summarize_overlap(pred_segments, true_segments),
                row,
            )

    if "raw_event_count" in row and "event_event_count" in row:
        row["delta_event_count_event_minus_raw"] = int(row["event_event_count"] - row["raw_event_count"])
        row["delta_short_event_count_event_minus_raw"] = int(
            row.get("event_short_event_count", 0) - row.get("raw_short_event_count", 0)
        )
        row["delta_coverage_ratio_event_minus_raw"] = float(
            row.get("event_coverage_ratio", 0.0) - row.get("raw_coverage_ratio", 0.0)
        )
        row["delta_mean_event_length_event_minus_raw"] = float(
            row.get("event_mean_event_length", 0.0) - row.get("raw_mean_event_length", 0.0)
        )
        if "raw_overlap_non_overlap_pred_event_count" in row and "event_overlap_non_overlap_pred_event_count" in row:
            row["delta_non_overlap_pred_event_count_event_minus_raw"] = int(
                row["event_overlap_non_overlap_pred_event_count"] - row["raw_overlap_non_overlap_pred_event_count"]
            )

    summary_path = exp_dir / "summary_metrics.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary_metrics = json.load(f)
        event_result = summary_metrics.get("event_consistency_result", {})
        flatten_metrics("summary_event", event_result, row)

    thresholds_path = exp_dir / "thresholds.json"
    if thresholds_path.exists():
        with thresholds_path.open("r", encoding="utf-8") as f:
            thresholds = json.load(f)
        for key in ("global_threshold", "event_high_threshold", "event_low_threshold"):
            if key in thresholds:
                row[key] = thresholds[key]

    return row


def find_experiment_dirs(root_dir):
    return sorted({path.parent for path in root_dir.rglob("test_output.pkl")})


def write_csv(rows, csv_path):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize anomaly event segments from test_output.pkl files.")
    parser.add_argument(
        "--root",
        type=str,
        default="output",
        help="Root directory to scan recursively for test_output.pkl files.",
    )
    parser.add_argument(
        "--short-event-threshold",
        type=int,
        default=3,
        help="Events shorter than this value are counted as short events.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional output prefix. Defaults to <root>/event_segment_stats",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root).resolve()
    exp_dirs = find_experiment_dirs(root_dir)
    if not exp_dirs:
        raise FileNotFoundError(f"No test_output.pkl found under: {root_dir}")

    rows = []
    for exp_dir in exp_dirs:
        try:
            rows.append(build_case_summary(exp_dir, args.short_event_threshold))
        except Exception as exc:
            rows.append({
                "experiment_dir": str(exp_dir),
                "status": f"failed: {exc}",
                "short_event_threshold": int(args.short_event_threshold),
            })

    if args.output_prefix:
        output_prefix = Path(args.output_prefix).resolve()
    else:
        output_prefix = root_dir / "event_segment_stats"

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")

    write_csv(rows, csv_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Scanned {len(exp_dirs)} experiment directories")
    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")


if __name__ == "__main__":
    main()
