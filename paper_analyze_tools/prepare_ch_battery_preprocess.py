"""预处理 CH-BATTERY 的 LFP/discharge 数据，生成可直接训练的 pkl 产物与辅助清单。"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ch_battery_utils import (
    _load_ch_battery_sequence,
    _resolve_feature_columns,
    _split_normal_vins,
    build_ch_battery_manifest,
)
from utils import flatten_sequence_collection, normalize_data


PICKLE_PREFIX = "CH_BATTERY_LFP_DISCHARGE"


def count_csv_rows(file_path: str) -> int:
    return int(pd.read_csv(file_path, usecols=[0]).shape[0])


def build_split(manifest_df: pd.DataFrame, train_ratio: float, seed: int):
    normal_df = manifest_df[manifest_df["fault_type"] == "normal"].copy()
    fault_df = manifest_df[manifest_df["fault_type"] != "normal"].copy()
    train_vins, test_normal_vins = _split_normal_vins(normal_df["vin"].unique(), train_ratio=train_ratio, seed=seed)
    train_manifest = normal_df[normal_df["vin"].isin(train_vins)].copy()
    test_manifest = pd.concat(
        [
            normal_df[normal_df["vin"].isin(test_normal_vins)].copy(),
            fault_df.copy(),
        ],
        ignore_index=True,
    ).sort_values(["sample_label", "fault_type", "vin", "cycle_index", "sample_id"]).reset_index(drop=True)
    return train_manifest, test_manifest, train_vins, test_normal_vins


def build_sequence_length_table(train_manifest: pd.DataFrame, test_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, current_df in [("train", train_manifest), ("test", test_manifest)]:
        for row in current_df.itertuples(index=False):
            rows.append(
                {
                    "split": split_name,
                    "sample_id": row.sample_id,
                    "fault_type": row.fault_type,
                    "vin": row.vin,
                    "cycle_index": int(row.cycle_index),
                    "sample_label": int(row.sample_label),
                    "sequence_length": count_csv_rows(row.file_path),
                }
            )
    return pd.DataFrame(rows)


def fit_train_scaler(train_manifest: pd.DataFrame, feature_columns: list[str]):
    train_sequences = [
        _load_ch_battery_sequence(row.file_path, feature_columns)
        for row in train_manifest.itertuples(index=False)
    ]
    train_concat = flatten_sequence_collection(train_sequences, dtype=np.float32)
    _, scaler = normalize_data(train_concat, scaler=None)
    return scaler


def build_split_payload(manifest_df: pd.DataFrame, feature_columns: list[str]):
    data_map = {}
    metadata_map = {}
    labels = []
    sample_ids = []

    for row in manifest_df.itertuples(index=False):
        sequence = _load_ch_battery_sequence(row.file_path, feature_columns)
        data_map[row.sample_id] = sequence
        metadata_map[row.sample_id] = {
            "sample_id": row.sample_id,
            "file_path": row.file_path,
            "relative_path": getattr(row, "relative_path", None),
            "chemistry": row.chemistry,
            "fault_type": row.fault_type,
            "vin": row.vin,
            "cycle_kind": row.cycle_kind,
            "cycle_index": int(row.cycle_index),
            "sample_label": int(row.sample_label),
            "severity": row.severity,
            "fault_cell_id": row.fault_cell_id,
            "fault_value": row.fault_value,
            "fault_unit": row.fault_unit,
        }
        sample_ids.append(row.sample_id)
        labels.append(int(row.sample_label))

    return data_map, metadata_map, np.asarray(labels, dtype=np.int32), sample_ids


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_pickle(path: Path, payload):
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_summary_md(path: Path, summary: dict):
    lines = [
        "# CH-BATTERY LFP Discharge Preprocess Summary",
        "",
        f"- chemistry: {summary['chemistry']}",
        f"- cycle_kind: {summary['cycle_kind']}",
        f"- feature_count: {summary['feature_count']}",
        f"- total_samples: {summary['total_samples']}",
        f"- train_samples: {summary['train_samples']}",
        f"- test_samples: {summary['test_samples']}",
        f"- train_normal_vins: {summary['train_normal_vins']}",
        f"- test_normal_vins: {summary['test_normal_vins']}",
        f"- fault_test_samples: {summary['fault_test_samples']}",
        f"- sequence_length_min: {summary['sequence_length_min']}",
        f"- sequence_length_median: {summary['sequence_length_median']}",
        f"- sequence_length_max: {summary['sequence_length_max']}",
        "",
        "## Fault Counts",
        "",
    ]
    for fault_type, count in summary["fault_counts"].items():
        lines.append(f"- {fault_type}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare CH-BATTERY LFP discharge artifacts.")
    parser.add_argument("--root", type=str, default="datasets/CH-BATTERY", help="CH-BATTERY root directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/CH-BATTERY/preprocessed/lfp_discharge",
        help="Directory used to save preprocess artifacts",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Normal VIN ratio used for training")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed used for VIN split")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = build_ch_battery_manifest(root, chemistry="LFP", cycle_kind="discharge")
    train_manifest, test_manifest, train_vins, test_normal_vins = build_split(
        manifest_df, train_ratio=args.train_ratio, seed=args.seed
    )
    feature_columns = _resolve_feature_columns(train_manifest.iloc[0]["file_path"])
    sequence_length_df = build_sequence_length_table(train_manifest, test_manifest)
    scaler = fit_train_scaler(train_manifest, feature_columns)
    train_data_map, train_meta_map, train_label, train_sample_ids = build_split_payload(train_manifest, feature_columns)
    test_data_map, test_meta_map, test_label, test_sample_ids = build_split_payload(test_manifest, feature_columns)

    manifest_df.to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    train_manifest.to_csv(output_dir / "train_manifest.csv", index=False, encoding="utf-8-sig")
    test_manifest.to_csv(output_dir / "test_manifest.csv", index=False, encoding="utf-8-sig")
    sequence_length_df.to_csv(output_dir / "sequence_lengths.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "feature_columns.json", feature_columns)
    write_json(output_dir / "train_vins.json", train_vins)
    write_json(output_dir / "test_normal_vins.json", test_normal_vins)
    write_json(output_dir / "train_sample_ids.json", train_sample_ids)
    write_json(output_dir / "test_sample_ids.json", test_sample_ids)
    write_pickle(output_dir / f"{PICKLE_PREFIX}_train.pkl", train_data_map)
    write_pickle(output_dir / f"{PICKLE_PREFIX}_test.pkl", test_data_map)
    write_pickle(output_dir / f"{PICKLE_PREFIX}_test_label.pkl", test_label)
    write_pickle(output_dir / f"{PICKLE_PREFIX}_train_meta.pkl", train_meta_map)
    write_pickle(output_dir / f"{PICKLE_PREFIX}_test_meta.pkl", test_meta_map)
    write_pickle(output_dir / f"{PICKLE_PREFIX}_scaler.pkl", scaler)

    fault_counts = manifest_df.groupby("fault_type")["sample_id"].count().sort_index().to_dict()
    length_values = sequence_length_df["sequence_length"].to_numpy(dtype=np.int32)
    summary = {
        "chemistry": "LFP",
        "cycle_kind": "discharge",
        "feature_count": int(len(feature_columns)),
        "feature_columns_path": str((output_dir / "feature_columns.json").resolve()),
        "train_pickle_path": str((output_dir / f"{PICKLE_PREFIX}_train.pkl").resolve()),
        "test_pickle_path": str((output_dir / f"{PICKLE_PREFIX}_test.pkl").resolve()),
        "test_label_pickle_path": str((output_dir / f"{PICKLE_PREFIX}_test_label.pkl").resolve()),
        "total_samples": int(len(manifest_df)),
        "train_samples": int(len(train_manifest)),
        "test_samples": int(len(test_manifest)),
        "train_normal_vins": int(len(train_vins)),
        "test_normal_vins": int(len(test_normal_vins)),
        "fault_test_samples": int(test_manifest["sample_label"].sum()),
        "sequence_length_min": int(length_values.min()),
        "sequence_length_median": float(np.median(length_values)),
        "sequence_length_max": int(length_values.max()),
        "fault_counts": {str(k): int(v) for k, v in fault_counts.items()},
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
        "root": str(root),
    }
    write_json(output_dir / "summary.json", summary)
    write_summary_md(output_dir / "summary.md", summary)

    print(f"[DONE] CH-BATTERY preprocess artifacts written to: {output_dir}")
    print(
        "[SUMMARY]",
        f"total_samples={summary['total_samples']}",
        f"train_samples={summary['train_samples']}",
        f"test_samples={summary['test_samples']}",
        f"feature_count={summary['feature_count']}",
        f"fault_test_samples={summary['fault_test_samples']}",
    )


if __name__ == "__main__":
    main()
