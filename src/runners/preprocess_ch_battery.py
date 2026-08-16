"""一次性缓存 CH-BatteryGen LFP 放电样本，避免多模型重复解析原始 CSV。"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from src.data.ch_battery_utils import (
    CH_BATTERY_CORE_FEATURE_COLUMNS,
    CH_BATTERY_DATASET_NAME,
    _load_ch_battery_sequence,
    _split_normal_vins,
    build_ch_battery_manifest,
    load_ch_battery_fault_details,
)
from sklearn.preprocessing import MinMaxScaler


def _dump_pickle(value, path: Path):
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="CH-BatteryGen V1.0 directory containing LFP/")
    parser.add_argument("--output-dir", default="", help="Defaults to <root>/processed/lfp_discharge")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "processed" / "lfp_discharge"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_ch_battery_manifest(root, chemistry="LFP", cycle_kind="discharge")
    normal_manifest = manifest[manifest["fault_type"] == "normal"].copy()
    fault_manifest = manifest[manifest["fault_type"] != "normal"].copy()
    train_vins, test_normal_vins = _split_normal_vins(
        normal_manifest["vin"].unique(), train_ratio=args.train_ratio, seed=args.seed
    )
    train_manifest = normal_manifest[normal_manifest["vin"].isin(train_vins)].copy()
    test_manifest = (
        pd.concat(
            [normal_manifest[normal_manifest["vin"].isin(test_normal_vins)], fault_manifest],
            ignore_index=True,
        )
        .sort_values(["sample_label", "fault_type", "vin", "cycle_index", "sample_id"])
        .reset_index(drop=True)
    )
    fault_details = load_ch_battery_fault_details(root, chemistry="LFP")

    def load_map(frame, fit_scaler=None):
        data_map, metadata = {}, {}
        for row in frame.itertuples(index=False):
            # Read only the seven modelled pack-level channels.  This avoids retaining 124 cell-voltage columns.
            sequence = _load_ch_battery_sequence(row.file_path, CH_BATTERY_CORE_FEATURE_COLUMNS)
            if fit_scaler is not None:
                fit_scaler.partial_fit(sequence)
            data_map[row.sample_id] = sequence
            detail = fault_details.get(row.vin, {})
            metadata[row.sample_id] = {
                "sample_id": row.sample_id, "file_path": row.file_path,
                "relative_path": row.relative_path, "chemistry": row.chemistry,
                "fault_type": row.fault_type, "vin": row.vin,
                "cycle_kind": row.cycle_kind, "cycle_index": int(row.cycle_index),
                "sample_label": int(row.sample_label),
                "severity": detail.get("severity"), "fault_cell_id": detail.get("fault_cell_id"),
                "fault_value": detail.get("fault_value"), "fault_unit": detail.get("fault_unit"),
            }
        return data_map, metadata

    # Raw 7-D maps are persisted; the scaler is fitted only on normal training VINs and applied by the data loader.
    scaler = MinMaxScaler()
    train_map, train_metadata = load_map(train_manifest, fit_scaler=scaler)
    test_map, test_metadata = load_map(test_manifest)
    test_label = test_manifest["sample_label"].to_numpy(dtype="int32")

    prefix = CH_BATTERY_DATASET_NAME
    _dump_pickle(train_map, output_dir / f"{prefix}_train.pkl")
    _dump_pickle(test_map, output_dir / f"{prefix}_test.pkl")
    _dump_pickle(test_label, output_dir / f"{prefix}_test_label.pkl")
    _dump_pickle(train_metadata, output_dir / f"{prefix}_train_meta.pkl")
    _dump_pickle(test_metadata, output_dir / f"{prefix}_test_meta.pkl")
    _dump_pickle(scaler, output_dir / f"{prefix}_scaler.pkl")
    (output_dir / "feature_columns.json").write_text(
        json.dumps(CH_BATTERY_CORE_FEATURE_COLUMNS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "dataset": prefix,
        "root": str(root),
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
        "feature_columns": CH_BATTERY_CORE_FEATURE_COLUMNS,
        "train_samples": len(train_map),
        "test_samples": len(test_map),
        "test_fault_samples": int(test_label.sum()),
        "train_normal_vins": len(train_vins),
        "test_normal_vins": len(test_normal_vins),
        "source": "raw_csv_cached",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Cache directory: {output_dir}")


if __name__ == "__main__":
    main()
