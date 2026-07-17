"""提供数据探索分析的统一入口，输出统计表、图表和 Markdown 报告。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.dataset_analysis import DEFAULT_RANDOM_STATE, run_dataset_analysis
from src.data.ch_battery_utils import CH_BATTERY_DATASET_NAME
from src.project_paths import REPORT_ROOT, resolve_dataset_root


def parse_args():
    parser = argparse.ArgumentParser(description="Run dataset exploration and feature analysis.")
    parser.add_argument(
        "--dataset",
        type=str.upper,
        required=True,
        choices=[
            "MSL",
            "NASA_RANDOM_CHARGE",
            "NASA_RANDOM_DISCHARGE",
            "BMS",
            CH_BATTERY_DATASET_NAME,
            "CALCE",
            "CALCE2",
        ],
        help="Dataset name used for analysis.",
    )
    parser.add_argument("--output-dir", type=str, default="", help="Directory used to save analysis outputs.")
    parser.add_argument("--nasa_battery_id", type=str, default="", help="Single NASA/NASA_RANDOM entity.")
    parser.add_argument("--nasa_train_batteries", type=str, default="", help="Comma separated NASA train entities.")
    parser.add_argument("--nasa_test_batteries", type=str, default="", help="Comma separated NASA test entities.")
    parser.add_argument(
        "--ch_battery_root",
        type=str,
        default=str(resolve_dataset_root("CH-BATTERY", "CH-BATTERY")),
        help="CH-BATTERY root directory.",
    )
    parser.add_argument("--ch_battery_preprocessed_dir", type=str, default="", help="Optional CH-BATTERY processed directory.")
    parser.add_argument("--ch_battery_train_ratio", type=float, default=0.8, help="CH-BATTERY normal VIN train ratio.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed used in analysis sampling.")
    return parser.parse_args()


def resolve_output_dir(args) -> Path:
    if args.output_dir.strip():
        return Path(args.output_dir).resolve()
    return (REPORT_ROOT / "analysis" / args.dataset.lower()).resolve()


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args)
    manifest = run_dataset_analysis(
        dataset=args.dataset,
        output_dir=output_dir,
        nasa_battery_id=args.nasa_battery_id,
        nasa_train_batteries=args.nasa_train_batteries,
        nasa_test_batteries=args.nasa_test_batteries,
        ch_battery_root=args.ch_battery_root,
        ch_battery_preprocessed_dir=args.ch_battery_preprocessed_dir,
        ch_battery_train_ratio=args.ch_battery_train_ratio,
        seed=args.seed,
    )

    print("Analysis finished")
    print(f"Dataset    : {args.dataset}")
    print(f"Output dir : {manifest['output_dir']}")
    print(f"Summary    : {manifest['summary_path']}")
    print(f"Report     : {manifest['report_path']}")


if __name__ == "__main__":
    main()
