#!/usr/bin/env python3
"""Serialize formal CH-BatteryGen research splits for remote execution.

The output contains only the seven scaled input channels, entity metadata and
the exact train/validation/test VIN memberships.  Fault labels are retained in
test metadata for evaluation only and are never used to fit the scaler.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ch_battery_utils import load_ch_battery_research_split


FORMAT = "ch_batterygen_formal_70_10_20_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/CH-BatteryGen/V1.0")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--split_seeds", nargs="+", type=int, default=[3407],
                        help="VIN partition seeds. Model seeds are applied only at training time.")
    parser.add_argument("--chemistry", default="LFP", choices=("LFP", "NCM"))
    parser.add_argument("--cycle_kind", default="discharge", choices=("charge", "discharge"))
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"format": FORMAT, "chemistry": args.chemistry, "cycle_kind": args.cycle_kind, "split_seeds": []}
    for seed in args.split_seeds:
        split = load_ch_battery_research_split(
            root=args.root, chemistry=args.chemistry, cycle_kind=args.cycle_kind, seed=seed,
            train_ratio=0.70, validation_ratio=0.10,
        )
        destination = output_root / f"seed{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "split.pkl"
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump({"format": FORMAT, "split": split}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(target)
        row = {
            "seed": int(seed), "train_samples": len(split["train"]), "validation_samples": len(split["validation"]),
            "test_samples": len(split["test"]), "train_vins": len(split["train_vins"]),
            "validation_vins": len(split["validation_vins"]), "test_normal_vins": len(split["test_normal_vins"]),
        }
        manifest["split_seeds"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
