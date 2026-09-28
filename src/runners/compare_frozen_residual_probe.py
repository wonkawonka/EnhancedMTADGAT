"""Evaluate native frozen baseline scores on the probe's identical selected endpoints."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.external_protocol import load_external_protocol_data
from src.runners.run_frozen_residual_router_probe import (
    FrozenWindows, load_frozen, top_five_percent,
)
from src.runners.run_unified_external_baselines import _batch_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-root", type=Path,
        default=Path("runs/internal/residual_router_probe_lfp_seed3407"),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen baseline comparison")
    device = torch.device(args.device)
    data = load_external_protocol_data(
        "CH_LFP_MIXED_CYCLE", seed=3407, ch_split_seed=3407
    )
    results = []
    for method in ("mtad_gat", "tranad"):
        target = args.probe_root / method
        probe = pd.read_csv(target / "test_sample_scores.csv")
        model, config, checkpoint = load_frozen(method, device)
        windows = FrozenWindows(data.test_sequences, config["window"])
        loader = DataLoader(windows, batch_size=256, shuffle=False, num_workers=0)
        by_sample = defaultdict(list)
        with torch.inference_mode():
            for values, samples, _ in loader:
                values = values.to(device)
                native = _batch_scores(
                    method, model, None, values, config["window"]
                )
                for sample, score in zip(samples.tolist(), native.tolist()):
                    by_sample[int(sample)].append(float(score))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rows = []
        for index, sample_id in enumerate(data.entity_ids):
            if index not in by_sample:
                continue
            label = int(np.asarray(data.test_labels[index]).reshape(-1)[0])
            rows.append({
                "sample_id": sample_id, "label": label,
                "native_baseline": top_five_percent(by_sample[index]),
                "n_endpoints": len(by_sample[index]),
            })
        baseline = pd.DataFrame(rows)
        merged = probe.merge(
            baseline, on=["sample_id", "label", "n_endpoints"],
            validate="one_to_one",
        )
        if len(merged) != len(probe) or len(baseline) != len(probe):
            raise RuntimeError("Native baseline and probe samples do not match")
        if not np.array_equal(
            merged["sample_id"].to_numpy(), probe["sample_id"].to_numpy()
        ):
            raise RuntimeError("Sample ordering changed")
        baseline.to_csv(target / "native_baseline_same_endpoints.csv", index=False)
        labels = merged["label"].to_numpy(dtype=int)
        row = {"method": method, "sample_count": len(merged),
               "checkpoint": str(checkpoint)}
        for name, column in (
            ("native", "native_baseline"),
            ("slow", "slow_z"),
            ("fast", "fast_z"),
            ("relation", "relation_z"),
            ("fixed_average", "avg"),
            ("normal_only_route", "routed"),
        ):
            score = merged[column].to_numpy(dtype=float)
            row[name + "_auroc"] = float(roc_auc_score(labels, score))
            row[name + "_ap"] = float(average_precision_score(labels, score))
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.probe_root / "native_baseline_comparison.json").write_text(
        json.dumps({
            "protocol": "same_test_samples_same_selected_endpoints_frozen_checkpoints",
            "sample_aggregation": "top_5pct_selected_endpoints",
            "test_label_use": "metrics_only",
            "results": results,
        }, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
