"""Frozen MTAD-GAT/TranAD residual evidence and normal-only condition routing probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.data.external_protocol import load_external_protocol_data
from src.models.mtad_gat import Enhanced_MTADGAT
from src.models.unified_external_baselines import build_reconstruction_model

BRANCHES = ("slow", "fast", "relation")
METHODS = ("mtad_gat", "tranad")
SOURCE_ROOT = Path("runs/internal/research_13_lfp_mixed_cycle_formal")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_endpoints(length: int, window: int = 32) -> list[tuple[int, int]]:
    """Four separated eight-window blocks, each on the native stride-4 grid."""
    grid = list(range(window - 1, length, 4))
    if len(grid) <= 32:
        return [(endpoint, 0) for endpoint in grid]
    starts = np.linspace(0, len(grid) - 8, 4).round().astype(int)
    return [(grid[start + offset], block)
            for block, start in enumerate(starts) for offset in range(8)]


class FrozenWindows(Dataset):
    def __init__(self, sequences: list[np.ndarray], model_window: int):
        self.sequences = sequences
        self.window = model_window
        self.refs = [(sample, endpoint, block)
                     for sample, sequence in enumerate(sequences)
                     for endpoint, block in selected_endpoints(len(sequence))]

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, item: int):
        sample, endpoint, block = self.refs[item]
        values = self.sequences[sample][endpoint - self.window + 1:endpoint + 1]
        return torch.from_numpy(np.asarray(values, dtype=np.float32)), sample, block


def load_frozen(method: str, device: torch.device):
    path = SOURCE_ROOT / (method + "_seed3407")
    config = json.loads((path / "config.json").read_text())
    if config["dataset"] != "CH_LFP_MIXED_CYCLE" or config["seed"] != 3407:
        raise RuntimeError("Checkpoint protocol does not match mixed-cycle LFP seed 3407")
    if config["window"] != (32 if method == "mtad_gat" else 10):
        raise RuntimeError("Unexpected native window")
    if method == "mtad_gat":
        model = Enhanced_MTADGAT(
            n_features=7, window_size=32, out_dim=7, gru_hid_dim=64,
            forecast_hid_dim=64, recon_hid_dim=64, dropout=0.2,
        )
    else:
        model = build_reconstruction_model("tranad", 7, 10, 64, 32)
    checkpoint = path / "model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model, config, checkpoint


@torch.inference_mode()
def residuals_by_sample(method: str, model, sequences, device):
    windows = FrozenWindows(sequences, 32 if method == "mtad_gat" else 10)
    loader = DataLoader(windows, batch_size=256, shuffle=False, num_workers=0)
    grouped = defaultdict(list)
    for values, samples, blocks in loader:
        values = values.to(device)
        if method == "mtad_gat":
            _, reconstruction = model(values)
        else:
            _, reconstruction = model(values)
        residual = (values[:, -1] - reconstruction[:, -1]).cpu().numpy()
        for sample, block, vector in zip(samples.tolist(), blocks.tolist(), residual):
            grouped[int(sample)].append((int(block), vector.astype(np.float32)))
    return grouped, len(windows)


def endpoint_scores(entries, precision):
    by_block = defaultdict(list)
    for block, residual in entries:
        by_block[block].append(residual)
    branch_values = {name: [] for name in BRANCHES}
    for block in sorted(by_block):
        residual = np.asarray(by_block[block], dtype=np.float64)
        slow = np.empty_like(residual)
        for index in range(len(residual)):
            slow[index] = residual[max(0, index - 4):index + 1].mean(axis=0)
        fast = residual - slow
        branch_values["slow"].extend(np.mean(slow * slow, axis=1).tolist())
        branch_values["fast"].extend(np.mean(fast * fast, axis=1).tolist())
        branch_values["relation"].extend(
            (np.einsum("ni,ij,nj->n", residual, precision, residual) / residual.shape[1]).tolist()
        )
    return branch_values


def top_five_percent(values):
    values = np.asarray(values, dtype=np.float64)
    count = max(1, int(np.ceil(len(values) * 0.05)))
    return float(np.partition(values, -count)[-count:].mean())


def cycle_of(metadata):
    cycle = str(metadata["cycle_kind"]).lower()
    if cycle not in ("charge", "discharge"):
        raise ValueError("Unexpected cycle kind " + cycle)
    return cycle


def dynamic_value(sequence):
    current = np.asarray(sequence[:, 1], dtype=np.float64)
    return float(np.abs(np.diff(current)).mean()) if len(current) > 1 else 0.0


def score_partition(grouped, sequences, ids, metadata, precision, partition):
    rows = []
    excluded = []
    for index, (sample_id, sequence) in enumerate(zip(ids, sequences)):
        if index not in grouped:
            excluded.append(sample_id)
            continue
        evidence = endpoint_scores(grouped[index], precision)
        row = {
            "partition": partition, "sample_id": sample_id,
            "label": int(metadata[sample_id]["sample_label"]),
            "vin": str(metadata[sample_id]["vin"]),
            "fault_type": str(metadata[sample_id]["fault_type"]),
            "cycle": cycle_of(metadata[sample_id]),
            "dynamics_value": dynamic_value(sequence),
            "n_endpoints": len(grouped[index]),
        }
        row.update({name: top_five_percent(evidence[name]) for name in BRANCHES})
        rows.append(row)
    return pd.DataFrame(rows), excluded


def score_metrics(labels, values):
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, values)), float(average_precision_score(labels, values))


def fit_scale_and_route(train, test):
    if set(train["label"]) != {0}:
        raise RuntimeError("Router calibration must use normal training samples only")
    cutoffs = {}
    for cycle in ("charge", "discharge"):
        subset = train.loc[train.cycle == cycle, "dynamics_value"].to_numpy()
        if len(subset) < 10:
            raise RuntimeError("Insufficient normal training samples in " + cycle)
        cutoffs[cycle] = np.quantile(subset, [1 / 3, 2 / 3]).tolist()
    for frame in (train, test):
        frame["dynamics"] = [
            ("low" if value <= cutoffs[cycle][0] else
             "medium" if value <= cutoffs[cycle][1] else "high")
            for cycle, value in zip(frame.cycle, frame.dynamics_value)
        ]
        frame["group"] = frame.cycle + "/" + frame.dynamics
    normalization = {}
    for name in BRANCHES:
        transformed = np.log1p(train[name].to_numpy(dtype=np.float64))
        q25, median, q75 = np.quantile(transformed, [0.25, 0.5, 0.75])
        scale = max(float(q75 - q25), 1e-6)
        normalization[name] = {"median": float(median), "iqr": scale}
        for frame in (train, test):
            frame[name + "_z"] = (
                np.log1p(frame[name].to_numpy(dtype=np.float64)) - median
            ) / scale
    weights = {}
    for group in sorted(set(train.group)):
        selected = train.loc[train.group == group]
        if len(selected) < 10:
            raise RuntimeError("Insufficient normal routing samples in " + group)
        tail = np.asarray([
            max(0.1, float(np.quantile(selected[name + "_z"], 0.95)))
            for name in BRANCHES
        ])
        inverse = 1.0 / tail
        weights[group] = (inverse / inverse.sum()).tolist()
    for frame in (train, test):
        evidence = frame[[name + "_z" for name in BRANCHES]].to_numpy()
        frame["avg"] = evidence.mean(axis=1)
        frame["routed"] = np.asarray([
            np.dot(weights[group], row)
            for group, row in zip(frame.group, evidence)
        ])
    return cutoffs, normalization, weights


def evaluate(test):
    rows = []
    for group_name, subset in [("all", test)] + [
        (name, test.loc[test.group == name]) for name in sorted(set(test.group))
    ] + [
        (cycle, test.loc[test.cycle == cycle]) for cycle in ("charge", "discharge")
    ] + [
        (dynamic, test.loc[test.dynamics == dynamic])
        for dynamic in ("low", "medium", "high")
    ]:
        labels = subset.label.to_numpy(dtype=int)
        for branch in BRANCHES + ("avg", "routed"):
            result = score_metrics(labels, subset[branch + "_z"] if branch in BRANCHES else subset[branch])
            rows.append({
                "group": group_name, "score": branch, "n": len(subset),
                "normal": int((labels == 0).sum()), "fault": int((labels == 1).sum()),
                "auroc": None if result is None else result[0],
                "ap": None if result is None else result[1],
            })
    return pd.DataFrame(rows)


def run_method(method, data, output, device):
    model, config, checkpoint = load_frozen(method, device)
    train_grouped, train_windows = residuals_by_sample(
        method, model, data.train_sequences, device
    )
    test_grouped, test_windows = residuals_by_sample(
        method, model, data.test_sequences, device
    )
    del model
    torch.cuda.empty_cache()
    training_residual = np.asarray(
        [vector for entries in train_grouped.values() for _, vector in entries],
        dtype=np.float64,
    )
    covariance = LedoitWolf().fit(training_residual)
    precision = covariance.precision_
    train_ids = sorted(data.metadata["train_sample_metadata"])
    train, train_excluded = score_partition(
        train_grouped, data.train_sequences, train_ids,
        data.metadata["train_sample_metadata"], precision, "train"
    )
    test, test_excluded = score_partition(
        test_grouped, data.test_sequences, data.entity_ids,
        data.metadata["sample_metadata"], precision, "test"
    )
    cutoffs, normalization, weights = fit_scale_and_route(train, test)
    metrics = evaluate(test)
    target = output / method
    target.mkdir(parents=True, exist_ok=True)
    train.to_csv(target / "train_sample_scores.csv", index=False)
    test.to_csv(target / "test_sample_scores.csv", index=False)
    metrics.to_csv(target / "metrics.csv", index=False)
    manifest = {
        "dataset": "CH_LFP_MIXED_CYCLE",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "model_seed": 3407, "split_seed": 3407,
        "native_window": config["window"], "common_endpoint_start": 31,
        "endpoint_stride": 4, "blocks_per_sample": 4, "endpoints_per_block": 8,
        "slow_ma_endpoints": 5, "sample_aggregation": "top_5pct_endpoints",
        "covariance": "LedoitWolf_train_normal_endpoint_residual",
        "score_scale": "log1p_train_normal_sample_median_IQR",
        "route": "inverse_train_normal_group_q95_after_global_scale",
        "train_windows": train_windows, "test_windows": test_windows,
        "train_samples": len(train), "test_samples": len(test),
        "excluded_train": train_excluded, "excluded_test": test_excluded,
        "dynamics_train_normal_tertiles": cutoffs,
        "score_normalization": normalization, "route_weights": weights,
        "test_labels_used_for_route": False,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({
        "method": method, "train_windows": train_windows,
        "test_windows": test_windows, "excluded_test": len(test_excluded),
        "overall": metrics.loc[metrics.group == "all",
                               ["score", "auroc", "ap"]].round(4).to_dict("records"),
    }, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/internal/residual_router_probe_lfp_seed3407"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen residual probe requires CUDA")
    device = torch.device(args.device)
    torch.manual_seed(3407)
    np.random.seed(3407)
    data = load_external_protocol_data(
        "CH_LFP_MIXED_CYCLE", seed=3407, ch_split_seed=3407
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        target = args.output / method / "manifest.json"
        if target.exists():
            print("SKIP", method, "existing manifest", flush=True)
            continue
        run_method(method, data, args.output, device)


if __name__ == "__main__":
    main()
