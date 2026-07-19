"""Unified external baselines for the NC battery dataset.

The runner integrates the source paper's models plus a standard Isolation
Forest baseline while reusing this project's lazy data loader, vehicle split
and vehicle-level evaluation protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.nc_battery import (
    SnippetRecord,
    aggregate_vehicle_scores,
    build_index,
    load_snippet,
    split_vehicle_folds,
)
from src.models.nc_official_baselines import NCDynamicVAE, NCGDN, NCLSTMAutoEncoder
from src.project_paths import MANUAL_RUNS_ROOT, resolve_dataset_root
from src.runners.train_nc_battery import _metrics


METHODS = ("dyad", "gdn", "auto_encoder", "deepsvdd", "lstm_ad", "isolation_forest")
METHOD_EPOCHS = {
    "dyad": {1: 3, 2: 3, 3: 5},
    "gdn": 20,
    "auto_encoder": 20,
    "deepsvdd": 10,
    "lstm_ad": 20,
    "isolation_forest": 0,
}


class OfficialChannelNormalizer:
    """DyAD's public-code normalizer estimated from the first 200 snippets."""

    def __init__(self, records):
        arrays = [load_snippet(record.path)[0] for record in records[:200]]
        if not arrays:
            raise ValueError("No normal training snippets available for normalization")
        stacked = np.stack(arrays)
        self.mean = np.mean(np.mean(stacked, axis=1), axis=0)
        self.std = np.mean(np.std(stacked, axis=1), axis=0)
        self.minimum = np.min(stacked, axis=(0, 1))
        self.maximum = np.max(stacked, axis=(0, 1))
        self.scale = np.maximum(np.maximum(1e-4, self.std), 0.1 * (self.maximum - self.minimum))
        self.sample_count = len(arrays)

    def transform(self, values):
        return ((values - self.mean) / self.scale).astype(np.float32, copy=False)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "sample_count": self.sample_count}


class PositionStandardizer:
    """Training-only per-position standardization used by official flattened AE/SVDD."""

    def __init__(self, records):
        total = None
        total_square = None
        count = 0
        for record in records:
            # The official AE/SVDD code flattens only the first six channels.
            values = load_snippet(record.path)[0][:, :6].astype(np.float64)
            total = values.copy() if total is None else total + values
            total_square = np.square(values) if total_square is None else total_square + np.square(values)
            count += 1
        if count == 0:
            raise ValueError("No normal training snippets available for normalization")
        self.mean = total / count
        variance = np.maximum(total_square / count - np.square(self.mean), 1e-8)
        self.scale = np.sqrt(variance)

    def transform(self, values):
        return ((values[:, :6] - self.mean) / self.scale).astype(np.float32, copy=False)

    def state_dict(self):
        return {"kind": "train_only_per_position_zscore"}


class FullSnippetDataset(Dataset):
    def __init__(self, records, normalizer=None, feature_count=7):
        self.records = list(records)
        self.normalizer = normalizer
        self.feature_count = int(feature_count)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        values, _ = load_snippet(record.path)
        values = values[:, : self.feature_count]
        if self.normalizer is not None:
            values = self.normalizer.transform(values)
        snippet = f"{Path(record.path).parent.name}/{Path(record.path).stem}"
        mileage = float("nan") if record.mileage is None else float(record.mileage)
        return torch.from_numpy(values).float(), record.car, record.label, snippet, mileage


class FlatAutoEncoder(nn.Module):
    """Eight-layer flattened AE configuration described in Supplementary Note 2."""

    def __init__(self, input_dim=128 * 6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.Sigmoid(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 32), nn.Sigmoid(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.Sigmoid(),
            nn.Linear(64, input_dim),
        )

    def forward(self, values):
        flat = values.flatten(1)
        latent = self.encoder(flat)
        return self.decoder(latent), latent


def _loader(records, normalizer, batch_size, workers, shuffle, feature_count=7):
    return DataLoader(
        FullSnippetDataset(records, normalizer, feature_count=feature_count),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _gdn_windows(values, window=32, stride=16):
    pieces, targets = [], []
    for endpoint in range(window, values.shape[1], stride):
        pieces.append(values[:, endpoint - window:endpoint])
        targets.append(values[:, endpoint])
    return torch.cat(pieces, dim=0), torch.cat(targets, dim=0)


def _train_reconstruction(model, loader, device, method, epochs, learning_rate, brand):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mileage_values = np.asarray([record.mileage for record in loader.dataset.records if record.mileage is not None], dtype=float)
    mileage_min = float(np.min(mileage_values)) if mileage_values.size else 0.0
    mileage_scale = float(np.ptp(mileage_values)) if mileage_values.size else 1.0
    mileage_scale = max(mileage_scale, 1e-6)
    step = 0
    for epoch in range(epochs):
        model.train()
        total, batches = 0.0, 0
        for values, _, _, _, mileage in loader:
            values = values.to(device, non_blocking=True)
            if method == "auto_encoder" and values.shape[0] < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            if method == "auto_encoder":
                reconstruction, _ = model(values)
                loss = nn.functional.mse_loss(reconstruction, values.flatten(1))
            elif method == "dyad":
                response, mean, logvar, mileage_prediction = model(values)
                target = model.target(values)
                reconstruction_loss = nn.functional.smooth_l1_loss(response, target)
                kl = -0.5 * torch.sum(1 + logvar - mean.square() - logvar.exp()) / values.shape[0]
                valid = torch.isfinite(mileage)
                if torch.any(valid):
                    mileage_target = ((mileage[valid].to(device) - mileage_min) / mileage_scale).float()
                    mileage_loss = nn.functional.mse_loss(mileage_prediction[valid], mileage_target)
                else:
                    mileage_loss = torch.zeros((), device=device)
                nll_weight, anneal0, mileage_weight = {
                    1: (10.0, 0.01, 0.001),
                    2: (5.0, 0.1, 1.0),
                    3: (10.0, 0.1, 0.001),
                }[brand]
                kl_weight = anneal0 * min(1.0, step / 500.0)
                loss = nll_weight * reconstruction_loss + kl_weight * kl + mileage_weight * mileage_loss
            elif method == "gdn":
                windows, targets = _gdn_windows(values)
                loss = nn.functional.mse_loss(model(windows), targets)
            elif method == "lstm_ad":
                loss = nn.functional.l1_loss(model(values), values)
            else:
                raise ValueError(method)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
            step += 1
        print(f"[{method}] epoch={epoch + 1}/{epochs} loss={total / max(1, batches):.6f}")


def _train_svdd(loader, device, epochs, learning_rate):
    autoencoder = FlatAutoEncoder().to(device)
    _train_reconstruction(autoencoder, loader, device, "auto_encoder", max(3, epochs // 2), learning_rate, 1)
    encoder = autoencoder.encoder
    encoder.eval()
    latent_parts = []
    with torch.no_grad():
        for values, *_ in loader:
            latent_parts.append(encoder(values.to(device).flatten(1)).cpu())
    center = torch.cat(latent_parts).mean(0).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        encoder.train()
        total, batches = 0.0, 0
        for values, *_ in loader:
            latent = encoder(values.to(device).flatten(1))
            loss = torch.mean(torch.sum((latent - center) ** 2, dim=1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        print(f"[deepsvdd] epoch={epoch + 1}/{epochs} loss={total / max(1, batches):.6f}")
    return encoder, center


def _fit_isolation_forest(records, normalizer, seed, train_sample_limit, trees):
    """Fit the classical baseline on normal snippets (all when limit is zero)."""
    train_sample_limit = int(train_sample_limit)
    sample_count = len(records) if train_sample_limit <= 0 else min(len(records), train_sample_limit)
    if sample_count == len(records):
        indices = np.arange(len(records))
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(records), size=sample_count, replace=False))
    feature_dimension = int(np.prod(normalizer.mean.shape))
    samples = np.empty((sample_count, feature_dimension), dtype=np.float32)
    for output_index, record_index in enumerate(indices):
        values, _ = load_snippet(records[int(record_index)].path)
        samples[output_index] = normalizer.transform(values).reshape(-1)
    model = IsolationForest(
        n_estimators=int(trees),
        max_samples=min(256, sample_count),
        contamination="auto",
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(samples)
    return model, sample_count


def _score_isolation_forest(records, normalizer, model, batch_size):
    scores = []
    batch_size = max(1, int(batch_size))
    feature_dimension = int(np.prod(normalizer.mean.shape))
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        values = np.empty((len(batch_records), feature_dimension), dtype=np.float32)
        for index, record in enumerate(batch_records):
            snippet, _ = load_snippet(record.path)
            values[index] = normalizer.transform(snippet).reshape(-1)
        # sklearn assigns larger scores to normal samples; reverse the sign so
        # all project metrics consistently interpret larger values as anomalous.
        scores.extend((-model.score_samples(values)).tolist())
    return np.asarray(scores, dtype=float)


@torch.no_grad()
def _score_records(method, records, normalizer, model, device, batch_size, workers, auxiliary=None):
    if method == "isolation_forest":
        return _score_isolation_forest(records, normalizer, model, batch_size)
    feature_count = 7 if method == "dyad" else 6
    loader = _loader(records, normalizer, batch_size, workers, False, feature_count=feature_count)
    model.eval()
    scores = []
    for values, *_ in loader:
        values = values.to(device, non_blocking=True)
        if method == "dyad":
            response, *_ = model(values)
            error = torch.mean((response - model.target(values)) ** 2, dim=(1, 2))
        elif method == "auto_encoder":
            reconstruction, _ = model(values)
            error = torch.mean((reconstruction - values.flatten(1)) ** 2, dim=1)
        elif method == "deepsvdd":
            center = auxiliary
            latent = model(values.flatten(1))
            error = torch.sum((latent - center) ** 2, dim=1)
        elif method == "gdn":
            windows, targets = _gdn_windows(values)
            window_error = torch.mean((model(windows) - targets) ** 2, dim=1)
            error = window_error.view(-1, values.shape[0]).transpose(0, 1).mean(1)
        elif method == "lstm_ad":
            reconstruction = model(values)
            error = torch.mean(torch.abs(reconstruction - values), dim=(1, 2))
        else:
            raise ValueError(method)
        scores.extend(error.cpu().numpy().tolist())
    return np.asarray(scores, dtype=float)


def _aggregate_records(records, scores, ratio):
    score_map, car_map, label_map = {}, {}, {}
    for record, score in zip(records, scores):
        snippet = f"{Path(record.path).parent.name}/{Path(record.path).stem}"
        score_map[snippet] = [float(score)]
        car_map[snippet] = record.car
        label_map[record.car] = record.label
    return aggregate_vehicle_scores(score_map, car_map, label_map, ratio)


def _build_result(
    method,
    calibration_records,
    calibration_scores,
    test_records,
    test_scores,
    top_ratio,
    threshold_mode,
):
    calibration_vehicle_scores, calibration_labels, _ = _aggregate_records(
        calibration_records, calibration_scores, top_ratio
    )
    vehicle_scores, labels, cars = _aggregate_records(test_records, test_scores, top_ratio)
    sensitivity = {}
    for ratio in (0.01, 0.05, 0.10, 0.20, 1.0):
        calibrated_scores, calibrated_labels, _ = _aggregate_records(
            calibration_records, calibration_scores, ratio
        )
        scores, ratio_labels, _ = _aggregate_records(test_records, test_scores, ratio)
        key = f"top_{int(ratio * 100)}pct" if ratio < 1 else "mean"
        sensitivity[key] = _metrics(
            scores,
            ratio_labels,
            calibrated_scores,
            calibrated_labels,
            threshold_mode,
        )
    return {
        "method": method,
        "metrics": _metrics(
            vehicle_scores,
            labels,
            calibration_vehicle_scores,
            calibration_labels,
            threshold_mode,
        ),
        "aggregation_sensitivity": sensitivity,
    }, vehicle_scores, labels, cars


def _implementation_metadata(method):
    metadata = {
        "dyad": (
            "integrated_compatibility",
            "DyAD/model/dynamic_vae.py",
            "project-local modern PyTorch implementation retaining the public brand-specific DynamicVAE architecture and hyperparameters",
        ),
        "gdn": (
            "integrated_compatibility",
            "GDN_battery/models/GDN.py",
            "project-local pure-PyTorch implementation of the public learned top-k battery graph, without the legacy PyG dependency",
        ),
        "lstm_ad": (
            "integrated_compatibility",
            "Recurrent-Autoencoder-modify/graphs/models/recurrent_autoencoder.py",
            "project-local modern PyTorch implementation retaining the public recurrent encoder/autoregressive decoder",
        ),
        "auto_encoder": (
            "compatibility_port",
            "AE_and_SVDD/traditional_methods.py",
            "official flattened six-channel architecture/config retained; streaming loader replaces legacy PyOD input path",
        ),
        "deepsvdd": (
            "compatibility_port",
            "AE_and_SVDD/traditional_methods.py",
            "official six-channel/10-epoch protocol retained; current-runtime torch training replaces legacy PyOD API",
        ),
        "isolation_forest": (
            "integrated_sklearn",
            "sklearn.ensemble.IsolationForest",
            "classical normal-only baseline fitted from all training snippets; each tree uses the standard bounded random subsample and all validation/test snippets are scored",
        ),
    }
    kind, source_file, note = metadata[method]
    return {
        "implementation_kind": kind,
        "official_source_file": source_file,
        "implementation_note": note,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Unified external baselines on the NC battery data")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--battery_brand", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--battery_fold", type=int, choices=range(5), required=True)
    parser.add_argument(
        "--battery_split_protocol",
        choices=("strict_normal_validation", "paper_protocol"),
        default="strict_normal_validation",
    )
    parser.add_argument("--tsinghua_ev_root", default=str(resolve_dataset_root("TSINGHUA-EV", "TSINGHUA_EV")))
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--epochs", type=int, default=0, help="0 uses the paper's method/brand setting")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--vehicle_top_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_snippets_per_vehicle", type=int, default=0, help="Smoke-test cap; 0 keeps full data")
    parser.add_argument(
        "--isolation_train_samples",
        type=int,
        default=0,
        help="Maximum normal training snippets for Isolation Forest; 0 keeps all.",
    )
    parser.add_argument("--isolation_trees", type=int, default=200)
    parser.add_argument("--require_cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this external baseline")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = build_index(args.tsinghua_ev_root, args.battery_brand)
    splits = split_vehicle_folds(
        records,
        args.battery_fold,
        seed=args.seed,
        protocol=args.battery_split_protocol,
    )
    if args.max_snippets_per_vehicle > 0:
        capped_splits = {}
        for split_name, split_records in splits.items():
            counts = {}
            capped_splits[split_name] = []
            for record in split_records:
                if counts.get(record.car, 0) >= args.max_snippets_per_vehicle:
                    continue
                capped_splits[split_name].append(record)
                counts[record.car] = counts.get(record.car, 0) + 1
        splits = capped_splits

    normalizer = None
    if args.method == "dyad":
        normalizer = OfficialChannelNormalizer(splits["train"])
    elif args.method in {"auto_encoder", "deepsvdd", "isolation_forest"}:
        normalizer = PositionStandardizer(splits["train"])

    feature_count = 7 if args.method == "dyad" else 6
    train_loader = None
    if args.method != "isolation_forest":
        train_loader = _loader(
            splits["train"], normalizer, args.batch_size, args.num_workers, True,
            feature_count=feature_count,
        )
    model = None
    auxiliary = None
    isolation_fit_samples = None
    epochs = args.epochs
    if epochs <= 0:
        default_epochs = METHOD_EPOCHS.get(args.method, 0)
        epochs = default_epochs[args.battery_brand] if isinstance(default_epochs, dict) else default_epochs

    if args.method == "dyad":
        paper_lr = {1: 0.005, 2: 0.0001, 3: 0.005}[args.battery_brand]
        model = NCDynamicVAE(args.battery_brand).to(device)
        _train_reconstruction(model, train_loader, device, args.method, epochs, paper_lr, args.battery_brand)
    elif args.method == "auto_encoder":
        model = FlatAutoEncoder().to(device)
        _train_reconstruction(model, train_loader, device, args.method, epochs, args.learning_rate, args.battery_brand)
    elif args.method == "deepsvdd":
        model, auxiliary = _train_svdd(train_loader, device, epochs, args.learning_rate)
    elif args.method == "gdn":
        model = NCGDN().to(device)
        _train_reconstruction(model, train_loader, device, args.method, epochs, args.learning_rate, args.battery_brand)
    elif args.method == "lstm_ad":
        model = NCLSTMAutoEncoder().to(device)
        _train_reconstruction(model, train_loader, device, args.method, epochs, args.learning_rate, args.battery_brand)
    elif args.method == "isolation_forest":
        model, isolation_fit_samples = _fit_isolation_forest(
            splits["train"], normalizer, args.seed,
            args.isolation_train_samples, args.isolation_trees,
        )

    calibration_scores = _score_records(
        args.method, splits["calibration"], normalizer, model, device,
        args.batch_size, args.num_workers, auxiliary,
    )
    test_scores = _score_records(
        args.method, splits["test"], normalizer, model, device,
        args.batch_size, args.num_workers, auxiliary,
    )
    result, vehicle_scores, labels, cars = _build_result(
        args.method,
        splits["calibration"],
        calibration_scores,
        splits["test"],
        test_scores,
        args.vehicle_top_ratio,
        (
            "paper_labelled_f1"
            if args.battery_split_protocol == "paper_protocol"
            else "normal_p99"
        ),
    )
    result.update({
        "source": "Zhang et al., Nature Communications 2023 public-architecture compatibility implementation",
        "brand": args.battery_brand,
        "fold": args.battery_fold,
        "epochs": epochs,
        "protocol": args.battery_split_protocol,
        "protocol_source": (
            "project strict normal-only calibration"
            if args.battery_split_protocol == "strict_normal_validation"
            else "Zhang et al. 2023 Supplementary Note 2, Five fold evaluation"
        ),
        "counts": {
            "train_snippets": len(splits["train"]),
            "calibration_snippets": len(splits["calibration"]),
            "test_snippets": len(splits["test"]),
            "train_vehicles": len({record.car for record in splits["train"]}),
            "calibration_vehicles": len({record.car for record in splits["calibration"]}),
            "test_vehicles": len({record.car for record in splits["test"]}),
        },
        "normalizer": None if normalizer is None else normalizer.state_dict(),
        **_implementation_metadata(args.method),
    })
    if args.method == "isolation_forest":
        result["isolation_forest"] = {
            "fit_samples": isolation_fit_samples,
            "train_sample_limit": args.isolation_train_samples,
            "n_estimators": args.isolation_trees,
            "max_samples_per_tree": min(256, isolation_fit_samples),
        }
    output_dir = Path(args.output_dir) if args.output_dir else (
        MANUAL_RUNS_ROOT / "NC_EXTERNAL" / f"{args.method}_b{args.battery_brand}_f{args.battery_fold}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "isolation_forest":
        with (output_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif model is not None:
        torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "vehicle_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["car", "label", "score"])
        writer.writerows(zip(cars, labels.tolist(), vehicle_scores.tolist()))
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
