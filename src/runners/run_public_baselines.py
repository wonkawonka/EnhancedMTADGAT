"""Small, reproducible public-data baselines used by the comparison table.

This runner deliberately keeps the added baselines self-contained: PCA/SPE is a
classical point-wise subspace detector and USAD is the two-autoencoder window
detector described by Audibert et al.  Both use the project's MSL/SMAP loader,
training-only normalization and raw point-level AP/AUROC.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.utils import get_nasa_telemetry_sequence_data


class WindowReferenceDataset(Dataset):
    def __init__(self, sequences, lookback, stride=1, references=None, labels=None):
        self.sequences = list(sequences)
        self.lookback = int(lookback)
        self.stride = max(int(stride), 1)
        self.labels = labels
        if references is None:
            self.references = [
                (sequence_index, start)
                for sequence_index, sequence in enumerate(self.sequences)
                for start in range(0, max(0, len(sequence) - self.lookback), self.stride)
            ]
        else:
            self.references = list(references)

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        sequence_index, start = self.references[index]
        values = self.sequences[sequence_index][start : start + self.lookback]
        window = torch.from_numpy(np.asarray(values, dtype=np.float32)).flatten()
        if self.labels is None:
            return window
        label = int(np.asarray(self.labels[sequence_index])[start + self.lookback])
        return window, label


class TwoAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        hidden_dim = max(8, min(int(hidden_dim), input_dim))
        latent_dim = max(4, min(int(latent_dim), hidden_dim))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, values):
        return self.decoder(self.encoder(values))


def _device(use_cuda):
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _runtime_base(device, model_parameters=0):
    result = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_total_memory_mb": (
            float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
            if device.type == "cuda" else None
        ),
        "model_parameters": int(model_parameters),
    }
    return result


def _write_result(output_dir, args, metrics, runtime):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with (output_dir / "runtime.json").open("w", encoding="utf-8") as handle:
        json.dump(runtime, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"metrics": metrics, "runtime": runtime}, ensure_ascii=False, indent=2))


def _ranking_metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    result = {
        "sample_count": int(len(labels)),
        "anomaly_count": int(np.sum(labels > 0)),
        "average_precision": float(average_precision_score(labels, scores)),
    }
    result["auroc"] = float(roc_auc_score(labels, scores)) if np.unique(labels).size > 1 else None
    return result


def run_pca_spe(args, train_sequences, test_sequences, test_labels, output_dir):
    device = torch.device("cpu")
    started = time.perf_counter()
    train_points = np.concatenate(train_sequences, axis=0).astype(np.float32, copy=False)
    rng = np.random.default_rng(int(args.seed))
    if args.sample_limit > 0 and len(train_points) > args.sample_limit:
        train_points = train_points[rng.choice(len(train_points), int(args.sample_limit), replace=False)]
    components = max(1, min(int(args.pca_components), train_points.shape[1], len(train_points)))
    pca = IncrementalPCA(n_components=components, batch_size=max(components * 4, 256))
    for start in range(0, len(train_points), 4096):
        batch = train_points[start : start + 4096]
        if len(batch) >= components:
            pca.partial_fit(batch)
    preprocessing_seconds = time.perf_counter() - started

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    score_parts, label_parts = [], []
    for sequence, labels in zip(test_sequences, test_labels):
        transformed = pca.transform(sequence)
        reconstruction = pca.inverse_transform(transformed)
        score_parts.append(np.mean((sequence - reconstruction) ** 2, axis=1))
        label_parts.append(np.asarray(labels, dtype=np.int32))
    inference_seconds = time.perf_counter() - inference_started
    scores = np.concatenate(score_parts)
    labels = np.concatenate(label_parts)
    runtime = _runtime_base(device, 0)
    runtime.update({
        "preprocessing_seconds": float(preprocessing_seconds),
        "training_seconds": 0.0,
        "inference_seconds": float(inference_seconds),
        "window_count": int(len(scores)),
        "windows_per_second": float(len(scores) / max(inference_seconds, 1e-9)),
        "milliseconds_per_window": float(1000.0 * inference_seconds / max(len(scores), 1)),
        "peak_cuda_memory_mb": 0.0,
    })
    metrics = {"dataset": args.dataset, "method": "pca_spe", "components": components}
    metrics.update(_ranking_metrics(scores, labels))
    _write_result(output_dir, args, metrics, runtime)


def _make_references(sequences, lookback, stride, limit, seed):
    references = [
        (sequence_index, start)
        for sequence_index, sequence in enumerate(sequences)
        for start in range(0, max(0, len(sequence) - lookback), stride)
    ]
    if limit > 0 and len(references) > limit:
        rng = np.random.default_rng(int(seed))
        references = [references[index] for index in rng.choice(len(references), limit, replace=False)]
    return references


def run_usad(args, train_sequences, test_sequences, test_labels, output_dir):
    device = _device(args.use_cuda)
    started = time.perf_counter()
    train_references = _make_references(
        train_sequences, args.lookback, args.stride, args.sample_limit, args.seed
    )
    train_dataset = WindowReferenceDataset(
        train_sequences, args.lookback, args.stride, references=train_references
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    input_dim = int(args.lookback * train_sequences[0].shape[1])
    ae1 = TwoAutoEncoder(input_dim, args.hidden_dim, args.latent_dim).to(device)
    ae2 = TwoAutoEncoder(input_dim, args.hidden_dim, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(list(ae1.parameters()) + list(ae2.parameters()), lr=args.learning_rate)
    preprocessing_seconds = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    total_steps = max(1, len(train_loader))
    for epoch in range(int(args.epochs)):
        ae1.train(); ae2.train()
        for step, values in enumerate(train_loader):
            values = values.to(device, non_blocking=True)
            reconstruction1 = ae1(values)
            reconstruction2 = ae2(values)
            chained = ae2(reconstruction1.detach())
            fraction = float(epoch * total_steps + step + 1) / max(1, int(args.epochs) * total_steps)
            loss1 = fraction * nn.functional.mse_loss(reconstruction1, values)
            loss1 = loss1 + (1.0 - fraction) * nn.functional.mse_loss(chained, values)
            loss2 = fraction * nn.functional.mse_loss(reconstruction2, values)
            loss2 = loss2 - (1.0 - fraction) * nn.functional.mse_loss(chained, values)
            optimizer.zero_grad(set_to_none=True)
            (loss1 + loss2).backward()
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started

    test_dataset = WindowReferenceDataset(test_sequences, args.lookback, args.stride, labels=test_labels)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    inference_started = time.perf_counter()
    scores, labels = [], []
    ae1.eval(); ae2.eval()
    with torch.no_grad():
        for values, batch_labels in test_loader:
            values = values.to(device, non_blocking=True)
            err1 = (ae1(values) - values).square().mean(dim=1)
            err2 = (ae2(values) - values).square().mean(dim=1)
            scores.append((0.5 * (err1 + err2)).cpu().numpy())
            labels.append(np.asarray(batch_labels, dtype=np.int32))
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    parameter_count = sum(parameter.numel() for parameter in ae1.parameters()) + sum(
        parameter.numel() for parameter in ae2.parameters()
    )
    runtime = _runtime_base(device, parameter_count)
    runtime.update({
        "preprocessing_seconds": float(preprocessing_seconds),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "window_count": int(len(scores)),
        "windows_per_second": float(len(scores) / max(inference_seconds, 1e-9)),
        "milliseconds_per_window": float(1000.0 * inference_seconds / max(len(scores), 1)),
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0
        ),
    })
    metrics = {"dataset": args.dataset, "method": "usad"}
    metrics.update(_ranking_metrics(scores, labels))
    _write_result(output_dir, args, metrics, runtime)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PCA/SPE or USAD on MSL/SMAP.")
    parser.add_argument("--method", choices=["pca_spe", "usad"], required=True)
    parser.add_argument("--dataset", choices=["MSL", "SMAP"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--normalize", type=lambda value: str(value).lower() == "true", default=True)
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--pca_components", type=int, default=8)
    parser.add_argument("--sample_limit", type=int, default=100000)
    parser.add_argument("--use_cuda", type=lambda value: str(value).lower() == "true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_sequences, _, test_sequences, test_labels = get_nasa_telemetry_sequence_data(
        args.dataset, val_ratio=0.0, normalize=args.normalize
    )
    output_dir = Path(args.output_dir).resolve()
    if args.method == "pca_spe":
        run_pca_spe(args, train_sequences, test_sequences, test_labels, output_dir)
    else:
        run_usad(args, train_sequences, test_sequences, test_labels, output_dir)


if __name__ == "__main__":
    main()
