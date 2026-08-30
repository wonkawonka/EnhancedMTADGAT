"""Run a DyAD-style DynamicVAE compatibility experiment on the private BMS data.

The DyAD paper defines a seven-channel NC-Battery task and does not publish a
BMS task.  This runner therefore keeps the DynamicVAE encoder/decoder and
training objective, while making the BMS adaptation explicit: all 35 BMS
features enter the encoder, ``SYS_I`` and ``BMSnRSOC`` are the two condition
channels supplied to the decoder, and the remaining 33 channels are scored as
responses.  The result must be reported as *DyAD-style BMS compatibility*, not
as an original paper DyAD-BMS number.

BMS is released as a normal-only interval.  We follow the existing external
protocol: the official training interval is split per cluster into a 90/10
model/calibration part, the independent test interval is scored with
``lookback=100`` and ``stride=10``, and the threshold is the pooled validation
P99.  Only false-alarm/stability statistics are emitted; supervised metrics
are intentionally absent.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.regime_utils import BMS_REGIME_NAMES, derive_bms_regime_labels
from src.data.utils import BMS_FEATURE_NAMES, get_bms_cluster_data
from src.project_paths import EXTERNAL_RUNS_ROOT


class BMSWindowDataset(Dataset):
    """Lazy fixed-window view preserving the originating BMS cluster."""

    def __init__(self, sequences, window: int, stride: int, limit: int = 0, seed: int = 3407):
        self.sequences = [np.asarray(sequence, dtype=np.float32) for sequence in sequences]
        self.window = int(window)
        self.stride = max(1, int(stride))
        references = [
            (sequence_index, start)
            for sequence_index, sequence in enumerate(self.sequences)
            for start in range(0, max(0, len(sequence) - self.window), self.stride)
        ]
        if int(limit) > 0 and len(references) > int(limit):
            rng = np.random.default_rng(int(seed))
            selected = np.sort(rng.choice(len(references), int(limit), replace=False))
            references = [references[int(index)] for index in selected]
        self.references = references

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        sequence_index, start = self.references[index]
        values = self.sequences[sequence_index][start : start + self.window]
        endpoint = start + self.window - 1
        return torch.from_numpy(values), int(sequence_index), int(endpoint)


class DyADBMS(nn.Module):
    """DynamicVAE with an explicit BMS condition/response task.

    The bidirectional two-GRU/latent-VAE shape follows the public DyAD
    DynamicVAE.  The only task-specific choice is the BMS channel split.
    """

    def __init__(
        self,
        n_features: int = 35,
        condition_indices: tuple[int, ...] = (22, 3),
        hidden_size: int = 256,
        latent_size: int = 16,
        num_layers: int = 1,
        noise_scale: float = 0.01,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.condition_indices = tuple(int(index) for index in condition_indices)
        self.order = self.condition_indices + tuple(
            index for index in range(self.n_features) if index not in self.condition_indices
        )
        self.response_indices = self.order[len(self.condition_indices) :]
        self.condition_count = len(self.condition_indices)
        self.response_count = len(self.response_indices)
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.num_layers = int(num_layers)
        self.noise_scale = float(noise_scale)
        self.bidirectional = True
        self.hidden_factor = 2 * self.num_layers

        self.encoder_rnn = nn.GRU(
            self.n_features,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.decoder_rnn = nn.GRU(
            self.condition_count,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )
        hidden_dimension = self.hidden_size * self.hidden_factor
        self.hidden2mean = nn.Linear(hidden_dimension, self.latent_size)
        self.hidden2log_v = nn.Linear(hidden_dimension, self.latent_size)
        self.latent2hidden = nn.Linear(self.latent_size, hidden_dimension)
        self.outputs2embedding = nn.Linear(self.hidden_size * 2, self.response_count)

    def _ordered(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, :, list(self.order)]

    def target(self, values: torch.Tensor) -> torch.Tensor:
        return self._ordered(values)[:, :, self.condition_count :]

    def forward(self, values: torch.Tensor):
        ordered = self._ordered(values)
        _, hidden = self.encoder_rnn(ordered)
        batch_size = values.shape[0]
        hidden_flat = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        mean = self.hidden2mean(hidden_flat)
        log_v = self.hidden2log_v(hidden_flat)
        std = torch.exp(0.5 * log_v)
        latent = mean
        if self.training:
            latent = mean + torch.randn_like(std) * std * self.noise_scale
        decoder_hidden = self.latent2hidden(latent)
        decoder_hidden = decoder_hidden.view(batch_size, self.hidden_factor, self.hidden_size)
        decoder_hidden = decoder_hidden.transpose(0, 1).contiguous()
        decoded, _ = self.decoder_rnn(ordered[:, :, : self.condition_count], decoder_hidden)
        response = self.outputs2embedding(decoded)
        return response, mean, log_v


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(args) -> torch.device:
    if args.use_cuda and not torch.cuda.is_available():
        if args.require_cuda:
            raise RuntimeError("CUDA is required but unavailable")
        print("[dyad_bms] CUDA unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")


def _make_loaders(train_sequences, validation_sequences, test_sequences, args):
    train_dataset = BMSWindowDataset(
        train_sequences, args.lookback, args.stride, args.train_window_limit, args.seed
    )
    validation_dataset = BMSWindowDataset(
        validation_sequences, args.lookback, args.stride, args.eval_window_limit, args.seed
    )
    test_dataset = BMSWindowDataset(
        test_sequences, args.lookback, args.stride, args.eval_window_limit, args.seed
    )
    kwargs = dict(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available() and bool(args.use_cuda),
    )
    return (
        DataLoader(train_dataset, shuffle=True, drop_last=False, **kwargs),
        DataLoader(validation_dataset, shuffle=False, **kwargs),
        DataLoader(test_dataset, shuffle=False, **kwargs),
    )


@torch.no_grad()
def _score_loader(model, loader, device):
    model.eval()
    scores = []
    sequence_ids = []
    endpoints = []
    for values, ids, ends in loader:
        values = values.to(device, non_blocking=True)
        response, _, _ = model(values)
        target = model.target(values)
        score = (response - target).square().mean(dim=(1, 2))
        scores.extend(score.detach().cpu().numpy().astype(np.float64).tolist())
        sequence_ids.extend(ids.numpy().astype(np.int32).tolist())
        endpoints.extend(ends.numpy().astype(np.int64).tolist())
    return np.asarray(scores, dtype=np.float64), np.asarray(sequence_ids, dtype=np.int32), np.asarray(endpoints, dtype=np.int64)


def _cluster_rows(scores, sequence_ids, threshold, cluster_names):
    rows = []
    for index, cluster in enumerate(cluster_names):
        values = scores[sequence_ids == index]
        alarms = values > threshold
        rate = float(np.mean(alarms)) if len(values) else None
        rows.append({
            "cluster": cluster,
            "window_count": int(len(values)),
            "false_alarm_count": int(np.sum(alarms)),
            "false_alarm_rate": rate,
            "false_alarms_per_10k_windows": None if rate is None else rate * 10000.0,
        })
    return rows


def _block_rows(scores, sequence_ids, threshold, block_size, cluster_names):
    rows = []
    for index, cluster in enumerate(cluster_names):
        values = scores[sequence_ids == index]
        for block_index, start in enumerate(range(0, len(values), int(block_size))):
            block = values[start : start + int(block_size)]
            alarms = block > threshold
            rate = float(np.mean(alarms)) if len(block) else None
            rows.append({
                "cluster": cluster,
                "block_index": int(block_index),
                "start_window": int(start),
                "stop_window": int(start + len(block)),
                "window_count": int(len(block)),
                "false_alarm_count": int(np.sum(alarms)),
                "false_alarm_rate": rate,
                "false_alarms_per_10k_windows": None if rate is None else rate * 10000.0,
            })
    return rows


def _regime_rows(test_sequences, test_scores, test_sequence_ids, test_endpoints, threshold, cluster_names, current_index, window):
    rows = []
    for index, sequence in enumerate(test_sequences):
        mask_sequence = test_sequence_ids == index
        scores = test_scores[mask_sequence]
        endpoints = test_endpoints[mask_sequence]
        regimes = derive_bms_regime_labels(sequence, current_index)[endpoints]
        for regime_id in sorted(np.unique(regimes).tolist()):
            selected = scores[regimes == regime_id]
            alarms = selected > threshold
            rate = float(np.mean(alarms)) if len(selected) else None
            rows.append({
                "cluster": cluster_names[index],
                "regime_id": int(regime_id),
                "regime": BMS_REGIME_NAMES.get(int(regime_id), str(regime_id)),
                "window_size": int(window),
                "window_count": int(len(selected)),
                "false_alarm_count": int(np.sum(alarms)),
                "false_alarm_rate": rate,
                "false_alarms_per_10k_windows": None if rate is None else rate * 10000.0,
            })
    return rows


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="DyAD-style compatibility runner for normal-only BMS")
    parser.add_argument("--output_dir", default=str(EXTERNAL_RUNS_ROOT / "dyad_bms_local_seed3407"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--nll_weight", type=float, default=10.0)
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--kl_warmup_steps", type=int, default=500)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--noise_scale", type=float, default=0.01)
    parser.add_argument("--threshold_quantile", type=float, default=0.99)
    parser.add_argument("--block_size", type=int, default=1000)
    parser.add_argument("--train_window_limit", type=int, default=0, help="Smoke-test cap; 0 keeps all windows")
    parser.add_argument("--eval_window_limit", type=int, default=0, help="Smoke-test cap; 0 evaluates all windows")
    parser.add_argument("--use_cuda", type=lambda value: str(value).lower() == "true", default=True)
    parser.add_argument("--require_cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    _seed_everything(args.seed)
    device = _device(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    (train_map, _), (test_map, label_map) = get_bms_cluster_data(normalize=True)
    cluster_names = sorted(train_map)
    if any(label_map.get(name) is not None for name in cluster_names):
        raise ValueError("DyAD BMS runner expects the current normal-only BMS release")
    train_sequences, validation_sequences, test_sequences = [], [], []
    for cluster in cluster_names:
        values = np.asarray(train_map[cluster], dtype=np.float32)
        split = int(len(values) * (1.0 - float(args.val_ratio)))
        if split <= args.lookback or split >= len(values) - args.lookback:
            raise ValueError(f"BMS cluster {cluster} is too short for the configured split/window")
        train_sequences.append(values[:split])
        validation_sequences.append(values[split:])
        test_sequences.append(np.asarray(test_map[cluster], dtype=np.float32))
    train_loader, validation_loader, test_loader = _make_loaders(
        train_sequences, validation_sequences, test_sequences, args
    )
    loading_seconds = time.perf_counter() - started

    condition_indices = (
        BMS_FEATURE_NAMES.index("SYS_I"),
        BMS_FEATURE_NAMES.index("BMSnRSOC"),
    )
    model = DyADBMS(
        n_features=len(BMS_FEATURE_NAMES),
        condition_indices=condition_indices,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        noise_scale=args.noise_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(args.epochs)), eta_min=0.1 * args.learning_rate
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    global_step = 0
    for epoch in range(int(args.epochs)):
        model.train()
        total_loss = total_reconstruction = total_kl = 0.0
        batches = 0
        for values, _, _ in train_loader:
            values = values.to(device, non_blocking=True)
            response, mean, log_v = model(values)
            target = model.target(values)
            reconstruction_loss = nn.functional.smooth_l1_loss(response, target)
            kl = -0.5 * torch.sum(1.0 + log_v - mean.square() - log_v.exp()) / values.shape[0]
            kl_factor = float(args.kl_weight) * min(
                1.0, float(global_step + 1) / max(1, int(args.kl_warmup_steps))
            )
            loss = float(args.nll_weight) * reconstruction_loss + kl_factor * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            total_reconstruction += float(reconstruction_loss.detach())
            total_kl += float(kl.detach())
            batches += 1
            global_step += 1
        scheduler.step()
        print(
            f"[dyad_bms] epoch={epoch + 1}/{args.epochs} "
            f"loss={total_loss / max(1, batches):.6f} "
            f"reconstruction={total_reconstruction / max(1, batches):.6f} "
            f"kl={total_kl / max(1, batches):.6f}"
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started

    inference_started = time.perf_counter()
    validation_scores, validation_ids, validation_endpoints = _score_loader(model, validation_loader, device)
    test_scores, test_ids, test_endpoints = _score_loader(model, test_loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    threshold = float(np.quantile(validation_scores, float(args.threshold_quantile)))
    alarms = test_scores > threshold
    false_alarm_rate = float(np.mean(alarms)) if len(alarms) else None
    cluster_rows = _cluster_rows(test_scores, test_ids, threshold, cluster_names)
    block_rows = _block_rows(test_scores, test_ids, threshold, args.block_size, cluster_names)
    regime_rows = _regime_rows(
        test_sequences,
        test_scores,
        test_ids,
        test_endpoints,
        threshold,
        cluster_names,
        BMS_FEATURE_NAMES.index("BMSnI"),
        args.lookback,
    )
    cluster_rates = np.asarray([row["false_alarm_rate"] for row in cluster_rows if row["false_alarm_rate"] is not None])
    block_rates = np.asarray([row["false_alarm_rate"] for row in block_rows if row["false_alarm_rate"] is not None])
    regime_rates = np.asarray([row["false_alarm_rate"] for row in regime_rows if row["false_alarm_rate"] is not None])
    metrics = {
        "dataset": "BMS",
        "method": "dyad_style",
        "seed": int(args.seed),
        "evaluation_kind": "normal_only",
        "protocol": "BMS external normal-only compatibility: per-cluster 90/10 model/calibration, independent test",
        "protocol_warning": "DyAD has no original BMS task; this is a DyAD-style DynamicVAE compatibility adaptation",
        "input_feature_scope": "all_35",
        "condition_features": [BMS_FEATURE_NAMES[index] for index in condition_indices],
        "response_feature_count": int(model.response_count),
        "response_features": [BMS_FEATURE_NAMES[index] for index in model.response_indices],
        "window": int(args.lookback),
        "stride": int(args.stride),
        "threshold": threshold,
        "threshold_source": f"pooled_normal_validation_q{float(args.threshold_quantile):.4f}",
        "sample_count": int(len(test_scores)),
        "validation_sample_count": int(len(validation_scores)),
        "false_alarm_rate": false_alarm_rate,
        "false_alarms_per_10k_windows": None if false_alarm_rate is None else false_alarm_rate * 10000.0,
        "false_alarm_count": int(np.sum(alarms)),
        "cluster_count": len(cluster_names),
        "cluster_false_alarm_rate_std": float(np.std(cluster_rates)) if cluster_rates.size else None,
        "cluster_false_alarm_rate_range": float(np.ptp(cluster_rates)) if cluster_rates.size else None,
        "block_false_alarm_rate_std": float(np.std(block_rates)) if block_rates.size else None,
        "block_false_alarm_rate_p95": float(np.quantile(block_rates, 0.95)) if block_rates.size else None,
        "block_false_alarm_rate_max": float(np.max(block_rates)) if block_rates.size else None,
        "regime_false_alarm_rate_std": float(np.std(regime_rates)) if regime_rates.size else None,
        "regime_false_alarm_rate_range": float(np.ptp(regime_rates)) if regime_rates.size else None,
        "reported_supervised_fault_metrics": False,
    }
    runtime = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "data_loading_seconds": float(loading_seconds),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_window_count": int(len(train_loader.dataset)),
        "validation_window_count": int(len(validation_loader.dataset)),
        "test_window_count": int(len(test_loader.dataset)),
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0,
    }
    config = {
        **vars(args),
        "resolved_device": str(device),
        "clusters": cluster_names,
        "feature_names": BMS_FEATURE_NAMES,
        "condition_indices": list(condition_indices),
        "response_indices": list(model.response_indices),
        "model": "public DyAD DynamicVAE shape; BMS task adapter",
        "normalization": "get_bms_cluster_data(normalize=True), shared MaxAbsScaler over official train interval",
    }
    _write_json(output / "config.json", config)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "runtime.json", runtime)
    _write_json(output / "bms_false_alarm_by_cluster.json", cluster_rows)
    _write_json(output / "bms_false_alarm_by_block.json", block_rows)
    _write_json(output / "bms_false_alarm_by_regime.json", regime_rows)
    np.savez_compressed(
        output / "scores.npz",
        validation_scores=validation_scores,
        validation_sequence_ids=validation_ids,
        validation_endpoints=validation_endpoints,
        test_scores=test_scores,
        test_sequence_ids=test_ids,
        test_endpoints=test_endpoints,
    )
    torch.save(model.state_dict(), output / "model.pt")
    print(json.dumps({"metrics": metrics, "runtime": runtime}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
