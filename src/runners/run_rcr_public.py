"""Run the frozen RCR protocol on MSL/SMAP without crossing entity boundaries.

The public telemetry channels have no documented current/SOC controls.  Their
condition vector therefore follows C3's anonymous-context rule: four marginal
statistics are computed for every non-target channel and pooled by mean/max.
No test labels enter training, calibration, checkpoint selection, or threshold
selection.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from src.models.rcr import RCRBackbone
from src.project_paths import processed_dataset_path
from src.runners.run_rcr import atomic_json, backbone_train, state_hash
from src.runners.run_unified_external_baselines import WindowDataset


def _read(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _split(values: np.ndarray, lengths: list[int]) -> list[np.ndarray]:
    boundaries = np.cumsum([0, *map(int, lengths)])
    if boundaries[-1] != len(values):
        raise ValueError("Telemetry lengths do not match concatenated array")
    return [np.asarray(values[boundaries[i]:boundaries[i + 1]], dtype=np.float32)
            for i in range(len(lengths))]


def anonymous_context_features(x: torch.Tensor) -> torch.Tensor:
    """C3-compatible pooled descriptors from anonymous context channels 1..D-1."""
    context = x[..., 1:] if x.shape[-1] > 1 else x
    delta = context[:, -1] - context[:, 0]
    mean_abs_diff = context.diff(dim=1).abs().mean(dim=1)
    descriptors = torch.stack(
        (context.mean(dim=1), context.std(dim=1, unbiased=False),
         context.amax(dim=1) - context.amin(dim=1), mean_abs_diff), dim=-1
    )
    return torch.cat((descriptors.mean(dim=1), descriptors.amax(dim=1)), dim=1)


def load_public(dataset: str, window: int, stride: int, val_ratio: float, seed: int,
                train_limit: int = 0, eval_limit: int = 0):
    # MSL and SMAP share the project's historical ``data/processed`` bundle.
    directory = processed_dataset_path("DATA", "data")
    train = np.asarray(_read(directory / f"{dataset}_train.pkl"), dtype=np.float32)
    test = np.asarray(_read(directory / f"{dataset}_test.pkl"), dtype=np.float32)
    labels = np.asarray(_read(directory / f"{dataset}_test_label.pkl"), dtype=np.int8)
    train_lengths = list(_read(directory / f"{dataset}_train_lengths.pkl"))
    test_lengths = list(_read(directory / f"{dataset}_test_lengths.pkl"))
    scaler = MinMaxScaler().fit(train)
    train = scaler.transform(train).astype(np.float32)
    test = scaler.transform(test).astype(np.float32)
    train_sequences = _split(train, train_lengths)
    test_sequences = _split(test, test_lengths)
    label_sequences = _split(labels[:, None], test_lengths)

    fit_sequences, validation_sequences = [], []
    for sequence in train_sequences:
        cut = int(len(sequence) * (1.0 - val_ratio))
        cut = min(max(cut, window), len(sequence) - window)
        fit_sequences.append(sequence[:cut])
        validation_sequences.append(sequence[cut:])
    if any(len(x) < window for x in fit_sequences + validation_sequences + test_sequences):
        raise ValueError("At least one entity is shorter than the requested window")

    data = {
        "train": WindowDataset(fit_sequences, window, stride, limit=train_limit, seed=seed),
        "validation": WindowDataset(validation_sequences, window, stride, limit=eval_limit, seed=seed),
        "test": WindowDataset(test_sequences, window, stride, limit=eval_limit, seed=seed),
    }
    ids = {partition: [f"{dataset}_{i:02d}" for i in range(len(ds.sequences))]
           for partition, ds in data.items()}
    return data, ids, label_sequences, scaler


def cache_residuals(model, data, ids, root: Path):
    expected_hash = state_hash(model)
    for partition, dataset in data.items():
        directory = root / "residuals" / partition
        directory.mkdir(parents=True, exist_ok=True)
        done = directory / "complete.json"
        if done.exists():
            record = json.loads(done.read_text())
            if record["backbone_hash"] != expected_hash or record["windows"] != len(dataset):
                raise RuntimeError(f"Changed cache source for {partition}")
            continue
        n, d = len(dataset), model.variables
        residual = np.lib.format.open_memmap(
            directory / "residual.npy", mode="w+", dtype="float32",
            shape=(n, dataset.window, d),
        )
        condition = np.lib.format.open_memmap(
            directory / "condition.npy", mode="w+", dtype="float32", shape=(n, 8)
        )
        references = np.lib.format.open_memmap(
            directory / "references.npy", mode="w+", dtype="int64", shape=(n, 2)
        )
        offset = 0
        with torch.inference_mode():
            for x, sequence, end in DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True):
                x = x.cuda(non_blocking=True)
                stop = offset + len(x)
                residual[offset:stop] = (x - model(x)).cpu().numpy()
                condition[offset:stop] = anonymous_context_features(x).cpu().numpy()
                references[offset:stop] = np.column_stack((sequence.numpy(), end.numpy()))
                offset = stop
        residual.flush(); condition.flush(); references.flush()
        atomic_json(done, {
            "backbone_hash": expected_hash,
            "windows": n,
            "sample_ids": ids[partition],
            "reference_columns": ["sequence_id", "window_end_index"],
            "condition": "anonymous_context_mean_max_marginal_statistics",
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MSL", "SMAP"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs-backbone", type=int, default=50)
    parser.add_argument("--train-limit", type=int, default=0, help="Smoke only")
    parser.add_argument("--eval-limit", type=int, default=0, help="Smoke only")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; run outside sandbox")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.benchmark = False

    data, ids, label_sequences, scaler = load_public(
        args.dataset, args.window, args.stride, args.val_ratio, args.seed,
        args.train_limit, args.eval_limit,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        "dataset": args.dataset, "seed": args.seed, "window": args.window,
        "stride": args.stride, "val_ratio": args.val_ratio,
        "feature_graph": True, "temporal": True,
        "epochs": args.epochs_backbone, "patience": 8, "lr": 0.001,
        "weight_decay": 0.0001, "batch_size": 128,
        "train_limit": args.train_limit, "eval_limit": args.eval_limit,
        "smoke": bool(args.train_limit or args.eval_limit),
        "implementation": "rcr_v1_public_telemetry",
        "nasa_feature_scope": "all_to_target0_primary",
        "condition": "C3 anonymous context channels 1..D-1, marginal mean/max pooling",
        "selection": "normal temporal validation only",
    }
    protocol = args.output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != settings:
        raise RuntimeError("Output protocol differs; use a separate output directory")
    atomic_json(protocol, settings)
    mapping = {
        "feature_columns": [f"dim_{i}" for i in range(data["train"].sequences[0].shape[1])],
        "control_indices": list(range(1, data["train"].sequences[0].shape[1])),
        "sample_ids": ids,
        "windows": {name: len(ds) for name, ds in data.items()},
    }
    atomic_json(args.output / "mapping.json", mapping)
    np.savez(args.output / "input_normalization.npz",
             data_min=scaler.data_min_, data_max=scaler.data_max_,
             scale=scaler.scale_, offset=scaler.min_,
             feature_columns=np.asarray(mapping["feature_columns"]))
    np.savez(args.output / "test_labels_by_entity.npz",
             **{f"entity_{i:02d}": values[:, 0].astype(np.int8)
                for i, values in enumerate(label_sequences)})

    train_args = SimpleNamespace(
        no_feature_graph=False, no_temporal=False, seed=args.seed,
        epochs_backbone=args.epochs_backbone,
    )
    model = backbone_train(train_args, args.output, data, mapping["feature_columns"].__len__())
    cache_residuals(model, data, ids, args.output)
    atomic_json(args.output / "public_stage_complete.json", {
        "dataset": args.dataset, "backbone_hash": state_hash(model),
        "windows": mapping["windows"], "test_labels_used": False,
    })
    print(json.dumps({"status": "complete", "dataset": args.dataset,
                      "output": str(args.output), "windows": mapping["windows"]}))


if __name__ == "__main__":
    main()
