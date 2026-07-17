"""Fast runtime/data checks before spending a Kaggle GPU session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import torch

from src.args import get_parser
from src.data.tsinghua_ev_utils import load_tsinghua_ev_snippets
from src.models.model_factory import build_model
from src.project_paths import REPORT_ROOT, resolve_dataset_root


def parse_args():
    parser = argparse.ArgumentParser(description="Check CUDA, Tsinghua data and model forward/backward.")
    parser.add_argument("--tsinghua-ev-root", default=str(resolve_dataset_root("TSINGHUA-EV", "TSINGHUA_EV")))
    parser.add_argument("--output", default=str(REPORT_ROOT / "generated" / "preflight.json"))
    parser.add_argument("--sample-scan", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Enable a Kaggle GPU accelerator before training.")

    root = Path(args.tsinghua_ev_root)
    train_zip = root / "Train.zip"
    if not train_zip.exists() and not any(root.rglob("*.pkl")):
        raise FileNotFoundError(f"Tsinghua labeled data not found under {root}")

    archive_count = None
    if train_zip.exists():
        with ZipFile(train_zip) as archive:
            archive_count = sum(name.lower().endswith(".pkl") for name in archive.namelist())

    records = load_tsinghua_ev_snippets(root)
    scanned = records[:max(1, args.sample_scan)]
    label_counts = {
        "normal": sum(record["label"] == 0 for record in records),
        "abnormal": sum(record["label"] == 1 for record in records),
    }

    model_args = get_parser().parse_args([
        "--dataset", "TSINGHUA_EV",
        "--model_name", "mtad_gat_c4_physics",
        "--lookback", "64",
        "--use_transformer", "true",
        "--use_regime_condition", "true",
        "--use_revin", "false",
        "--use_physical_state_encoding", "true",
    ])
    model = build_model(model_args, n_features=7, window_size=64, out_dim=7, target_dims=None).cuda()
    sample = torch.randn(4, 64, 7, device="cuda", requires_grad=True)
    with torch.amp.autocast("cuda"):
        prediction, reconstruction = model(sample)
        loss = prediction.square().mean() + reconstruction.square().mean()
    loss.backward()

    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2),
        "tsinghua_root": str(root),
        "train_zip_entries": archive_count,
        "loaded_labeled_snippets": len(records),
        "label_counts": label_counts,
        "sample_shapes": [list(record["data"].shape) for record in scanned],
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "forward_backward": "passed",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
