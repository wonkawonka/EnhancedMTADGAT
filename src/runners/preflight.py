"""Fast runtime/data checks before spending a Kaggle GPU session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import torch

from src.args import get_parser
from src.data.nc_battery import load_snippet, prepared_index_path, resolve_brand_root
from src.models.model_factory import build_model
from src.project_paths import REPORT_ROOT, processed_dataset_path, resolve_dataset_root


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
    bms_root = resolve_dataset_root("BMS", "BMS")
    bms_processed = processed_dataset_path("BMS")
    bms_train_files = sorted(bms_processed.glob("BMS_*_train.pkl"))
    brand_reports = {}
    for brand in (1, 2, 3):
        brand_root = resolve_brand_root(root, brand)
        paths = sorted(
            path
            for folder in ("data", "train", "test")
            for path in (brand_root / folder).glob("*.pkl")
        )
        if not paths:
            raise FileNotFoundError(f"No official snippets found under {brand_root}")
        shapes = [list(load_snippet(path)[0].shape) for path in paths[:max(1, args.sample_scan)]]
        brand_reports[f"brand{brand}"] = {
            "snippet_files": len(paths),
            "sample_shapes": shapes,
            "index_ready": prepared_index_path(brand).is_file(),
            "index_path": str(prepared_index_path(brand)),
        }

    model_args = get_parser().parse_args([
        "--dataset", "TSINGHUA_EV",
        "--model_name", "mtad_gat_c4_physics",
        "--lookback", "127",
        "--use_transformer", "true",
        "--use_regime_condition", "true",
        "--use_revin", "false",
        "--use_physical_state_encoding", "true",
    ])
    model = build_model(model_args, n_features=7, window_size=127, out_dim=7, target_dims=None).cuda()
    sample = torch.randn(2, 127, 7, device="cuda", requires_grad=True)
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
        "bms_raw_root": str(bms_root),
        "bms_processed_root": str(bms_processed),
        "bms_train_files": len(bms_train_files),
        "brands": brand_reports,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "forward_backward": "passed",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
