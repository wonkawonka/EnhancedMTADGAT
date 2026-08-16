#!/usr/bin/env python3
"""Continue evaluation for the downloaded SWaT/WADI models without training."""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOAD = PROJECT_ROOT / "runs/kaggle_downloads/swat_wadi_v8"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD)
    parser.add_argument("--datasets", nargs="+", default=["WADI"], choices=["SWAT", "WADI"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def recover_config(log_path):
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("{") and line.rstrip().endswith("}"):
                config = ast.literal_eval(line.strip())
                config["use_cuda"] = True
                config["require_cuda"] = True
                config["predict_num_workers"] = 0
                config["predict_pin_memory"] = True
                return config
    raise RuntimeError(f"Could not recover model configuration from {log_path}")


def main():
    args = parse_args()
    extracted = args.download_root / "extracted"
    source_output = extracted / "output"
    source_logs = extracted / "logs"
    datasets_root = args.download_root / "EnhancedMTADGAT/datasets"
    evaluation_root = args.download_root / "evaluation"
    runs_root = evaluation_root / "runs"
    log_root = evaluation_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    registry = json.loads((extracted / "run_registry.json").read_text(encoding="utf-8"))
    selected = [exp for exp in registry["experiments"] if exp["dataset"] in args.datasets]
    failures = []
    for experiment in selected:
        name = experiment["name"]
        dataset = experiment["dataset"]
        model_dir = runs_root / "manual" / dataset / name
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_output / name / "model.pt", model_dir / "model.pt")
        config = recover_config(source_logs / f"{name}.log")
        (model_dir / "config.txt").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        summary_path = model_dir / "summary_metrics.json"
        if summary_path.exists() and not args.force:
            print(f"[skip] {name}: {summary_path} already exists", flush=True)
            continue

        command = [
            sys.executable,
            "-m",
            "src.runners.predict",
            "--dataset",
            dataset,
            "--model_id",
            name,
            "--use_cuda",
            "true",
            "--require_cuda",
            "true",
            "--predict_num_workers",
            "0",
            "--predict_pin_memory",
            "true",
            "--save_output",
            "true",
        ]
        env = os.environ.copy()
        env["MTAD_GAT_DATASETS_ROOT"] = str(datasets_root)
        env["MTAD_GAT_RUNS_ROOT"] = str(runs_root)
        log_path = log_root / f"{name}.log"
        print(f"[evaluate] {name}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            failures.append(name)
            print(f"[failed] {name}; see {log_path}", flush=True)
        else:
            print(f"[complete] {name}: {summary_path}", flush=True)

    if failures:
        raise SystemExit(f"Evaluation failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
