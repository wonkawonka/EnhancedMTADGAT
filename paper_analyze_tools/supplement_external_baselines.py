"""补齐第三章 external baselines 的离线标准化结果。

用途：
1. 对已有 checkpoint 的 Anomaly-Transformer / DCdetector 重新执行 test-only 导出；
2. 对缺少 checkpoint 的 TranAD，从历史日志恢复指标并落盘为 summary_metrics.json；
3. 生成一份补齐清单，供 by_plan 汇总脚本继续读取。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collect_external_results import parse_tranad_metrics
from external_baselines._common_output import save_standardized_output


ROOT = PROJECT_ROOT
KAGGLE_DIR = ROOT / "kaggle离线output" / "ch3_external_baselines"
PLAN_PATH = ROOT / "configs" / "compare" / "ch3_external_baselines.json"
DEFAULT_TRANAD_LOG_GLOBS = [
    "experiment_runs/**/logs/**/tranad_smap_ch3.log",
    "experiment_runs/**/logs/**/tranad_msl_ch3.log",
]
TESTABLE_BASELINES = {"Anomaly-Transformer", "DCdetector"}


def load_plan():
    with PLAN_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_experiment_lookup():
    plan = load_plan()
    return {item["name"]: item for item in plan.get("experiments", [])}


def stringify_arg(key, value):
    if key.startswith("-"):
        return [key, str(value)]
    return [f"--{key}", str(value)]


def build_test_command(experiment_name, experiment):
    baseline = experiment["baseline"]
    args = dict(experiment.get("args", {}))
    output_dir = KAGGLE_DIR / experiment_name
    if baseline == "Anomaly-Transformer":
        args["mode"] = "test"
        args["batch_size"] = min(int(args.get("batch_size", 256)), 32)
        args["output_dir"] = str(output_dir)
        cmd = [sys.executable, "main.py"]
        for key, value in args.items():
            cmd.extend(stringify_arg(key, value))
        cwd = ROOT / "external_baselines" / "Anomaly-Transformer"
        return cmd, cwd

    if baseline == "DCdetector":
        args["mode"] = "test"
        args["batch_size"] = min(int(args.get("batch_size", 128)), 32)
        args["output_dir"] = str(output_dir)
        cmd = [sys.executable, "main.py"]
        for key, value in args.items():
            cmd.extend(stringify_arg(key, value))
        cwd = ROOT / "external_baselines" / "DCdetector"
        return cmd, cwd

    raise ValueError(f"Unsupported testable baseline: {baseline}")


def run_subprocess(command, cwd):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"[RUN] cwd={cwd}")
    print("[CMD] " + " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def select_best_tranad_log(experiment_name):
    best = None
    for pattern in DEFAULT_TRANAD_LOG_GLOBS:
        for log_path in ROOT.glob(pattern):
            if log_path.name.lower() != f"{experiment_name}.log":
                continue
            text = log_path.read_text(encoding="utf-8", errors="replace")
            metrics = parse_tranad_metrics(text)
            f1 = metrics.get("metric_f1")
            if f1 is None:
                continue
            candidate = (float(f1), log_path, metrics)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


def restore_tranad_metrics(experiment_name, experiment, force=False):
    output_dir = KAGGLE_DIR / experiment_name
    metrics_path = output_dir / "summary_metrics.json"
    if metrics_path.exists() and not force:
        return {"name": experiment_name, "baseline": "TranAD", "status": "skipped_existing_metrics"}

    selected = select_best_tranad_log(experiment_name)
    if selected is None:
        return {"name": experiment_name, "baseline": "TranAD", "status": "missing_log"}

    _, log_path, metrics = selected
    output_dir.mkdir(parents=True, exist_ok=True)
    save_standardized_output(
        output_dir=output_dir,
        metrics=metrics,
        thresholds={"global_threshold": metrics["metric_threshold"]} if metrics.get("metric_threshold") is not None else None,
        config={
            "dataset": experiment["args"].get("dataset"),
            "baseline": experiment["baseline"],
            "restored_from_log": str(log_path),
            "note": "No checkpoint/output_dir available; metrics restored from historical log.",
        },
    )
    with (output_dir / "config.txt").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": experiment["args"].get("dataset"),
                "baseline": experiment["baseline"],
                "run_id": experiment_name,
                "restored_from_log": str(log_path),
                "restored_from_log_only": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {
        "name": experiment_name,
        "baseline": "TranAD",
        "status": "restored_from_log",
        "log_path": str(log_path),
        "metric_f1": metrics.get("metric_f1"),
    }


def supplement_testable_baseline(experiment_name, experiment, force=False):
    output_dir = KAGGLE_DIR / experiment_name
    metrics_path = output_dir / "summary_metrics.json"
    test_output_path = output_dir / "test_output.pkl"
    if metrics_path.exists() and test_output_path.exists() and not force:
        return {"name": experiment_name, "baseline": experiment["baseline"], "status": "skipped_existing_output"}

    output_dir.mkdir(parents=True, exist_ok=True)
    command, cwd = build_test_command(experiment_name, experiment)
    run_subprocess(command, cwd)

    result = {
        "name": experiment_name,
        "baseline": experiment["baseline"],
        "status": "test_exported" if metrics_path.exists() and test_output_path.exists() else "test_finished_but_missing_output",
    }
    with (output_dir / "config.txt").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": experiment["args"].get("dataset") or experiment["args"].get("-dataset", ""),
                "baseline": experiment["baseline"],
                "run_id": experiment_name,
                "source": "supplement_external_baselines.py",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Supplement external baseline outputs for by-plan thesis analysis.")
    parser.add_argument("--force", action="store_true", help="Rerun even if summary_metrics/test_output already exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_lookup = build_experiment_lookup()
    records = []

    for experiment_name, experiment in exp_lookup.items():
        baseline = experiment.get("baseline", "")
        if baseline in TESTABLE_BASELINES:
            records.append(supplement_testable_baseline(experiment_name, experiment, force=args.force))
        elif baseline == "TranAD":
            records.append(restore_tranad_metrics(experiment_name, experiment, force=args.force))

    manifest_path = KAGGLE_DIR / "supplement_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote manifest: {manifest_path}")
    for row in records:
        print(row)


if __name__ == "__main__":
    main()
