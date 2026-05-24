import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import compare_experiments as internal_runner
import run_external_baselines as external_runner


INTERNAL_PLANS = [
    "ch3_main_results.json",
    "ch3_ablation.json",
    "ch4_bms_main.json",
    "ch4_bms_ablation.json",
]

EXTERNAL_PLANS = [
    "ch3_external_baselines.json",
]


def stream_subprocess(command, cwd, log_path, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})

    start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)

        return_code = process.wait()

    duration_sec = time.perf_counter() - start
    return return_code, duration_sec


def summarize_records(records):
    summary = {
        "overall": {},
        "by_plan": [],
        "by_dataset": [],
        "by_model": [],
        "by_family": [],
    }

    if not records:
        return summary

    def build_group(items, key):
        groups = defaultdict(list)
        for item in items:
            groups[item[key]].append(item)
        rows = []
        for name, group_items in sorted(groups.items(), key=lambda kv: kv[0]):
            durations = [x["duration_sec"] for x in group_items]
            success_count = sum(x["status"] == "done" for x in group_items)
            rows.append(
                {
                    key: name,
                    "count": len(group_items),
                    "success_count": success_count,
                    "avg_sec": round(sum(durations) / len(durations), 2),
                    "min_sec": round(min(durations), 2),
                    "max_sec": round(max(durations), 2),
                }
            )
        return rows

    all_durations = [x["duration_sec"] for x in records]
    summary["overall"] = {
        "count": len(records),
        "success_count": sum(x["status"] == "done" for x in records),
        "avg_sec": round(sum(all_durations) / len(all_durations), 2),
        "min_sec": round(min(all_durations), 2),
        "max_sec": round(max(all_durations), 2),
    }
    summary["by_plan"] = build_group(records, "plan_name")
    summary["by_dataset"] = build_group(records, "dataset")
    summary["by_model"] = build_group(records, "model")
    summary["by_family"] = build_group(records, "family")
    return summary


def write_registry(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_internal_plan(project_root, plan_path, batch_root, python_exec, registry):
    plan = internal_runner.load_plan(plan_path)
    common_args = dict(plan.get("common_args", {}))
    plan_name = internal_runner.sanitize_name(plan.get("plan_name", plan_path.stem))
    logs_dir = batch_root / "logs" / plan_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    for idx, experiment in enumerate(plan.get("experiments", []), start=1):
        name = internal_runner.sanitize_name(experiment.get("name", f"exp-{idx:02d}"))
        merged_args = dict(common_args)
        merged_args.update(experiment.get("args", {}))
        merged_args["epochs"] = 1

        run_id = merged_args.get("run_id", name)
        merged_args["run_id"] = f"{run_id}_speed1e"
        dataset = str(merged_args.get("dataset", "UNKNOWN")).upper()
        model_name = str(merged_args.get("model_name", "internal"))

        output_dir = batch_root / "output" / plan_name / name
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{name}.log"
        command = internal_runner.build_train_command(project_root, python_exec, merged_args)

        print(f"\n[INTERNAL {idx}/{len(plan.get('experiments', []))}] {plan_name} :: {name}")
        return_code, duration_sec = stream_subprocess(
            command,
            cwd=project_root,
            log_path=log_path,
            extra_env={
                "PLAN_OUTPUT_DIR": str(output_dir),
                "DISABLE_TORCH_COMPILE": "1",
                "SPEED_BENCHMARK_TRAIN_ONLY": "1",
            },
        )

        record = {
            "plan_name": plan_name,
            "experiment_name": name,
            "family": "internal",
            "dataset": dataset,
            "model": model_name,
            "epochs": 1,
            "duration_sec": round(duration_sec, 2),
            "status": "done" if return_code == 0 else "failed",
            "return_code": return_code,
            "log_path": str(log_path),
            "output_dir": str(output_dir),
            "command": command,
        }
        registry["records"].append(record)
        registry["summary"] = summarize_records(registry["records"])
        write_registry(batch_root / "speed_registry.json", registry)


def force_external_one_epoch(experiment):
    baseline = str(experiment.get("baseline", ""))
    args = experiment.setdefault("args", {})

    if baseline == "TranAD":
        args["num_epochs"] = 1
    if baseline in {"Anomaly-Transformer", "DCdetector"}:
        args["num_epochs"] = 1
    if baseline == "GDN":
        args["-epoch"] = 1
    if baseline == "GANF":
        if "epochs" in args:
            args["epochs"] = 1
        if "--epochs" in args:
            args["--epochs"] = 1

    if "num_epochs" in args:
        args["num_epochs"] = 1
    if "epoch" in args:
        args["epoch"] = 1
    if "-epoch" in args:
        args["-epoch"] = 1


def run_external_plan(project_root, plan_path, batch_root, python_exec, registry):
    plan = external_runner.load_plan(plan_path)
    plan_name = external_runner.sanitize_name(plan.get("plan_name", plan_path.stem))
    logs_dir = batch_root / "logs" / plan_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    common_env = dict(plan.get("common_env", {}))

    experiments = list(plan.get("experiments", []))
    for idx, raw_experiment in enumerate(experiments, start=1):
        experiment = deepcopy(raw_experiment)
        name = external_runner.sanitize_name(experiment.get("name", f"external-{idx:02d}")).lower()
        force_external_one_epoch(experiment)

        cwd = external_runner.resolve_cwd(project_root, experiment)
        output_dir = batch_root / "output" / plan_name / name
        output_dir.mkdir(parents=True, exist_ok=True)

        args = experiment.setdefault("args", {})
        existing_keys = list(args.keys())
        has_output_key = any(str(k).strip("-") == "output_dir" for k in existing_keys)
        if not has_output_key:
            if any(str(k).startswith("-") for k in existing_keys):
                args["-output_dir"] = str(output_dir)
            else:
                args["output_dir"] = str(output_dir)

        command = external_runner.build_command(experiment, python_exec)
        log_path = logs_dir / f"{name}.log"
        env_payload = dict(common_env)
        env_payload.update(experiment.get("env", {}))

        raw_dataset = args.get("dataset", args.get("-dataset", "UNKNOWN"))
        dataset = str(raw_dataset).upper()
        model_name = str(experiment.get("baseline", "external"))

        print(f"\n[EXTERNAL {idx}/{len(experiments)}] {plan_name} :: {name}")
        return_code, duration_sec = stream_subprocess(
            command,
            cwd=cwd,
            log_path=log_path,
            extra_env=env_payload,
        )

        record = {
            "plan_name": plan_name,
            "experiment_name": name,
            "family": "external",
            "dataset": dataset,
            "model": model_name,
            "epochs": 1,
            "duration_sec": round(duration_sec, 2),
            "status": "done" if return_code == 0 else "failed",
            "return_code": return_code,
            "log_path": str(log_path),
            "output_dir": str(output_dir),
            "command": command,
        }
        registry["records"].append(record)
        registry["summary"] = summarize_records(registry["records"])
        write_registry(batch_root / "speed_registry.json", registry)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure real 1-epoch speed for all compare plans.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for all runs.")
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Also run external baseline plans.",
    )
    parser.add_argument(
        "--skip-internal",
        action="store_true",
        help="Skip internal plans and only run selected external plans.",
    )
    parser.add_argument(
        "--external-scope",
        type=str,
        choices=["all", "ch3", "ch4"],
        default="all",
        help="Which external plans to run when external benchmarking is enabled.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_root = project_root / "configs" / "compare"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = project_root / "experiment_runs" / f"epoch_speed_benchmark_{timestamp}"
    batch_root.mkdir(parents=True, exist_ok=True)

    registry = {
        "generated_at": timestamp,
        "batch_root": str(batch_root),
        "python": args.python,
        "include_external": args.include_external,
        "skip_internal": args.skip_internal,
        "external_scope": args.external_scope,
        "records": [],
        "summary": {},
    }
    write_registry(batch_root / "speed_registry.json", registry)

    if not args.skip_internal:
        for file_name in INTERNAL_PLANS:
            run_internal_plan(project_root, config_root / file_name, batch_root, args.python, registry)

    if args.include_external:
        if args.external_scope == "ch3":
            selected_external_plans = ["ch3_external_baselines.json"]
        elif args.external_scope == "ch4":
            selected_external_plans = ["ch4_bms_external_baselines.json"]
        else:
            selected_external_plans = EXTERNAL_PLANS

        for file_name in selected_external_plans:
            run_external_plan(project_root, config_root / file_name, batch_root, args.python, registry)

    registry["summary"] = summarize_records(registry["records"])
    write_registry(batch_root / "speed_registry.json", registry)

    print("\nSpeed benchmark finished")
    print(f"Registry: {batch_root / 'speed_registry.json'}")
    print(json.dumps(registry["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
