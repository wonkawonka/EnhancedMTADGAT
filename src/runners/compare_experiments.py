"""根据 `configs/internal` 计划批量运行内部实验。"""


import argparse

import itertools

import json

import os

import re

import shutil

import subprocess

import sys

from datetime import datetime

from pathlib import Path


from src.project_paths import INTERNAL_RUNS_ROOT, PROJECT_ROOT


def sanitize_name(value):

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())

    return cleaned.strip("-") or "exp"


def load_plan(plan_path):

    with open(plan_path, "r", encoding="utf-8") as f:

        return json.load(f)


def bool_to_cli(value):

    return "true" if value else "false"


def normalize_cli_value(value):

    if isinstance(value, bool):

        return bool_to_cli(value)

    if value is None:

        return None

    if isinstance(value, (int, float)):

        return str(value)

    return str(value)


def resolve_plan_output_dir(batch_root, experiment_name):

    """当前批次下单个实验的输出目录。"""

    return batch_root / "output" / experiment_name


def resolve_checkpoint_path(output_dir):

    return output_dir / "last_checkpoint.pt"


def expand_experiment_seeds(experiments, seeds):

    """Expand global seeds, while allowing costly main runs to opt in locally."""

    expanded = []

    for experiment in experiments:

        experiment_seeds = experiment.get("seeds", seeds)

        if not experiment_seeds:

            item = dict(experiment)

            item.pop("seeds", None)

            expanded.append(item)

            continue

        for seed in experiment_seeds:

            item = dict(experiment)

            item.pop("seeds", None)

            item["args"] = dict(experiment.get("args", {}))

            item["args"]["seed"] = int(seed)

            base_name = str(experiment.get("name", "experiment"))

            item["name"] = f"{base_name}_seed{seed}"

            if item["args"].get("run_id"):

                item["args"]["run_id"] = f"{item['args']['run_id']}_seed{seed}"

            expanded.append(item)

    return expanded


def expand_experiment_matrix(experiments):

    """Expand compact Cartesian experiment matrices (for example brand x fold)."""

    expanded = []

    for experiment in experiments:

        matrix = experiment.get("matrix", {})

        if not matrix:

            expanded.append(experiment)

            continue

        keys = list(matrix)

        for values in itertools.product(*(matrix[key] for key in keys)):

            item = dict(experiment)

            item.pop("matrix", None)

            item["args"] = dict(experiment.get("args", {}))

            suffix = []

            for key, value in zip(keys, values):

                item["args"][key] = value

                short_key = {"battery_brand": "b", "battery_fold": "f"}.get(key, key)

                suffix.append(f"{short_key}{value}")

            item["name"] = f"{experiment.get('name', 'experiment')}_{'_'.join(suffix)}"

            if item["args"].get("run_id"):

                item["args"]["run_id"] = f"{item['args']['run_id']}_{'_'.join(suffix)}"

            expanded.append(item)

    return expanded


def build_train_command(project_root, python_executable, merged_args, runner="src.runners.train"):

    cmd = [python_executable, "-m", runner]

    for key, value in merged_args.items():

        if value is None:

            continue

        cli_value = normalize_cli_value(value)

        if cli_value is None:

            continue

        cmd.extend([f"--{key}", cli_value])

    return cmd


def stream_subprocess(command, cwd, log_path, extra_env=None):

    env = os.environ.copy()

    if extra_env:

        env.update(extra_env)

    with open(log_path, "w", encoding="utf-8") as log_file:

        process = subprocess.Popen(

            command,

            cwd=str(cwd),

            env=env,

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            encoding="utf-8",

            errors="replace",

        )


        assert process.stdout is not None

        for line in process.stdout:

            log_file.write(line)

            try:

                print(line, end="")

            except UnicodeEncodeError:

                print(line.encode("utf-8", errors="replace").decode("utf-8", errors="replace"), end="")


        return process.wait()


def pack_experiment_output(output_dir):

    output_dir = Path(output_dir)

    if not output_dir.exists():

        return None

    zip_path = output_dir.parent / f"{output_dir.name}.zip"

    if zip_path.exists():

        zip_path.unlink()

    shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(output_dir))

    size_mb = zip_path.stat().st_size / (1024 * 1024)

    print(f"[PACK] {zip_path.name} ({size_mb:.1f} MB)")

    return zip_path


def pack_batch_output(batch_root):
    """Archive the complete batch, including logs and run_registry.json."""
    batch_root = Path(batch_root)
    zip_path = batch_root.parent / f"{batch_root.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(
        str(zip_path.with_suffix("")),
        "zip",
        root_dir=str(batch_root.parent),
        base_dir=batch_root.name,
    )
    print(f"[PACK BATCH] {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.1f} MB)")
    return zip_path


def parse_args():

    parser = argparse.ArgumentParser(description="Batch runner for comparison experiments.")

    parser.add_argument(

        "--plan",

        type=str,

        required=True,

        help="Path to a JSON experiment plan.",

    )

    parser.add_argument(

        "--python",

        type=str,

        default=sys.executable,

        help="Python executable used to run src.runners.train.",

    )

    parser.add_argument(

        "--only",

        type=str,

        default="",

        help="Comma separated experiment names to run.",

    )

    parser.add_argument(

        "--skip-existing",

        action="store_true",

        help="Skip experiments whose output directory already contains model.pt.",

    )

    parser.add_argument(

        "--dry-run",

        action="store_true",

        help="Only print resolved commands without executing training.",

    )

    parser.add_argument(

        "--resume",

        action="store_true",

        help="Resume unfinished experiments from last_checkpoint.pt when available.",

    )

    parser.add_argument(

        "--batch-tag",

        type=str,

        default="",

        help="Optional suffix for runs/internal/<plan_name>__<batch_tag>. Leave empty to reuse a stable plan root.",

    )

    return parser.parse_args()


def main():

    args = parse_args()

    project_root = PROJECT_ROOT

    plan_path = Path(args.plan).resolve()

    plan = load_plan(plan_path)


    common_args = dict(plan.get("common_args", {}))

    experiments = expand_experiment_matrix(plan.get("experiments", []))

    experiments = expand_experiment_seeds(experiments, plan.get("seeds", []))

    if not experiments:

        raise ValueError(f"No experiments found in plan: {plan_path}")


    only_names = {

        sanitize_name(item) for item in args.only.split(",") if item.strip()

    }


    plan_name = sanitize_name(plan.get("plan_name", plan_path.stem))

    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    batch_root_name = plan_name if not args.batch_tag.strip() else f"{plan_name}__{sanitize_name(args.batch_tag)}"

    batch_root = INTERNAL_RUNS_ROOT / batch_root_name

    logs_dir = batch_root / "logs"

    logs_dir.mkdir(parents=True, exist_ok=True)


    resolved_registry = {

        "plan_path": str(plan_path),

        "plan_name": plan_name,

        "batch_root": str(batch_root),

        "batch_root_name": batch_root_name,

        "generated_at": batch_timestamp,

        "common_args": common_args,

        "seeds": plan.get("seeds", []),

        "experiments": [],

    }


    selected_experiments = []

    for idx, experiment in enumerate(experiments, start=1):

        name = sanitize_name(experiment.get("name", f"exp-{idx:02d}"))

        if only_names and name not in only_names:

            continue

        selected_experiments.append((idx, name, experiment))


    if not selected_experiments:

        raise ValueError("No experiments selected to run.")


    print(f"Loaded {len(selected_experiments)} experiments from {plan_path.name}")


    for run_position, (idx, name, experiment) in enumerate(selected_experiments, start=1):

        merged_args = dict(common_args)

        merged_args.update(experiment.get("args", {}))

        dataset = merged_args.get("dataset")

        if not dataset:

            raise ValueError(f"Experiment '{name}' is missing required arg 'dataset'")


        run_id = merged_args.get("run_id")

        if not run_id:

            run_id = sanitize_name(f"{idx:02d}-{name}-{batch_timestamp}")

            merged_args["run_id"] = run_id


        # 输出到 runs/internal/<batch>/output/<name>/ 下

        output_dir = resolve_plan_output_dir(batch_root, name)

        plan_extra_env = {"PLAN_OUTPUT_DIR": str(output_dir)}

        checkpoint_path = resolve_checkpoint_path(output_dir)

        if args.resume:

            merged_args["resume"] = True

        command = build_train_command(
            project_root,
            args.python,
            merged_args,
            runner=experiment.get("runner", plan.get("runner", "src.runners.train")),
        )

        log_path = logs_dir / f"{name}.log"

        result = {

            "name": name,

            "dataset": str(dataset).upper(),

            "run_id": run_id,

            "output_dir": str(output_dir),

            "checkpoint_path": str(checkpoint_path),

            "log_path": str(log_path),

            "args": merged_args,

            "command": command,

            "status": "pending",

            "return_code": None,

        }


        model_path = output_dir / "model.pt"

        skip_ready = model_path.exists()
        if args.skip_existing and skip_ready and str(dataset).upper() == "BMS":
            summary_path = output_dir / "bms_operational_summary.json"
            summary = {}
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    summary = {}
            if "false_alarm_rate" not in summary:
                cluster_names = {
                    path.parent.name: None
                    for path in output_dir.glob("*/test_output.pkl")
                }
                if cluster_names:
                    from src.analysis.regime_evaluation import save_bms_operational_report

                    save_bms_operational_report(
                        output_dir,
                        cluster_names,
                        int(merged_args.get("lookback", 100)),
                    )
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
            skip_ready = "false_alarm_rate" in summary

        if args.skip_existing and skip_ready:

            result["status"] = "skipped_existing"

            resolved_registry["experiments"].append(result)

            print(f"[SKIP] {name} -> {output_dir}")

            continue


        print(f"\n[{run_position}/{len(selected_experiments)}] {name}")

        if args.resume and checkpoint_path.exists():

            print(f"[RESUME] {name} -> {checkpoint_path}")

        print(" ".join(command))


        if args.dry_run:

            result["status"] = "dry_run"

            resolved_registry["experiments"].append(result)

            continue


        return_code = stream_subprocess(command, cwd=project_root, log_path=log_path, extra_env=plan_extra_env)

        result["return_code"] = return_code

        result["status"] = "done" if return_code == 0 else "failed"

        resolved_registry["experiments"].append(result)


        pack_experiment_output(output_dir)


        registry_path = batch_root / "run_registry.json"

        with open(registry_path, "w", encoding="utf-8") as f:

            json.dump(resolved_registry, f, indent=2, ensure_ascii=False)


        if return_code != 0:

            print(f"[FAILED] {name} -> return code {return_code}")

        else:

            print(f"[DONE] {name} -> {output_dir}")


    registry_path = batch_root / "run_registry.json"

    with open(registry_path, "w", encoding="utf-8") as f:

        json.dump(resolved_registry, f, indent=2, ensure_ascii=False)

    succeeded = sum(item["status"] == "done" for item in resolved_registry["experiments"])

    failed = sum(item["status"] == "failed" for item in resolved_registry["experiments"])

    skipped = sum(item["status"] == "skipped_existing" for item in resolved_registry["experiments"])

    dry_runs = sum(item["status"] == "dry_run" for item in resolved_registry["experiments"])

    bms_comparison = None
    if not args.dry_run:
        from src.analysis.regime_evaluation import save_bms_conditioning_comparison

        bms_comparison = save_bms_conditioning_comparison(batch_root)
        resolved_registry["batch_archive"] = str(batch_root.parent / f"{batch_root.name}.zip")
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(resolved_registry, f, indent=2, ensure_ascii=False)
        pack_batch_output(batch_root)

    print("\nBatch finished")

    print(f"Registry : {registry_path}")

    print(f"Success  : {succeeded}")

    print(f"Failed   : {failed}")

    print(f"Skipped  : {skipped}")

    print(f"Dry run  : {dry_runs}")

    if bms_comparison is not None:
        print(f"BMS pair : {batch_root / 'bms_conditioning_comparison.json'}")


if __name__ == "__main__":

    main()
