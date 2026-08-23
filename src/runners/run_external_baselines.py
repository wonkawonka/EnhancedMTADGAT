"""根据计划批量运行外部基线并记录输出。"""


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


from src.project_paths import EXTERNAL_RUNS_ROOT, PROJECT_ROOT


BASELINE_DIR_MAP = {

    "GDN": "GDN",

    "Anomaly-Transformer": "Anomaly-Transformer",

    "TranAD": "TranAD",

    "DCdetector": "DCdetector",

    "GANF": "GANF",

}


def sanitize_name(value):

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())

    return cleaned.strip("-") or "exp"


def pack_batch_output(batch_root):
    """Archive output, logs and registry as one downloadable batch."""
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


def load_plan(plan_path):

    with open(plan_path, "r", encoding="utf-8") as f:

        return json.load(f)


def expand_experiment_matrix(experiments):
    """Expand Cartesian matrices or explicit conditional argument cases."""
    expanded = []
    for experiment in experiments:
        cases = experiment.get("cases", [])
        if cases:
            if experiment.get("matrix"):
                raise ValueError(
                    f"Experiment {experiment.get('name', 'experiment')!r} cannot define both cases and matrix"
                )
            for case in cases:
                if not isinstance(case, dict) or not case:
                    raise ValueError("Each experiment case must be a non-empty object")
                item = dict(experiment)
                item.pop("cases", None)
                item["args"] = {**experiment.get("args", {}), **case}
                suffix = "_".join(
                    f"{ {'brand_fold': 'f'}.get(key, key) }{value}" for key, value in case.items()
                )
                item["name"] = f"{experiment.get('name', 'experiment')}_{suffix}"
                expanded.append(item)
            continue
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
            expanded.append(item)
    return expanded


def normalize_command_token(token, python_executable):

    token = str(token)

    if token == "{python}":

        return python_executable

    return token


def normalize_cli_value(value):

    if value is None:

        return None

    if isinstance(value, bool):

        return "true" if value else "false"

    return str(value)


def normalize_flag(flag_name):

    flag_name = str(flag_name)

    if flag_name.startswith("-"):

        return flag_name

    if len(flag_name) == 1:

        return f"-{flag_name}"

    return f"--{flag_name}"


def resolve_external_baseline_dir(project_root, baseline_name):

    if baseline_name not in BASELINE_DIR_MAP:

        available = ", ".join(sorted(BASELINE_DIR_MAP))

        raise ValueError(f"Unknown baseline '{baseline_name}'. Available baselines: {available}")

    return project_root / "external_baselines" / BASELINE_DIR_MAP[baseline_name]


def resolve_cwd(project_root, experiment):

    cwd = experiment.get("cwd")

    if cwd:

        cwd_path = Path(cwd)

        return cwd_path if cwd_path.is_absolute() else (project_root / cwd_path).resolve()


    baseline_name = experiment.get("baseline")

    if not baseline_name:

        raise ValueError(f"Experiment '{experiment.get('name', 'unknown')}' must provide baseline or cwd")

    return resolve_external_baseline_dir(project_root, baseline_name)


def resolve_marker_path(cwd, marker_value):

    marker_path = Path(marker_value)

    return marker_path if marker_path.is_absolute() else (cwd / marker_path)


def build_command(experiment, python_executable):

    raw_command = experiment.get("command")

    if raw_command:

        if not isinstance(raw_command, list):

            raise ValueError(f"Experiment '{experiment.get('name', 'unknown')}' command must be a list")

        return [normalize_command_token(token, python_executable) for token in raw_command]


    module = experiment.get("module")

    script = experiment.get("script")

    if not script and not module:

        raise ValueError(
            f"Experiment '{experiment.get('name', 'unknown')}' must provide command, module or script"
        )


    command = [python_executable, "-m", str(module)] if module else [python_executable, str(script)]

    for value in experiment.get("positional_args", []):

        command.append(str(value))


    for flag_name in experiment.get("flags", []):

        command.append(normalize_flag(flag_name))


    for key, value in experiment.get("args", {}).items():

        cli_value = normalize_cli_value(value)

        if cli_value is None:

            continue

        command.extend([normalize_flag(key), cli_value])


    return command


def stream_subprocess(command, cwd, log_path, extra_env=None):

    env = os.environ.copy()

    if extra_env:

        env.update({str(k): str(v) for k, v in extra_env.items()})


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

            print(line, end="")

            log_file.write(line)


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


def parse_args():

    parser = argparse.ArgumentParser(description="Batch runner for external baseline repositories.")

    parser.add_argument("--plan", type=str, required=True, help="Path to a JSON external baseline plan.")

    parser.add_argument(

        "--python",

        type=str,

        default=sys.executable,

        help="Python executable used when building python-based commands.",

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

        help="Skip experiments whose skip marker already exists.",

    )

    parser.add_argument(

        "--dry-run",

        action="store_true",

        help="Only print resolved commands without executing them.",

    )

    parser.add_argument(

        "--batch-tag",

        type=str,

        default="",

        help="Optional suffix; by default a stable plan directory is reused for resume/skip.",

    )

    return parser.parse_args()


def main():

    args = parse_args()

    project_root = PROJECT_ROOT

    plan_path = Path(args.plan).resolve()

    plan = load_plan(plan_path)


    common_args = dict(plan.get("common_args", {}))
    common_cwd = plan.get("common_cwd")
    common_required_outputs = plan.get("required_outputs")
    pack_experiments = bool(plan.get("pack_experiments", True))
    experiments = []
    for raw_experiment in plan.get("experiments", []):
        experiment = dict(raw_experiment)
        if common_cwd is not None:
            experiment.setdefault("cwd", common_cwd)
        if common_required_outputs is not None:
            experiment.setdefault("required_outputs", common_required_outputs)
        experiment["args"] = {**common_args, **raw_experiment.get("args", {})}
        experiments.append(experiment)
    experiments = expand_experiment_matrix(experiments)

    if not experiments:

        raise ValueError(f"No experiments found in plan: {plan_path}")


    only_names = {sanitize_name(item).lower() for item in args.only.split(",") if item.strip()}

    plan_name = sanitize_name(plan.get("plan_name", plan_path.stem))

    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    batch_root_name = plan_name if not args.batch_tag.strip() else f"{plan_name}__{sanitize_name(args.batch_tag)}"

    batch_root = EXTERNAL_RUNS_ROOT / batch_root_name

    logs_dir = batch_root / "logs"

    logs_dir.mkdir(parents=True, exist_ok=True)


    common_env = dict(plan.get("common_env", {}))

    selected_experiments = []

    for idx, experiment in enumerate(experiments, start=1):

        name = sanitize_name(experiment.get("name", f"external-{idx:02d}")).lower()

        baseline_name = sanitize_name(experiment.get("baseline", "")).lower()

        if only_names:

            if (
                name in only_names
                or baseline_name in only_names
                or any(name.startswith(f"{selected}_") for selected in only_names)
            ):

                pass

            else:

                continue

        selected_experiments.append((idx, name, experiment))


    if not selected_experiments:

        raise ValueError("No external baseline experiments selected to run.")


    registry = {

        "plan_name": plan_name,

        "plan_path": str(plan_path),

        "generated_at": batch_timestamp,

        "batch_root": str(batch_root),

        "experiments": [],

    }


    print(f"Loaded {len(selected_experiments)} external baseline experiments from {plan_path.name}")


    for idx, name, experiment in selected_experiments:

        cwd = resolve_cwd(project_root, experiment)


        # 注入 output_dir 以统一输出

        output_dir = batch_root / "output" / name

        experiment.setdefault("args", {})

        existing_keys = experiment["args"].keys()

        has_output_key = any(k.strip("-") == "output_dir" for k in existing_keys)

        if not has_output_key:

            if any(k.startswith("-") for k in existing_keys):

                experiment["args"]["-output_dir"] = str(output_dir)

            else:

                experiment["args"]["output_dir"] = str(output_dir)


        command = build_command(experiment, args.python)

        log_path = logs_dir / f"{name}.log"

        env_payload = dict(common_env)

        env_payload.update(experiment.get("env", {}))


        skip_marker_value = experiment.get("skip_if_exists", "")

        skip_marker = resolve_marker_path(cwd, skip_marker_value) if skip_marker_value else None


        # 同时检查基于 output_dir 的跳过标记（原生检查点现在写入这里）

        output_skip = output_dir / "checkpoints" if skip_marker_value else None


        record = {

            "name": name,

            "baseline": experiment.get("baseline", ""),

            "cwd": str(cwd),

            "output_dir": str(output_dir),

            "command": command,

            "log_path": str(log_path),

            "skip_marker": str(skip_marker) if skip_marker else "",

            "env": env_payload,

            "status": "pending",

            "return_code": None,

            "required_outputs": experiment.get("required_outputs", ["metrics.json"]),

        }


        if args.skip_existing:

            if all((output_dir / item).exists() for item in record["required_outputs"]):

                record["status"] = "skipped_existing"

            elif skip_marker and skip_marker.exists():

                record["status"] = "skipped_existing"

            elif output_skip and output_skip.exists():

                record["status"] = "skipped_existing"


        if record["status"] == "skipped_existing":

            registry["experiments"].append(record)

            print(f"[SKIP] {name} -> {skip_marker or output_skip}")

            continue


        print(f"\n[{idx}/{len(selected_experiments)}] {name}")

        print(f"CWD     : {cwd}")

        print("COMMAND : " + " ".join(command))


        if args.dry_run:

            record["status"] = "dry_run"

            registry["experiments"].append(record)

            continue


        return_code = stream_subprocess(command, cwd=cwd, log_path=log_path, extra_env=env_payload)

        record["return_code"] = return_code

        record["status"] = "done" if return_code == 0 else "failed"

        if return_code == 0 and pack_experiments:

            packed_output = pack_experiment_output(output_dir)

            if packed_output is not None:

                record["packed_output"] = str(packed_output)

        registry["experiments"].append(record)


        registry_path = batch_root / "run_registry.json"

        with open(registry_path, "w", encoding="utf-8") as f:

            json.dump(registry, f, indent=2, ensure_ascii=False)


        if return_code != 0:

            print(f"[FAILED] {name} -> return code {return_code}")

        else:

            print(f"[DONE] {name}")


    registry_path = batch_root / "run_registry.json"

    with open(registry_path, "w", encoding="utf-8") as f:

        json.dump(registry, f, indent=2, ensure_ascii=False)

    if not args.dry_run:
        registry["batch_archive"] = str(batch_root.parent / f"{batch_root.name}.zip")
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        pack_batch_output(batch_root)


    succeeded = sum(item["status"] == "done" for item in registry["experiments"])

    failed = sum(item["status"] == "failed" for item in registry["experiments"])

    skipped = sum(item["status"] == "skipped_existing" for item in registry["experiments"])

    dry_runs = sum(item["status"] == "dry_run" for item in registry["experiments"])


    print("\nExternal baseline batch finished")

    print(f"Registry : {registry_path}")

    print(f"Success  : {succeeded}")

    print(f"Failed   : {failed}")

    print(f"Skipped  : {skipped}")

    print(f"Dry run  : {dry_runs}")


if __name__ == "__main__":

    main()
