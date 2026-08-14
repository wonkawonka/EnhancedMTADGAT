"""Unified entry point for orchestrating preprocessing, experiments, analysis, and report generation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.project_paths import PROJECT_ROOT, resolve_dataset_root


def add_common_execution_args(parser):
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to run subcommands.")
    parser.add_argument("--dry-run", action="store_true", help="Only print resolved commands without executing them.")


def build_parser():
    parser = argparse.ArgumentParser(description="Unified entry for preprocess, experiment, analysis and report stages.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Run src.runners.preprocess.")
    add_common_execution_args(preprocess_parser)
    preprocess_parser.add_argument("--dataset", required=True, help="Dataset name.")
    preprocess_parser.add_argument("--group", default="", help="Optional SMD group such as 1-1.")

    internal_parser = subparsers.add_parser("internal", help="Run src.runners.compare_experiments.")
    add_common_execution_args(internal_parser)
    internal_parser.add_argument("--plan", required=True, help="Internal plan path.")
    internal_parser.add_argument("--skip-existing", action="store_true", help="Skip finished experiments.")
    internal_parser.add_argument("--resume", action="store_true", help="Resume unfinished experiments.")
    internal_parser.add_argument("--only", default="", help="Comma separated experiment names.")

    external_parser = subparsers.add_parser("external", help="Run src.runners.run_external_baselines.")
    add_common_execution_args(external_parser)
    external_parser.add_argument("--plan", required=True, help="External plan path.")
    external_parser.add_argument("--skip-existing", action="store_true", help="Skip finished experiments.")
    external_parser.add_argument("--only", default="", help="Comma separated experiment names.")

    analyze_parser = subparsers.add_parser("analyze", help="Run src.runners.analyze.")
    add_common_execution_args(analyze_parser)
    analyze_parser.add_argument("--dataset", required=True, help="Dataset name.")
    analyze_parser.add_argument("--output-dir", default="", help="Optional analysis output directory.")
    analyze_parser.add_argument("--nasa-battery-id", default="", help="Single NASA entity.")
    analyze_parser.add_argument("--nasa-train-batteries", default="", help="Comma separated NASA train entities.")
    analyze_parser.add_argument("--nasa-test-batteries", default="", help="Comma separated NASA test entities.")
    analyze_parser.add_argument(
        "--ch-battery-root",
        default=str(resolve_dataset_root("CH-BATTERY", "CH-BATTERY")),
        help="CH-BATTERY root directory.",
    )
    analyze_parser.add_argument("--ch-battery-preprocessed-dir", default="", help="Optional CH-BATTERY processed directory.")
    analyze_parser.add_argument("--ch-battery-train-ratio", type=float, default=0.8, help="CH-BATTERY train VIN ratio.")
    analyze_parser.add_argument("--seed", type=int, default=3407, help="Random seed.")

    report_parser = subparsers.add_parser("report", help="Run src.runners.build_report.")
    add_common_execution_args(report_parser)
    report_parser.add_argument("--output-dir", default="", help="Optional project report output directory.")

    preflight_parser = subparsers.add_parser("preflight", help="Check CUDA, data and model execution before training.")
    add_common_execution_args(preflight_parser)
    preflight_parser.add_argument("--tsinghua-ev-root", default=str(resolve_dataset_root("TSINGHUA-EV", "TSINGHUA_EV")))
    preflight_parser.add_argument("--output", default="")

    full_parser = subparsers.add_parser("full", help="Run multiple configured stages in order.")
    add_common_execution_args(full_parser)
    full_parser.add_argument("--preprocess-dataset", action="append", default=[], help="Repeatable preprocess dataset.")
    full_parser.add_argument("--internal-plan", action="append", default=[], help="Repeatable internal plan path.")
    full_parser.add_argument("--external-plan", action="append", default=[], help="Repeatable external plan path.")
    full_parser.add_argument("--analysis-dataset", action="append", default=[], help="Repeatable analysis dataset.")
    full_parser.add_argument("--skip-existing", action="store_true", help="Pass through to internal/external plans.")
    full_parser.add_argument("--resume", action="store_true", help="Pass through to internal plans.")
    full_parser.add_argument("--analysis-output-dir", default="", help="Optional shared analysis output directory.")
    full_parser.add_argument("--nasa-battery-id", default="", help="Single NASA entity used in analyze/full.")
    full_parser.add_argument("--nasa-train-batteries", default="", help="Comma separated NASA train entities used in analyze/full.")
    full_parser.add_argument("--nasa-test-batteries", default="", help="Comma separated NASA test entities used in analyze/full.")
    full_parser.add_argument("--seed", type=int, default=3407, help="Random seed used in analyze/full.")
    full_parser.add_argument("--report-output-dir", default="", help="Optional project report output directory.")

    return parser


def run_subcommand(command, dry_run=False):
    print(" ".join(command))
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


def handle_preprocess(args):
    command = [args.python, "-m", "src.runners.preprocess", "--dataset", args.dataset]
    if getattr(args, "group", "").strip():
        command.extend(["--group", args.group])
    return run_subcommand(command, dry_run=args.dry_run)


def handle_internal(args):
    command = [args.python, "-m", "src.runners.compare_experiments", "--plan", args.plan]
    if args.skip_existing:
        command.append("--skip-existing")
    if args.resume:
        command.append("--resume")
    if args.only.strip():
        command.extend(["--only", args.only])
    if args.dry_run:
        command.append("--dry-run")
    return run_subcommand(command, dry_run=False)


def handle_external(args):
    command = [args.python, "-m", "src.runners.run_external_baselines", "--plan", args.plan]
    if args.skip_existing:
        command.append("--skip-existing")
    if args.only.strip():
        command.extend(["--only", args.only])
    if args.dry_run:
        command.append("--dry-run")
    return run_subcommand(command, dry_run=False)


def handle_analyze(args):
    command = [args.python, "-m", "src.runners.analyze", "--dataset", args.dataset]
    if args.output_dir.strip():
        command.extend(["--output-dir", args.output_dir])
    if args.nasa_battery_id.strip():
        command.extend(["--nasa_battery_id", args.nasa_battery_id])
    if args.nasa_train_batteries.strip():
        command.extend(["--nasa_train_batteries", args.nasa_train_batteries])
    if args.nasa_test_batteries.strip():
        command.extend(["--nasa_test_batteries", args.nasa_test_batteries])
    if args.ch_battery_root.strip():
        command.extend(["--ch_battery_root", args.ch_battery_root])
    if args.ch_battery_preprocessed_dir.strip():
        command.extend(["--ch_battery_preprocessed_dir", args.ch_battery_preprocessed_dir])
    command.extend(["--ch_battery_train_ratio", str(args.ch_battery_train_ratio)])
    command.extend(["--seed", str(args.seed)])
    return run_subcommand(command, dry_run=args.dry_run)


def handle_report(args):
    command = [args.python, "-m", "src.runners.build_report"]
    if args.output_dir.strip():
        command.extend(["--output-dir", args.output_dir])
    return run_subcommand(command, dry_run=args.dry_run)


def handle_preflight(args):
    command = [args.python, "-m", "src.runners.preflight", "--tsinghua-ev-root", args.tsinghua_ev_root]
    if args.output.strip():
        command.extend(["--output", args.output])
    return run_subcommand(command, dry_run=args.dry_run)


def handle_full(args):
    stages = []
    for dataset in args.preprocess_dataset:
        stages.append([args.python, "-m", "src.runners.preprocess", "--dataset", dataset])
    for plan in args.internal_plan:
        command = [args.python, "-m", "src.runners.compare_experiments", "--plan", plan]
        if args.skip_existing:
            command.append("--skip-existing")
        if args.resume:
            command.append("--resume")
        stages.append(command)
    for plan in args.external_plan:
        command = [args.python, "-m", "src.runners.run_external_baselines", "--plan", plan]
        if args.skip_existing:
            command.append("--skip-existing")
        stages.append(command)
    for dataset in args.analysis_dataset:
        command = [args.python, "-m", "src.runners.analyze", "--dataset", dataset]
        if args.analysis_output_dir.strip():
            command.extend(["--output-dir", args.analysis_output_dir])
        if args.nasa_battery_id.strip():
            command.extend(["--nasa_battery_id", args.nasa_battery_id])
        if args.nasa_train_batteries.strip():
            command.extend(["--nasa_train_batteries", args.nasa_train_batteries])
        if args.nasa_test_batteries.strip():
            command.extend(["--nasa_test_batteries", args.nasa_test_batteries])
        command.extend(["--seed", str(args.seed)])
        stages.append(command)

    report_command = [args.python, "-m", "src.runners.build_report"]
    if args.report_output_dir.strip():
        report_command.extend(["--output-dir", args.report_output_dir])
    stages.append(report_command)

    for stage_index, command in enumerate(stages, start=1):
        print(f"[Stage {stage_index}/{len(stages)}]")
        return_code = run_subcommand(command, dry_run=args.dry_run)
        if return_code != 0:
            print(f"Stopped at stage {stage_index} with return code {return_code}")
            return return_code
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        return_code = handle_preprocess(args)
    elif args.command == "internal":
        return_code = handle_internal(args)
    elif args.command == "external":
        return_code = handle_external(args)
    elif args.command == "analyze":
        return_code = handle_analyze(args)
    elif args.command == "report":
        return_code = handle_report(args)
    elif args.command == "preflight":
        return_code = handle_preflight(args)
    elif args.command == "full":
        return_code = handle_full(args)
    else:
        parser.error(f"Unsupported command: {args.command}")
        return

    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
