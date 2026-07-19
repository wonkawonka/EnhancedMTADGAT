"""汇总分析产物和实验计划，生成项目级 Markdown 报告和消融矩阵。"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path

import pandas as pd

from src.project_paths import CONFIGS_ROOT, REPORT_ROOT


TRACKED_ABLATION_KEYS = [
    "dataset",
    "model_name",
    "use_transformer",
    "use_regime_condition",
    "regime_encoder_type",
    "regime_aux_lambda",
    "regime_condition_mode",
    "score_fusion_mode",
    "use_event_consistency",
    "use_physical_state_encoding",
    "use_physical_regularization",
    "use_physical_response_score",
    "physical_response_terms",
    "nasa_train_batteries",
    "nasa_test_batteries",
]


def load_json(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_plan_inventory(plan_dir: Path, plan_type: str) -> pd.DataFrame:
    rows = []
    for plan_path in sorted(plan_dir.glob("*.json")):
        plan = load_json(plan_path)
        global_seeds = plan.get("seeds", [])
        experiment_count = 0
        for experiment in plan.get("experiments", []):
            matrix_count = math.prod(len(values) for values in experiment.get("matrix", {}).values())
            seed_count = max(1, len(experiment.get("seeds", global_seeds)))
            experiment_count += max(1, matrix_count) * seed_count
        rows.append(
            {
                "plan_type": plan_type,
                "plan_name": plan.get("plan_name", plan_path.stem),
                "file_name": plan_path.name,
                "thesis_role": plan.get("_thesis_role", plan.get("_purpose", "")),
                "experiment_count": int(experiment_count),
                "comment": plan.get("_comment", ""),
            }
        )
    return pd.DataFrame(rows)


def collect_ablation_matrix(internal_plan_dir: Path) -> pd.DataFrame:
    rows = []
    for plan_path in sorted(internal_plan_dir.glob("*.json")):
        plan = load_json(plan_path)
        plan_name = str(plan.get("plan_name", plan_path.stem))
        thesis_role = str(plan.get("_thesis_role", ""))
        contains_ablation = any(
            "消融" in str(experiment.get("comment", ""))
            for experiment in plan.get("experiments", [])
        )
        if "ablation" not in plan_name.lower() and "消融" not in thesis_role and not contains_ablation:
            continue

        common_args = dict(plan.get("common_args", {}))
        for experiment in plan.get("experiments", []):
            merged_args = dict(common_args)
            merged_args.update(experiment.get("args", {}))
            row = {
                "plan_name": plan_name,
                "experiment_name": experiment.get("name", ""),
                "comment": experiment.get("comment", ""),
            }
            for key in TRACKED_ABLATION_KEYS:
                row[key] = merged_args.get(key)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["plan_name", "experiment_name", "comment"] + TRACKED_ABLATION_KEYS)

    return pd.DataFrame(rows)


def collect_analysis_reports(analysis_root: Path) -> pd.DataFrame:
    rows = []
    if not analysis_root.exists():
        return pd.DataFrame(columns=["dataset", "summary_path", "report_path"])

    for summary_path in sorted(analysis_root.glob("*/summary.json")):
        summary = load_json(summary_path)
        rows.append(
            {
                "dataset": summary.get("dataset", summary_path.parent.name),
                "summary_path": str(summary_path),
                "report_path": str(summary_path.parent / "dataset_report.md"),
                "feature_count": summary.get("feature_count"),
                "test_point_count": summary.get("test_point_count"),
                "label_available": summary.get("event_summary", {}).get("label_available"),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["_暂无数据_"]
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.fillna("").itertuples(index=False):
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def write_project_report(output_dir: Path, inventory_df: pd.DataFrame, ablation_df: pd.DataFrame, analysis_df: pd.DataFrame):
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(output_dir / "plan_inventory.csv", index=False, encoding="utf-8-sig")
    ablation_df.to_csv(output_dir / "ablation_matrix.csv", index=False, encoding="utf-8-sig")
    analysis_df.to_csv(output_dir / "analysis_inventory.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 项目报告总览",
        "",
        "## 实验计划盘点",
        "",
        *dataframe_to_markdown(inventory_df),
        "",
        "## 消融设计矩阵",
        "",
        *dataframe_to_markdown(ablation_df),
        "",
        "## 已生成分析报告",
        "",
        *dataframe_to_markdown(analysis_df),
        "",
        "## 说明",
        "",
        "- `plan_inventory.csv` 汇总当前 internal/external 计划。",
        "- `ablation_matrix.csv` 汇总当前消融设计的模块开关矩阵。",
        "- `analysis_inventory.csv` 汇总已生成的数据分析报告。",
    ]
    report_path = output_dir / "project_overview.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build project-level report index from plans and analysis outputs.")
    parser.add_argument("--output-dir", type=str, default="", help="Directory used to save generated project reports.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve() if args.output_dir.strip() else (REPORT_ROOT / "generated").resolve()

    internal_df = collect_plan_inventory(CONFIGS_ROOT / "internal", "internal")
    external_df = collect_plan_inventory(CONFIGS_ROOT / "external", "external")
    inventory_df = pd.concat([internal_df, external_df], ignore_index=True)
    ablation_df = collect_ablation_matrix(CONFIGS_ROOT / "internal")
    analysis_df = collect_analysis_reports(REPORT_ROOT / "analysis")

    report_path = write_project_report(output_dir, inventory_df, ablation_df, analysis_df)
    print("Report build finished")
    print(f"Output dir : {output_dir}")
    print(f"Overview   : {report_path}")


if __name__ == "__main__":
    main()
