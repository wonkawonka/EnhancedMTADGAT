import argparse
import csv
import json
import re
from pathlib import Path

LABELED_BINARY_DATASETS = {"SMAP", "MSL"}
CLASSIFICATION_METRIC_KEYS = {
    "metric_accuracy",
    "metric_f1",
    "metric_best_f1",
    "metric_pot_f1",
    "metric_precision",
    "metric_recall",
    "metric_pot_precision",
    "metric_pot_recall",
    "metric_auc",
    "metric_tp",
    "metric_tn",
    "metric_fp",
    "metric_fn",
}


def sanitize_name(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return cleaned.strip("-") or "exp"


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_plan_lookup(plan_path):
    plan = load_json(plan_path)
    experiments = plan.get("experiments", [])
    return {
        sanitize_name(item.get("name", "")): item
        for item in experiments
    }


def read_text_if_exists(file_path):
    path = Path(file_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def find_last_float(pattern, text):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    return float(matches[-1])


def find_last_int(pattern, text):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    return int(matches[-1])


def parse_tranad_metrics(log_text):
    return {
        "metric_f1": find_last_float(r"'f1':\s*([-+0-9.eE]+)", log_text),
        "metric_precision": find_last_float(r"'precision':\s*([-+0-9.eE]+)", log_text),
        "metric_recall": find_last_float(r"'recall':\s*([-+0-9.eE]+)", log_text),
        "metric_auc": find_last_float(r"'ROC/AUC':\s*([-+0-9.eE]+)", log_text),
        "metric_threshold": find_last_float(r"'threshold':\s*([-+0-9.eE]+)", log_text),
        "metric_tp": find_last_int(r"'TP':\s*([0-9]+)", log_text),
        "metric_tn": find_last_int(r"'TN':\s*([0-9]+)", log_text),
        "metric_fp": find_last_int(r"'FP':\s*([0-9]+)", log_text),
        "metric_fn": find_last_int(r"'FN':\s*([0-9]+)", log_text),
    }


def parse_anomaly_transformer_metrics(log_text):
    result = {
        "metric_threshold": find_last_float(r"Threshold\s*:\s*([-+0-9.eE]+)", log_text),
    }
    summary_matches = re.findall(
        r"Accuracy\s*:\s*([-+0-9.eE]+),\s*Precision\s*:\s*([-+0-9.eE]+),\s*Recall\s*:\s*([-+0-9.eE]+),\s*F-score\s*:\s*([-+0-9.eE]+)",
        log_text,
        flags=re.MULTILINE,
    )
    if summary_matches:
        accuracy, precision, recall, f1 = summary_matches[-1]
        result.update({
            "metric_accuracy": float(accuracy),
            "metric_precision": float(precision),
            "metric_recall": float(recall),
            "metric_f1": float(f1),
        })
    else:
        result.update({
            "metric_accuracy": None,
            "metric_precision": None,
            "metric_recall": None,
            "metric_f1": None,
        })
    return result


def parse_omnianomaly_metrics(log_text):
    return {
        "metric_best_f1": find_last_float(r"'best-f1':\s*([-+0-9.eE]+)", log_text),
        "metric_precision": find_last_float(r"'precision':\s*([-+0-9.eE]+)", log_text),
        "metric_recall": find_last_float(r"'recall':\s*([-+0-9.eE]+)", log_text),
        "metric_threshold": find_last_float(r"'threshold':\s*([-+0-9.eE]+)", log_text),
        "metric_pot_f1": find_last_float(r"'pot-f1':\s*([-+0-9.eE]+)", log_text),
        "metric_pot_precision": find_last_float(r"'pot-precision':\s*([-+0-9.eE]+)", log_text),
        "metric_pot_recall": find_last_float(r"'pot-recall':\s*([-+0-9.eE]+)", log_text),
        "metric_pot_threshold": find_last_float(r"'pot-threshold':\s*([-+0-9.eE]+)", log_text),
    }


def parse_gdn_metrics(log_text):
    return {
        "metric_f1": find_last_float(r"F1 score:\s*([-+0-9.eE]+)", log_text),
        "metric_precision": find_last_float(r"precision:\s*([-+0-9.eE]+)", log_text),
        "metric_recall": find_last_float(r"recall:\s*([-+0-9.eE]+)", log_text),
    }


def parse_metrics_for_baseline(baseline_name, log_text):
    if baseline_name == "TranAD":
        return parse_tranad_metrics(log_text)
    if baseline_name == "Anomaly-Transformer":
        return parse_anomaly_transformer_metrics(log_text)
    if baseline_name == "OmniAnomaly":
        return parse_omnianomaly_metrics(log_text)
    if baseline_name == "GDN":
        return parse_gdn_metrics(log_text)
    return {}


def infer_dataset_name(plan_experiment, registry_experiment):
    args = plan_experiment.get("args", {}) if plan_experiment else {}
    for key in ("dataset", "-dataset"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return registry_experiment.get("dataset", "")


def get_metric_protocol(dataset_name):
    dataset_name = str(dataset_name or "")
    if dataset_name in LABELED_BINARY_DATASETS:
        return "labeled_binary"
    return "unlabeled_score_only"


def adapt_metrics_for_dataset(dataset_name, metrics):
    protocol = get_metric_protocol(dataset_name)
    adapted = dict(metrics)
    adapted["metric_protocol"] = protocol
    adapted["classification_metrics_valid"] = protocol == "labeled_binary"

    if protocol != "labeled_binary":
        adapted["metric_note"] = (
            "This dataset does not use labeled binary anomaly metrics in the thesis protocol; "
            "raw parsed classification metrics are kept only for debugging."
        )
        for key in list(CLASSIFICATION_METRIC_KEYS):
            if key in adapted and adapted[key] is not None:
                adapted[f"raw_{key}"] = adapted[key]
                adapted[key] = None

    return adapted


def build_artifact_fields(registry_experiment, plan_experiment):
    cwd = Path(registry_experiment.get("cwd", ""))
    fields = {}

    skip_marker = registry_experiment.get("skip_marker", "")
    if skip_marker:
        fields["artifact_skip_marker"] = skip_marker
        fields["artifact_skip_marker_exists"] = Path(skip_marker).exists()

    baseline = registry_experiment.get("baseline", "")
    args = plan_experiment.get("args", {}) if plan_experiment else {}

    if baseline == "OmniAnomaly":
        result_dir = args.get("result_dir")
        save_dir = args.get("save_dir")
        if result_dir:
            fields["artifact_result_dir"] = str((cwd / result_dir).resolve())
        if save_dir:
            fields["artifact_model_dir"] = str((cwd / save_dir).resolve())
    elif baseline == "GDN":
        save_pattern = args.get("-save_path_pattern")
        if save_pattern:
            fields["artifact_pretrained_dir"] = str((cwd / "pretrained" / save_pattern).resolve())
            fields["artifact_results_dir"] = str((cwd / "results" / save_pattern).resolve())
    elif baseline == "Anomaly-Transformer":
        model_save_path = args.get("model_save_path")
        if model_save_path:
            fields["artifact_model_dir"] = str((cwd / model_save_path).resolve())
    elif baseline == "TranAD" and skip_marker:
        fields["artifact_checkpoint"] = skip_marker

    return fields


def collect_rows(registry_path):
    registry = load_json(registry_path)
    plan_path = Path(registry["plan_path"])
    plan_lookup = load_plan_lookup(plan_path)
    rows = []

    for experiment in registry.get("experiments", []):
        name = experiment.get("name", "")
        plan_experiment = plan_lookup.get(name, {})
        dataset_name = infer_dataset_name(plan_experiment, experiment)
        log_path = experiment.get("log_path", "")
        log_text = read_text_if_exists(log_path)
        row = {
            "experiment_name": name,
            "baseline": experiment.get("baseline", ""),
            "dataset": dataset_name,
            "status": experiment.get("status", ""),
            "return_code": experiment.get("return_code"),
            "cwd": experiment.get("cwd", ""),
            "log_path": log_path,
            "log_exists": bool(log_text),
        }

        for key, value in (plan_experiment.get("args", {}) if plan_experiment else {}).items():
            row[f"arg_{key}"] = value

        row.update(build_artifact_fields(experiment, plan_experiment))
        row.update(
            adapt_metrics_for_dataset(
                dataset_name,
                parse_metrics_for_baseline(row["baseline"], log_text),
            )
        )
        rows.append(row)

    return rows


def write_csv(rows, csv_path):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect external baseline results into flat CSV and JSON tables.")
    parser.add_argument(
        "--registry",
        type=str,
        required=True,
        help="Path to run_registry.json generated by run_external_baselines.py",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional output prefix. Defaults to the registry directory / external_results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    registry_path = Path(args.registry).resolve()
    rows = collect_rows(registry_path)

    if args.output_prefix:
        output_prefix = Path(args.output_prefix).resolve()
    else:
        output_prefix = registry_path.parent / "external_results"

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")

    write_csv(rows, csv_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Collected {len(rows)} rows")
    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")


if __name__ == "__main__":
    main()
