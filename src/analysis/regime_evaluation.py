"""Evaluation utilities for condition embeddings and unlabeled BMS operation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from src.data.regime_utils import NASA_REGIME_NAMES
from src.data.utils import SlidingWindowDataset, ensure_sequence_list


def _unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def _collect_nasa_embeddings(model, entity_data, window_size, max_windows, device):
    embeddings = []
    labels = []
    base_model = _unwrap_model(model)
    for values in entity_data.values():
        for sequence in ensure_sequence_list(values):
            tensor = sequence if isinstance(sequence, torch.Tensor) else torch.as_tensor(sequence, dtype=torch.float32)
            if len(tensor) <= window_size:
                continue
            dataset = SlidingWindowDataset(tensor, window_size, target_dim=None, stride=1)
            if len(dataset) > max_windows:
                indices = np.linspace(0, len(dataset) - 1, max_windows, dtype=np.int64).tolist()
                dataset = Subset(dataset, indices)
            loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    embedding = base_model.encode_regime(x).detach().cpu().numpy()
                    # The raw discrete step code is deliberately absent from model
                    # inputs. This weak sanity probe groups windows by current direction.
                    current = y[:, 0, 1].detach().cpu().numpy()
                    label = np.zeros(len(current), dtype=np.int64)
                    label[current > 0.05] = 1
                    label[current < -0.05] = 2
                    embeddings.append(embedding)
                    labels.append(label)
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(embeddings), np.concatenate(labels)


def save_nasa_regime_probe(
    output_dir,
    model,
    train_entities,
    test_entities,
    window_size,
    max_windows_per_sequence=3000,
):
    """Fit a train-battery linear probe and evaluate it on held-out batteries."""
    base_model = _unwrap_model(model)
    path = Path(output_dir) / "nasa_regime_probe.json"
    # Baseline and C4-only models do not expose a regime embedding.  Their
    # anomaly outputs remain valid; only this C3-specific probe is inapplicable.
    if not bool(getattr(base_model, "use_regime_condition", False)):
        report = {
            "available": False,
            "reason": "regime conditioning is disabled; NASA regime probe is not applicable",
        }
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    device = next(model.parameters()).device
    train_x, train_y = _collect_nasa_embeddings(
        model, train_entities, window_size, max_windows_per_sequence, device
    )
    test_x, test_y = _collect_nasa_embeddings(
        model, test_entities, window_size, max_windows_per_sequence, device
    )
    train_classes = np.unique(train_y)
    if train_x.size == 0 or test_x.size == 0 or train_classes.size < 2:
        report = {"available": False, "reason": "insufficient windows or fewer than two train regimes"}
    else:
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=3407)
        classifier.fit(train_x, train_y)
        prediction = classifier.predict(test_x)
        report = {
            "available": True,
            "label_semantics": NASA_REGIME_NAMES,
            "train_window_count": int(len(train_y)),
            "test_window_count": int(len(test_y)),
            "accuracy": float(accuracy_score(test_y, prediction)),
            "macro_f1": float(f1_score(test_y, prediction, average="macro")),
            "train_classes": train_classes.tolist(),
            "test_classes": np.unique(test_y).tolist(),
            "label_source": "current direction; raw step_type_code excluded from model inputs",
            "evaluation": "weak representation sanity probe across held-out batteries",
        }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _distribution_stats(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_median": None,
            f"{prefix}_mad": None,
            f"{prefix}_cv": None,
        }
    mean = float(np.mean(values))
    median = float(np.median(values))
    std = float(np.std(values))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_median": median,
        f"{prefix}_mad": float(np.median(np.abs(values - median))),
        f"{prefix}_cv": float(std / max(abs(mean), 1e-12)),
    }


def _false_alarm_stats(alarms):
    alarms = np.asarray(alarms, dtype=np.int64)
    count = int(np.sum(alarms))
    total = int(alarms.size)
    rate = float(count / total) if total else None
    return {
        "window_count": total,
        "false_alarm_count": count,
        "false_alarm_rate": rate,
        "false_alarms_per_10k_windows": None if rate is None else float(rate * 10_000),
    }


def save_bms_operational_report(output_dir, test_entities, window_size, block_size=1000):
    """Report empirical false alarms and score stability on known-normal BMS data."""
    output_dir = Path(output_dir)
    cluster_rows = []
    block_rows = []
    all_scores = []
    all_normalized_scores = []
    all_alarms = []
    block_size = max(1, int(block_size))

    for cluster_name in sorted(test_entities):
        score_path = output_dir / cluster_name / "test_output.pkl"
        if not score_path.exists():
            continue
        frame = pd.read_pickle(score_path)
        required = {"A_Pred_Global", "A_Score_Global"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{score_path} is missing required columns: {missing}")
        alarms = frame["A_Pred_Global"].to_numpy(dtype=np.int64)
        scores = frame["A_Score_Global"].to_numpy(dtype=np.float64)
        threshold = None
        if "Thresh_Global" in frame and len(frame):
            threshold = float(frame["Thresh_Global"].iloc[0])
        normalized_scores = scores / max(abs(threshold), 1e-12) if threshold is not None else scores

        row = {
            "cluster": cluster_name,
            "threshold": threshold,
            **_false_alarm_stats(alarms),
            **_distribution_stats(scores, "score"),
            **_distribution_stats(normalized_scores, "score_to_threshold"),
        }
        cluster_rows.append(row)
        all_scores.extend(scores.tolist())
        all_normalized_scores.extend(normalized_scores.tolist())
        all_alarms.extend(alarms.tolist())

        for block_index, start in enumerate(range(0, len(frame), block_size)):
            stop = min(len(frame), start + block_size)
            block_rows.append({
                "cluster": cluster_name,
                "block_index": int(block_index),
                "start_window": int(start),
                "stop_window": int(stop),
                **_false_alarm_stats(alarms[start:stop]),
                **_distribution_stats(normalized_scores[start:stop], "score_to_threshold"),
            })

    cluster_frame = pd.DataFrame(cluster_rows)
    block_frame = pd.DataFrame(block_rows)
    cluster_frame.to_csv(output_dir / "bms_false_alarm_by_cluster.csv", index=False)
    block_frame.to_csv(output_dir / "bms_false_alarm_by_block.csv", index=False)
    # Retain the previous filename so old collection scripts keep working.
    cluster_frame.to_csv(output_dir / "bms_operational_stability.csv", index=False)

    cluster_rates = cluster_frame.get("false_alarm_rate", pd.Series(dtype=float)).dropna().to_numpy()
    block_rates = block_frame.get("false_alarm_rate", pd.Series(dtype=float)).dropna().to_numpy()
    report = {
        "data_assumption": "the evaluated BMS interval is confirmed normal operation",
        "metric_semantics": "threshold exceedances are empirical false alarms, not detected faults",
        "threshold_source": "normal training scores via the model's epsilon threshold",
        "reported_supervised_fault_metrics": False,
        "window_size": int(window_size),
        "time_block_size_windows": int(block_size),
        "cluster_count": int(len(cluster_frame)),
        **_false_alarm_stats(all_alarms),
        **_distribution_stats(all_scores, "score"),
        **_distribution_stats(all_normalized_scores, "score_to_threshold"),
        "cluster_false_alarm_rate_std": float(np.std(cluster_rates)) if cluster_rates.size else None,
        "cluster_false_alarm_rate_range": float(np.ptp(cluster_rates)) if cluster_rates.size else None,
        "block_false_alarm_rate_std": float(np.std(block_rates)) if block_rates.size else None,
        "block_false_alarm_rate_p95": float(np.quantile(block_rates, 0.95)) if block_rates.size else None,
        "block_false_alarm_rate_max": float(np.max(block_rates)) if block_rates.size else None,
    }
    (output_dir / "bms_operational_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def save_bms_conditioning_comparison(batch_root):
    """Summarize all completed BMS models using normal-only diagnostics.

    No model is declared a fault detector here: each value is an empirical
    threshold-exceedance or score-stability statistic on confirmed-normal data.
    """
    batch_root = Path(batch_root)
    output_root = batch_root / "output"
    run_names = {
        "baseline": "bms_mtadgat",
        "c3_feature_gat": "bms_c3_feature_gat",
        "c4_backbone": "bms_mtadgat_c4_backbone",
        "c4_physical_consistency": "bms_c4_physical_consistency",
        # Backward-compatible names for the old plan.
        "unconditioned": "bms_frequency_regulation_unconditioned",
        "conditioned": "bms_frequency_regulation_conditioned",
    }
    reports = {}
    for label, run_name in run_names.items():
        path = output_root / run_name / "bms_operational_summary.json"
        if path.is_file():
            reports[label] = json.loads(path.read_text(encoding="utf-8"))

    if not reports:
        return None

    metric_names = (
        "false_alarm_rate",
        "false_alarms_per_10k_windows",
        "cluster_false_alarm_rate_std",
        "cluster_false_alarm_rate_range",
        "block_false_alarm_rate_std",
        "block_false_alarm_rate_p95",
        "block_false_alarm_rate_max",
        "score_to_threshold_cv",
    )
    rows = []
    reference_label = "baseline" if "baseline" in reports else next(iter(reports))
    for metric in metric_names:
        reference = reports[reference_label].get(metric)
        for label, report in reports.items():
            value = report.get(metric)
            delta = None if value is None or reference is None else float(value - reference)
            relative_change = None
            if delta is not None and abs(float(reference)) > 1e-12:
                relative_change = float(delta / abs(float(reference)))
            rows.append({
                "metric": metric,
                "model": label,
                "value": value,
                "reference_model": reference_label,
                "reference_value": reference,
                "minus_reference": delta,
                "relative_change": relative_change,
                "preferred_direction": "lower",
            })

    comparison = {
        "data_assumption": "all evaluated BMS windows are normal",
        "reference_model": reference_label,
        "interpretation": "negative deltas indicate fewer or more stable false alarms",
        "fault_detection_claim_supported": False,
        "metrics": rows,
    }
    pd.DataFrame(rows).to_csv(batch_root / "bms_conditioning_comparison.csv", index=False)
    (batch_root / "bms_conditioning_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    return comparison
