"""Train/evaluate internal models on the official vehicle-level battery data."""

from __future__ import annotations

import csv
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.args import apply_dataset_defaults, get_parser
from src.data.nc_battery import (
    BatterySnippetWindowDataset,
    RESPONSE_DIMS,
    StreamingMinMaxScaler,
    aggregate_vehicle_scores,
    build_index,
    load_snippet,
    split_vehicle_folds,
)
from src.models.model_factory import build_model, resolve_model_args
from src.project_paths import MANUAL_RUNS_ROOT
from src.runners.train import build_trainer, maybe_resume_trainer


def _loader(dataset, args, shuffle):
    options = {
        "batch_size": args.bs,
        "shuffle": shuffle,
        "num_workers": max(0, args.num_workers),
        "pin_memory": bool(torch.cuda.is_available()),
    }
    if args.num_workers > 0:
        options["persistent_workers"] = bool(args.persistent_workers)
        options["prefetch_factor"] = max(1, args.prefetch_factor)
    return DataLoader(dataset, **options)


def _branch_weights(pred_errors: np.ndarray, recon_errors: np.ndarray, mode: str, gamma: float):
    if mode != "quality_aware":
        return np.ones(pred_errors.shape[1]), np.full(pred_errors.shape[1], gamma)

    def stability(values):
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        return mad / (np.median(np.abs(values), axis=0) + 1e-6)

    pred_quality = 1.0 / (stability(pred_errors) + 1e-6)
    recon_quality = float(gamma) / (stability(recon_errors) + 1e-6)
    total = pred_quality + recon_quality + 1e-6
    return pred_quality / total, recon_quality / total


def _physical_window_error(x: np.ndarray, recons: np.ndarray, requested_terms: str) -> np.ndarray:
    """Dimensionless electro-thermal response discrepancy per model window."""
    enabled = {term.strip() for term in requested_terms.split(",") if term.strip()}
    terms = []
    if "voltage_rate" in enabled:
        terms.append(np.mean(np.abs(np.diff(x[:, :, 0], axis=1) - np.diff(recons[:, :, 0], axis=1)), axis=1))
    if "temperature_rate" in enabled:
        terms.append(np.mean(np.abs(np.diff(x[:, :, 5], axis=1) - np.diff(recons[:, :, 5], axis=1)), axis=1))
    if "charge_flow" in enabled:
        actual_flow = np.cumsum(x[:, :, 1], axis=1)
        recon_flow = np.cumsum(recons[:, :, 1], axis=1)
        actual_flow /= np.maximum(np.max(np.abs(actual_flow), axis=1, keepdims=True), 1e-6)
        recon_flow /= np.maximum(np.max(np.abs(recon_flow), axis=1, keepdims=True), 1e-6)
        terms.append(np.mean(np.abs(actual_flow - recon_flow), axis=1))
    if "voltage_spread" in enabled:
        terms.append(np.mean(np.abs((x[:, :, 3] - x[:, :, 4]) - (recons[:, :, 3] - recons[:, :, 4])), axis=1))
    if "temperature_spread" in enabled:
        terms.append(np.mean(np.abs((x[:, :, 5] - x[:, :, 6]) - (recons[:, :, 5] - recons[:, :, 6])), axis=1))
    if "soc_current_coupling" in enabled:
        actual_soc = np.diff(x[:, :, 2], axis=1)
        recon_soc = np.diff(recons[:, :, 2], axis=1)
        actual_current = x[:, 1:, 1]
        recon_current = recons[:, 1:, 1]
        actual_soc = actual_soc / np.maximum(np.mean(np.abs(actual_soc), axis=1, keepdims=True), 1e-6)
        recon_soc = recon_soc / np.maximum(np.mean(np.abs(recon_soc), axis=1, keepdims=True), 1e-6)
        actual_current = actual_current / np.maximum(np.mean(np.abs(actual_current), axis=1, keepdims=True), 1e-6)
        recon_current = recon_current / np.maximum(np.mean(np.abs(recon_current), axis=1, keepdims=True), 1e-6)
        terms.append(np.mean(np.abs((actual_soc - actual_current) - (recon_soc - recon_current)), axis=1))
    if not terms:
        return np.zeros(x.shape[0], dtype=np.float32)
    return np.mean(np.stack(terms, axis=1), axis=1)


@torch.no_grad()
def _collect_errors(model, loader, device, physical_terms):
    model.eval()
    pred_parts, recon_parts, physical_parts = [], [], []
    cars, labels, snippets = [], [], []
    for x, y, batch_cars, batch_labels, batch_snippets in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds, recons = model(x)
        if preds.ndim == 3:
            preds = preds.squeeze(1)
        y_last = y.squeeze(1)
        pred_error = torch.abs(y_last - preds).cpu().numpy()
        recon_error = torch.abs(x[:, -1, :] - recons[:, -1, :]).cpu().numpy()
        pred_parts.append(pred_error)
        recon_parts.append(recon_error)
        physical_parts.append(_physical_window_error(x.cpu().numpy(), recons.cpu().numpy(), physical_terms))
        cars.extend(str(value) for value in batch_cars)
        labels.extend(int(value) for value in batch_labels)
        snippets.extend(str(value) for value in batch_snippets)
    return {
        "pred": np.concatenate(pred_parts),
        "recon": np.concatenate(recon_parts),
        "physical": np.concatenate(physical_parts),
        "cars": cars,
        "labels": labels,
        "snippets": snippets,
    }


def _window_scores(errors, pred_weights, recon_weights, dims):
    combined = errors["pred"][:, dims] * pred_weights[dims]
    combined += errors["recon"][:, dims] * recon_weights[dims]
    return np.mean(combined, axis=1)


def _fit_physical_fusion(model_scores, physical_scores, max_weight):
    def robust_stats(values):
        center = float(np.median(values))
        scale = float(np.median(np.abs(values - center)) + 1e-6)
        stability = scale / (float(np.median(np.abs(values))) + 1e-6)
        return center, scale, stability

    model_center, model_scale, model_stability = robust_stats(model_scores)
    physical_center, physical_scale, physical_stability = robust_stats(physical_scores)
    weight = model_stability / (model_stability + physical_stability + 1e-6)
    return {
        "model_center": model_center,
        "model_scale": model_scale,
        "physical_center": physical_center,
        "physical_scale": physical_scale,
        "weight": float(np.clip(weight, 0.0, max_weight)),
    }


def _apply_physical_fusion(model_scores, physical_scores, calibration):
    aligned = calibration["model_center"] + calibration["model_scale"] * np.maximum(
        0.0,
        (physical_scores - calibration["physical_center"]) / calibration["physical_scale"],
    )
    weight = calibration["weight"]
    return (1.0 - weight) * model_scores + weight * aligned


def _aggregate(errors, window_scores, top_ratio):
    score_map, car_map, labels = {}, {}, {}
    for snippet, car, label, score in zip(
        errors["snippets"], errors["cars"], errors["labels"], window_scores
    ):
        score_map.setdefault(snippet, []).append(float(score))
        car_map[snippet] = car
        labels[car] = int(label)
    return aggregate_vehicle_scores(score_map, car_map, labels, top_ratio)


def _labelled_f1_threshold(scores, labels):
    """Select a labelled calibration threshold, preferring the stricter tie."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    unique = np.unique(scores)
    candidates = np.r_[np.nextafter(unique[0], -np.inf), unique]
    best_threshold, best_f1 = float(candidates[0]), -1.0
    for threshold in candidates:
        predictions = (scores > threshold).astype(np.int64)
        value = float(f1_score(labels, predictions, zero_division=0))
        if value > best_f1 or (value == best_f1 and threshold > best_threshold):
            best_threshold, best_f1 = float(threshold), value
    return best_threshold


def _metrics(
    scores,
    labels,
    calibration_scores,
    calibration_labels=None,
    threshold_mode="normal_p99",
):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    if calibration_labels is not None:
        calibration_labels = np.asarray(calibration_labels, dtype=np.int64)
    if threshold_mode == "paper_labelled_f1":
        if calibration_labels is None:
            raise ValueError("paper_labelled_f1 requires calibration labels")
        threshold = _labelled_f1_threshold(calibration_scores, calibration_labels)
    elif threshold_mode == "normal_p99":
        threshold = float(np.quantile(calibration_scores, 0.99))
    else:
        raise ValueError(f"Unsupported threshold mode: {threshold_mode}")
    predictions = (scores > threshold).astype(np.int64)
    fpr, tpr, _ = roc_curve(labels, scores)
    result = {
        "vehicle_auroc": float(roc_auc_score(labels, scores)),
        "vehicle_auprc": float(average_precision_score(labels, scores)),
        "vehicle_f1_at_calibrated_threshold": float(f1_score(labels, predictions, zero_division=0)),
        "calibration_threshold": threshold,
        "threshold_mode": threshold_mode,
        "tpr_at_fpr_1pct": float(np.max(tpr[fpr <= 0.01], initial=0.0)),
        "tpr_at_fpr_5pct": float(np.max(tpr[fpr <= 0.05], initial=0.0)),
    }
    if threshold_mode == "normal_p99":
        result["vehicle_f1_at_normal_p99"] = result["vehicle_f1_at_calibrated_threshold"]
        result["normal_validation_p99_threshold"] = threshold
    else:
        result["vehicle_f1_at_labelled_calibration"] = result[
            "vehicle_f1_at_calibrated_threshold"
        ]
    return result


def _evaluate_scoring_variant(
    normal_calibration_errors,
    threshold_calibration_errors,
    test_errors,
    pred_weights,
    recon_weights,
    dims,
    top_ratio,
    *,
    physical_max_weight=None,
    threshold_mode="normal_p99",
):
    """Evaluate a scoring rule without retraining or repeating model inference."""
    normal_calibration_window_scores = _window_scores(
        normal_calibration_errors, pred_weights, recon_weights, dims
    )
    threshold_calibration_window_scores = _window_scores(
        threshold_calibration_errors, pred_weights, recon_weights, dims
    )
    test_window_scores = _window_scores(test_errors, pred_weights, recon_weights, dims)
    physical_calibration = None
    if physical_max_weight is not None:
        physical_calibration = _fit_physical_fusion(
            normal_calibration_window_scores,
            normal_calibration_errors["physical"],
            physical_max_weight,
        )
        normal_calibration_window_scores = _apply_physical_fusion(
            normal_calibration_window_scores,
            normal_calibration_errors["physical"],
            physical_calibration,
        )
        threshold_calibration_window_scores = _apply_physical_fusion(
            threshold_calibration_window_scores,
            threshold_calibration_errors["physical"],
            physical_calibration,
        )
        test_window_scores = _apply_physical_fusion(
            test_window_scores,
            test_errors["physical"],
            physical_calibration,
        )

    _, _, normal_calibration_cars = _aggregate(
        normal_calibration_errors, normal_calibration_window_scores, top_ratio
    )
    calibration_scores, calibration_labels, calibration_cars = _aggregate(
        threshold_calibration_errors, threshold_calibration_window_scores, top_ratio
    )
    scores, labels, cars = _aggregate(test_errors, test_window_scores, top_ratio)
    aggregation_sensitivity = {}
    for ratio in (0.01, 0.05, 0.10, 0.20, 1.0):
        ratio_calibration_scores, ratio_calibration_labels, _ = _aggregate(
            threshold_calibration_errors, threshold_calibration_window_scores, ratio
        )
        ratio_scores, ratio_labels, _ = _aggregate(test_errors, test_window_scores, ratio)
        key = f"top_{int(ratio * 100)}pct" if ratio < 1 else "mean"
        aggregation_sensitivity[key] = _metrics(
            ratio_scores,
            ratio_labels,
            ratio_calibration_scores,
            ratio_calibration_labels,
            threshold_mode,
        )
    return {
        "score_dims": np.asarray(dims).tolist(),
        "physical_fusion": physical_calibration,
        "metrics": _metrics(
            scores, labels, calibration_scores, calibration_labels, threshold_mode
        ),
        "aggregation_sensitivity": aggregation_sensitivity,
        "normal_calibration_vehicle_count": len(normal_calibration_cars),
        "threshold_calibration_vehicle_count": len(calibration_cars),
        "test_vehicle_count": len(cars),
    }, scores, labels, cars, test_window_scores


def _natural_key(value):
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def _rank_correlation(left, right):
    if len(left) < 2:
        return None

    def average_ranks(values):
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return ranks

    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if np.std(left_rank) < 1e-12 or np.std(right_rank) < 1e-12:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _write_voltage_spread_case(
    output_dir,
    test_records,
    test_errors,
    test_window_scores,
    vehicle_scores,
    vehicle_labels,
    vehicle_cars,
):
    """Write one faulty-car score/spread case motivated by Supplementary Fig. 3."""
    faulty = [
        (float(score), str(car))
        for score, label, car in zip(vehicle_scores, vehicle_labels, vehicle_cars)
        if int(label) == 1
    ]
    if not faulty:
        return {"generated": False, "reason": "no_faulty_vehicle_in_test_split"}
    _, selected_car = max(faulty)

    snippet_scores = {}
    for snippet, car, score in zip(
        test_errors["snippets"], test_errors["cars"], test_window_scores
    ):
        if str(car) == selected_car:
            snippet_scores.setdefault(str(snippet), []).append(float(score))

    rows = []
    for record in test_records:
        if record.car != selected_car:
            continue
        path = Path(record.path)
        snippet = f"{path.parent.name}/{path.stem}"
        if snippet not in snippet_scores:
            continue
        values, metadata = load_snippet(record.path)
        timestamp = metadata.get("timestamp", metadata.get("time"))
        charge_segment = metadata.get("charge_segment", record.charge_segment)
        rows.append({
            "snippet": snippet,
            "timestamp": "" if timestamp is None else str(timestamp),
            "charge_segment": "" if charge_segment is None else str(charge_segment),
            "mileage": "" if record.mileage is None else float(record.mileage),
            "anomaly_score": float(np.mean(snippet_scores[snippet])),
            "mean_vmax_minus_vmin": float(np.mean(values[:, 3] - values[:, 4])),
            "max_vmax_minus_vmin": float(np.max(values[:, 3] - values[:, 4])),
        })
    if not rows:
        return {"generated": False, "reason": "selected_vehicle_has_no_scored_snippets"}

    if all(row["timestamp"] for row in rows):
        order_field = "timestamp"
    elif all(row["charge_segment"] for row in rows):
        order_field = "charge_segment"
    elif all(row["mileage"] != "" for row in rows):
        order_field = "mileage"
    else:
        order_field = "snippet"
    rows.sort(key=lambda row: _natural_key(row[order_field]))
    for index, row in enumerate(rows):
        row["charging_order"] = index
        row["order_source"] = order_field

    csv_path = output_dir / "voltage_spread_case.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.arange(len(rows))
    anomaly_scores = np.asarray([row["anomaly_score"] for row in rows])
    voltage_spreads = np.asarray([row["mean_vmax_minus_vmin"] for row in rows])
    fig, left_axis = plt.subplots(figsize=(12, 4.8))
    right_axis = left_axis.twinx()
    left_axis.plot(order, anomaly_scores, color="tab:red", linewidth=1.1, label="Anomaly score")
    right_axis.plot(
        order, voltage_spreads, color="tab:blue", linewidth=1.1,
        label="Mean Vmax-Vmin",
    )
    left_axis.set_xlabel(f"Charging snippet order ({order_field})")
    left_axis.set_ylabel("Anomaly score", color="tab:red")
    right_axis.set_ylabel("Mean Vmax-Vmin (raw released units)", color="tab:blue")
    left_axis.set_title(
        f"Faulty vehicle {selected_car}: score and cell-voltage spread\n"
        "Interpretation motivated by Zhang et al. (2023), Supplementary Fig. 3"
    )
    left_axis.grid(alpha=0.2)
    fig.tight_layout()
    figure_path = output_dir / "voltage_spread_case.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    return {
        "generated": True,
        "vehicle": selected_car,
        "selection": "highest_scoring_faulty_vehicle_in_test_split",
        "snippet_count": len(rows),
        "order_source": order_field,
        "score_spread_rank_correlation": _rank_correlation(anomaly_scores, voltage_spreads),
        "csv": csv_path.name,
        "figure": figure_path.name,
        "evidence_source": (
            "Zhang et al., Realistic Fault Detection of Li-ion Battery via "
            "Dynamical Deep Learning, Supplementary Figure 3"
        ),
        "interpretation_limit": "descriptive case analysis; not a training target or causal test",
    }


def main():
    args = apply_dataset_defaults(get_parser().parse_args())
    args.dataset = "TSINGHUA_EV"
    resolve_model_args(args)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = build_index(
        args.tsinghua_ev_root,
        args.battery_brand,
        max_snippets=args.battery_max_index_snippets,
    )
    splits = split_vehicle_folds(
        records,
        args.battery_fold,
        seed=args.seed,
        protocol=args.battery_split_protocol,
    )
    if args.battery_max_snippets_per_vehicle > 0:
        capped = {}
        for split_name, split_records in splits.items():
            counts = {}
            capped[split_name] = []
            for record in split_records:
                if counts.get(record.car, 0) >= args.battery_max_snippets_per_vehicle:
                    continue
                capped[split_name].append(record)
                counts[record.car] = counts.get(record.car, 0) + 1
        splits = capped
    scaler = StreamingMinMaxScaler().fit_records(splits["train"]) if args.normalize else None
    datasets = {
        name: BatterySnippetWindowDataset(
            split,
            args.lookback,
            scaler,
            windows_per_snippet=args.battery_windows_per_snippet,
            include_metadata=name in {"validation", "calibration", "test"},
        )
        for name, split in splits.items()
    }
    output_dir = Path(
        os.environ.get(
            "PLAN_OUTPUT_DIR",
            MANUAL_RUNS_ROOT / "TSINGHUA_EV" / (args.run_id or f"brand{args.battery_brand}_fold{args.battery_fold}"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[NC battery] brand={args.battery_brand} fold={args.battery_fold} "
        f"records={len(records)} windows(train/val/test)="
        f"{len(datasets['train'])}/{len(datasets['validation'])}/{len(datasets['test'])}"
    )

    train_loader = _loader(datasets["train"], args, True)
    validation_train_view = BatterySnippetWindowDataset(
        splits["validation"], args.lookback, scaler,
        windows_per_snippet=args.battery_windows_per_snippet,
        include_metadata=False,
    )
    val_loader = _loader(validation_train_view, args, False)
    model = build_model(args, 7, args.lookback, 7, target_dims=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    trainer = build_trainer(
        model, optimizer, args, args.lookback, 7, None,
        str(output_dir), str(output_dir / "logs"), json.dumps(vars(args), ensure_ascii=False),
    )
    maybe_resume_trainer(trainer, args)
    trainer.fit(train_loader, val_loader)

    try:
        state = torch.load(output_dir / "model.pt", map_location=trainer.device, weights_only=True)
    except TypeError:
        state = torch.load(output_dir / "model.pt", map_location=trainer.device)
    model.load_state_dict(state)
    validation_errors = _collect_errors(
        model, _loader(datasets["validation"], args, False), trainer.device, args.physical_response_terms
    )
    if splits["calibration"] is splits["validation"]:
        calibration_errors = validation_errors
    else:
        calibration_errors = _collect_errors(
            model,
            _loader(datasets["calibration"], args, False),
            trainer.device,
            args.physical_response_terms,
        )
    test_errors = _collect_errors(
        model, _loader(datasets["test"], args, False), trainer.device, args.physical_response_terms
    )
    pred_weights, recon_weights = _branch_weights(
        validation_errors["pred"], validation_errors["recon"], args.score_fusion_mode, args.gamma
    )
    all_dims = np.arange(7)
    response_dims = np.asarray(RESPONSE_DIMS)
    scoring_specs = {}
    if args.model_name == "mtad_gat":
        # A faithful all-channel baseline and its task-adapted response score
        # share exactly the same trained weights and inference errors.
        scoring_specs["all_channels"] = (all_dims, None)
        scoring_specs["response_channels"] = (response_dims, None)
        primary_variant = "all_channels" if args.battery_score_channels == "all" else "response_channels"
    else:
        scoring_specs["response_channels"] = (response_dims, None)
        primary_variant = "response_channels"
    if args.use_physical_response_score:
        scoring_specs["response_plus_physics"] = (
            response_dims,
            args.physical_response_max_weight,
        )
        primary_variant = "response_plus_physics"

    scoring_variants = {}
    variant_arrays = {}
    for variant_name, (variant_dims, max_physical_weight) in scoring_specs.items():
        (
            variant_result,
            variant_scores,
            variant_labels,
            variant_cars,
            variant_window_scores,
        ) = _evaluate_scoring_variant(
            validation_errors,
            calibration_errors,
            test_errors,
            pred_weights,
            recon_weights,
            variant_dims,
            args.battery_vehicle_top_ratio,
            physical_max_weight=max_physical_weight,
            threshold_mode=(
                "paper_labelled_f1"
                if args.battery_split_protocol == "paper_protocol"
                else "normal_p99"
            ),
        )
        scoring_variants[variant_name] = variant_result
        variant_arrays[variant_name] = (
            variant_scores, variant_labels, variant_cars, variant_window_scores
        )

    primary = scoring_variants[primary_variant]
    scores, labels, cars, primary_window_scores = variant_arrays[primary_variant]
    metrics = primary["metrics"]
    voltage_spread_case = _write_voltage_spread_case(
        output_dir,
        splits["test"],
        test_errors,
        primary_window_scores,
        scores,
        labels,
        cars,
    )
    result = {
        "protocol": args.battery_split_protocol,
        "protocol_source": (
            "project strict normal-only calibration"
            if args.battery_split_protocol == "strict_normal_validation"
            else "Zhang et al. 2023 Supplementary Note 2, Five fold evaluation"
        ),
        "brand": args.battery_brand,
        "fold": args.battery_fold,
        "model_name": args.model_name,
        "primary_scoring_variant": primary_variant,
        "score_channels": args.battery_score_channels,
        "score_dims": primary["score_dims"],
        "vehicle_top_ratio": args.battery_vehicle_top_ratio,
        "counts": {
            "indexed_snippets": len(records),
            "train_snippets": len(splits["train"]),
            "validation_snippets": len(splits["validation"]),
            "calibration_snippets": len(splits["calibration"]),
            "test_snippets": len(splits["test"]),
            "normal_calibration_vehicles": primary["normal_calibration_vehicle_count"],
            "threshold_calibration_vehicles": primary["threshold_calibration_vehicle_count"],
            "test_vehicles": primary["test_vehicle_count"],
        },
        "metrics": metrics,
        "aggregation_sensitivity": primary["aggregation_sensitivity"],
        "scoring_variants": scoring_variants,
        "prediction_weights": pred_weights.tolist(),
        "reconstruction_weights": recon_weights.tolist(),
        "physical_fusion": primary["physical_fusion"],
        "voltage_spread_case": voltage_spread_case,
        "scaler": None if scaler is None else scaler.state_dict(),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    with (output_dir / "vehicle_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["car", "label", "score"])
        writer.writerows(zip(cars, labels.tolist(), scores.tolist()))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
