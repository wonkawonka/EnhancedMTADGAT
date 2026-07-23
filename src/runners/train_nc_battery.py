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
from sklearn.metrics import auc, average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from src.args import apply_dataset_defaults, get_parser
from src.data.nc_battery import (
    BatterySnippetWindowDataset,
    PaperChannelNormalizer,
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


def _condition_descriptors(
    x: np.ndarray, mileages: np.ndarray | None = None, include_slow_state: bool = False
) -> np.ndarray:
    """Short-horizon controls plus an optional label-free slow operating state."""
    current = x[:, :, 1]
    soc = x[:, :, 2]
    descriptors = [
        np.mean(current, axis=1),
        np.std(current, axis=1),
        current[:, -1],
        np.mean(np.abs(np.diff(current, axis=1)), axis=1),
        soc[:, 0],
        soc[:, -1],
        soc[:, -1] - soc[:, 0],
    ]
    if include_slow_state:
        if mileages is None:
            raise ValueError("C3 slow-state descriptors require snippet mileage metadata")
        mileage = np.asarray(mileages, dtype=np.float32).reshape(-1)
        # Log mileage is a causal, label-free ageing proxy. Missing values are
        # represented by the normal-data median rather than a sentinel regime.
        finite = np.isfinite(mileage)
        fill = np.median(mileage[finite]) if np.any(finite) else 0.0
        descriptors.append(np.log1p(np.where(finite, np.maximum(mileage, 0.0), fill)))
    return np.stack(
        descriptors,
        axis=1,
    ).astype(np.float32)


@torch.no_grad()
def _collect_errors(
    model, loader, device, physical_terms, target_dims=None, use_condition_slow_state=False
):
    model.eval()
    pred_parts, recon_parts, pred_signed_parts, recon_signed_parts = [], [], [], []
    physical_parts, consistency_parts, condition_parts = [], [], []
    cars, labels, snippets = [], [], []
    for batch in loader:
        if len(batch) == 6:
            x, y, batch_cars, batch_labels, batch_snippets, batch_mileages = batch
        else:
            x, y, batch_cars, batch_labels, batch_snippets = batch
            batch_mileages = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds, recons = model(x)
        if preds.ndim == 3:
            preds = preds.squeeze(1)
        y_last = y.squeeze(1)
        if target_dims is not None:
            y_last = y_last[:, target_dims]
            x_last = x[:, -1, target_dims]
            recons_full = x.clone()
            recons_full[:, :, target_dims] = recons
        else:
            x_last = x[:, -1, :]
            recons_full = recons
        pred_signed = (y_last - preds).cpu().numpy()
        recon_signed = (x_last - recons[:, -1, :]).cpu().numpy()
        pred_error = np.abs(pred_signed)
        recon_error = np.abs(recon_signed)
        pred_parts.append(pred_error)
        recon_parts.append(recon_error)
        pred_signed_parts.append(pred_signed)
        recon_signed_parts.append(recon_signed)
        state_values = x.cpu().numpy()
        condition_parts.append(
            _condition_descriptors(
                state_values,
                None if batch_mileages is None else batch_mileages.cpu().numpy(),
                use_condition_slow_state,
            )
        )
        physical_parts.append(
            _physical_window_error(state_values, recons_full.cpu().numpy(), physical_terms)
        )
        base_model = model.module if hasattr(model, "module") else model
        consistency_prediction = getattr(base_model, "_physical_consistency_prediction", None)
        if consistency_prediction is None:
            consistency_parts.append(np.zeros(x.size(0), dtype=np.float32))
        else:
            consistency_target = x[:, :, base_model.physical_consistency_target_dims]
            consistency_parts.append(
                torch.mean(torch.abs(consistency_target - consistency_prediction), dim=(1, 2))
                .cpu().numpy()
            )
        cars.extend(str(value) for value in batch_cars)
        labels.extend(int(value) for value in batch_labels)
        snippets.extend(str(value) for value in batch_snippets)
    return {
        "pred": np.concatenate(pred_parts),
        "recon": np.concatenate(recon_parts),
        "pred_signed": np.concatenate(pred_signed_parts),
        "recon_signed": np.concatenate(recon_signed_parts),
        "physical": np.concatenate(physical_parts),
        "physical_consistency": np.concatenate(consistency_parts),
        "condition": np.concatenate(condition_parts),
        "cars": cars,
        "labels": labels,
        "snippets": snippets,
    }


def _condition_soft_assignments(features, centers, bandwidth):
    """Continuous responsibilities over normal operating-condition prototypes."""
    squared_distance = np.sum(
        (features[:, None, :] - centers[None, :, :]) ** 2, axis=2
    )
    logits = -squared_distance / max(2.0 * bandwidth * bandwidth, 1e-8)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)


def _fit_neural_condition_residual_model(errors, seed):
    """Fit p(residual | current/SOC trajectory) on normal windows only.

    This deliberately calibrates a *frozen* anomaly model.  The small MLP has
    no access to response channels or labels: it estimates the conditional
    mean and diagonal scale of signed MTAD residuals from control descriptors.
    """
    features = np.asarray(errors["condition"], dtype=np.float32)
    feature_center = np.median(features, axis=0)
    feature_scale = 1.4826 * np.median(np.abs(features - feature_center), axis=0)
    feature_scale = np.maximum(feature_scale, 1e-4)
    normalized = (features - feature_center) / feature_scale
    targets = np.concatenate(
        [np.asarray(errors["pred_signed"], dtype=np.float32),
         np.asarray(errors["recon_signed"], dtype=np.float32)], axis=1,
    )
    target_center = np.median(targets, axis=0)
    target_scale = 1.4826 * np.median(np.abs(targets - target_center), axis=0)
    target_scale = np.maximum(target_scale, 1e-5)
    standardized_targets = (targets - target_center) / target_scale

    # A local generator makes this post-hoc normal model reproducible without
    # perturbing the main training sampler's random-number trajectory.
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 173)
    net = torch.nn.Sequential(
        torch.nn.Linear(normalized.shape[1], 32), torch.nn.SiLU(),
        torch.nn.Linear(32, 32), torch.nn.SiLU(),
        torch.nn.Linear(32, standardized_targets.shape[1] * 2),
    )
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.copy_(torch.randn(module.weight.shape, generator=generator) * 0.05)
                module.bias.zero_()
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    x_tensor = torch.from_numpy(normalized)
    y_tensor = torch.from_numpy(standardized_targets)
    order = torch.randperm(len(x_tensor), generator=generator)
    batch_size = min(512, len(x_tensor))
    net.train()
    for _ in range(50):
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            output = net(x_tensor[index])
            mean, raw_scale = output.chunk(2, dim=1)
            # Bound the learned tolerance so a sparse operating condition
            # cannot make every residual look normal.
            scale = torch.nn.functional.softplus(raw_scale) + 0.15
            z = (y_tensor[index] - mean) / scale
            loss = (0.5 * z.square() + torch.log(scale)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    net.eval()
    return {
        "net": net,
        "feature_center": feature_center,
        "feature_scale": feature_scale,
        "target_center": target_center,
        "target_scale": target_scale,
        "method": "neural_heteroscedastic",
        "descriptor_dim": int(normalized.shape[1]),
    }


def _fit_condition_residual_calibration(
    errors, n_clusters, seed, method="hard_kmeans", temperature=1.0,
    descriptor_names=None,
):
    """Fit normal-only p(residual | continuous operating condition)."""
    if method == "neural_heteroscedastic":
        result = _fit_neural_condition_residual_model(errors, seed)
        result["descriptor_names"] = descriptor_names
        return result
    features = np.asarray(errors["condition"], dtype=np.float64)
    feature_center = np.median(features, axis=0)
    feature_scale = 1.4826 * np.median(np.abs(features - feature_center), axis=0)
    feature_scale = np.maximum(feature_scale, 1e-4)
    normalized = (features - feature_center) / feature_scale
    cluster_count = max(1, min(int(n_clusters), len(normalized)))
    estimator = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=int(seed),
        n_init=10,
        batch_size=min(2048, max(256, len(normalized))),
    ).fit(normalized)
    assignments = estimator.labels_
    if method == "soft_expert":
        center_distance = np.sqrt(
            np.sum((normalized - estimator.cluster_centers_[assignments]) ** 2, axis=1)
        )
        # A data-derived bandwidth keeps the temperature interpretable across folds.
        bandwidth = max(float(np.median(center_distance)), 0.5) * max(float(temperature), 1e-3)
        responsibilities = _condition_soft_assignments(
            normalized, estimator.cluster_centers_, bandwidth
        )
    else:
        bandwidth = None
        responsibilities = None

    profiles = {}
    for key in ("pred_signed", "recon_signed"):
        values = np.asarray(errors[key], dtype=np.float64)
        global_bias = np.median(values, axis=0)
        global_scale = 1.4826 * np.median(np.abs(values - global_bias), axis=0)
        global_scale = np.maximum(global_scale, 1e-5)
        biases, scales, counts = [], [], []
        for cluster in range(cluster_count):
            if responsibilities is not None:
                weights = responsibilities[:, cluster]
                count = float(np.sum(weights))
                # Clip only for estimating the normal profile: rare faulty-like
                # validation windows must not inflate a condition's tolerance.
                clipped = np.clip(
                    values,
                    global_bias - 6.0 * global_scale,
                    global_bias + 6.0 * global_scale,
                )
                local_bias = np.sum(weights[:, None] * clipped, axis=0) / max(count, 1e-8)
                local_scale = np.sqrt(
                    np.sum(weights[:, None] * (clipped - local_bias) ** 2, axis=0)
                    / max(count, 1e-8)
                )
            else:
                selected = values[assignments == cluster]
                count = len(selected)
                if count:
                    local_bias = np.median(selected, axis=0)
                    local_scale = 1.4826 * np.median(
                        np.abs(selected - local_bias), axis=0
                    )
                else:
                    local_bias, local_scale = global_bias, global_scale
            # Shrink small regimes toward the global normal distribution.
            reliability = count / (count + 64.0)
            biases.append(reliability * local_bias + (1.0 - reliability) * global_bias)
            scales.append(
                np.maximum(
                    reliability * local_scale + (1.0 - reliability) * global_scale,
                    0.1 * global_scale,
                )
            )
            counts.append(count)
        profiles[key] = {
            "bias": np.stack(biases),
            "scale": np.stack(scales),
            "global_bias": global_bias,
            "global_scale": global_scale,
            "counts": counts,
        }
    return {
        "estimator": estimator,
        "feature_center": feature_center,
        "feature_scale": feature_scale,
        "profiles": profiles,
        "method": method,
        "bandwidth": bandwidth,
        "descriptor_names": descriptor_names,
    }


def _apply_condition_residual_calibration(errors, calibration):
    if calibration["method"] == "neural_heteroscedastic":
        features = (errors["condition"] - calibration["feature_center"]) / calibration[
            "feature_scale"
        ]
        with torch.no_grad():
            output = calibration["net"](torch.from_numpy(features.astype(np.float32)))
            mean, raw_scale = output.chunk(2, dim=1)
            mean = mean.numpy() * calibration["target_scale"] + calibration["target_center"]
            scale = (
                (torch.nn.functional.softplus(raw_scale) + 0.15).numpy()
                * calibration["target_scale"]
            )
        calibrated = dict(errors)
        dims = errors["pred_signed"].shape[1]
        calibrated["pred"] = np.abs(errors["pred_signed"] - mean[:, :dims]) / scale[:, :dims]
        calibrated["recon"] = np.abs(errors["recon_signed"] - mean[:, dims:]) / scale[:, dims:]
        return calibrated
    features = (errors["condition"] - calibration["feature_center"]) / calibration[
        "feature_scale"
    ]
    if calibration["method"] == "soft_expert":
        responsibilities = _condition_soft_assignments(
            features,
            calibration["estimator"].cluster_centers_,
            calibration["bandwidth"],
        )
        assignments = None
    else:
        assignments = calibration["estimator"].predict(features)
        responsibilities = None
    calibrated = dict(errors)
    for signed_key, output_key in (
        ("pred_signed", "pred"),
        ("recon_signed", "recon"),
    ):
        profile = calibration["profiles"][signed_key]
        if responsibilities is None:
            bias = profile["bias"][assignments]
            scale = profile["scale"][assignments]
        else:
            bias = responsibilities @ profile["bias"]
            # Mixture variance includes within-expert uncertainty and the
            # disagreement between expert means.
            second_moment = responsibilities @ (profile["scale"] ** 2 + profile["bias"] ** 2)
            scale = np.sqrt(np.maximum(second_moment - bias ** 2, 1e-10))
        calibrated[output_key] = np.abs(errors[signed_key] - bias) / scale
    return calibrated


def _condition_calibration_metadata(calibration):
    return {
        "method": (
            "normal_only_continuous_heteroscedastic_residual_density"
            if calibration["method"] == "neural_heteroscedastic"
            else "normal_only_soft_condition_expert_residual_density"
            if calibration["method"] == "soft_expert"
            else "normal_only_operating_regime_robust_zscore"
        ),
        "descriptor_names": calibration["descriptor_names"] or [
            "current_mean", "current_std", "current_last", "current_delta_abs_mean",
            "soc_first", "soc_last", "soc_delta",
        ],
        "cluster_count": (
            None if calibration["method"] == "neural_heteroscedastic"
            else int(calibration["estimator"].n_clusters)
        ),
        "cluster_counts": (
            None if calibration["method"] == "neural_heteroscedastic"
            else calibration["profiles"]["pred_signed"]["counts"]
        ),
        "soft_assignment_bandwidth": (
            None if calibration["method"] == "neural_heteroscedastic"
            else calibration["bandwidth"]
        ),
        "fit_labels_used": False,
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


def _paper_labelled_snippet_threshold(scores, labels, granularity=1000):
    """Reproduce the released notebook's labelled snippet-rank threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-scores, kind="mergesort")
    ranked_scores = scores[order]
    ranked_labels = labels[order]
    best_fraction, best_n = -1.0, 1
    for n in range(1, 100):
        count = max(round(len(ranked_scores) * n / granularity), 1)
        fraction = float(np.mean(ranked_labels[:count]))
        if fraction > best_fraction:
            best_fraction, best_n = fraction, n
    count = max(round(len(ranked_scores) * best_n / granularity), 1)
    return float(ranked_scores[count - 1])


def _metrics(
    scores,
    labels,
    calibration_scores,
    calibration_labels=None,
    threshold_mode="normal_p99",
    calibration_threshold=None,
):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    if calibration_labels is not None:
        calibration_labels = np.asarray(calibration_labels, dtype=np.int64)
    if calibration_threshold is not None:
        threshold = float(calibration_threshold)
    elif threshold_mode == "paper_labelled_f1":
        if calibration_labels is None:
            raise ValueError("paper_labelled_f1 requires calibration labels")
        threshold = _labelled_f1_threshold(calibration_scores, calibration_labels)
    elif threshold_mode == "normal_p99":
        threshold = float(np.quantile(calibration_scores, 0.99))
    else:
        raise ValueError(f"Unsupported threshold mode: {threshold_mode}")
    predictions = (scores > threshold).astype(np.int64)
    fpr, tpr, _ = roc_curve(labels, scores)
    average_precision = float(average_precision_score(labels, scores))
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    trapezoidal_pr_auc = float(auc(recall_curve, precision_curve))
    result = {
        "vehicle_auroc": float(roc_auc_score(labels, scores)),
        # Project convention requested for the thesis: PR-AUC means Average
        # Precision. Keep trapezoidal integration as an explicitly named aid.
        "vehicle_pr_auc": average_precision,
        "vehicle_average_precision": average_precision,
        "vehicle_auprc": average_precision,
        "vehicle_pr_auc_trapezoid": trapezoidal_pr_auc,
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
    condition_calibration=None,
    consistency_max_weight=None,
    threshold_mode="normal_p99",
):
    """Evaluate a scoring rule without retraining or repeating model inference."""
    if condition_calibration is not None:
        normal_scoring_errors = _apply_condition_residual_calibration(
            normal_calibration_errors, condition_calibration
        )
        threshold_scoring_errors = _apply_condition_residual_calibration(
            threshold_calibration_errors, condition_calibration
        )
        test_scoring_errors = _apply_condition_residual_calibration(
            test_errors, condition_calibration
        )
    else:
        normal_scoring_errors = normal_calibration_errors
        threshold_scoring_errors = threshold_calibration_errors
        test_scoring_errors = test_errors
    normal_calibration_window_scores = _window_scores(
        normal_scoring_errors, pred_weights, recon_weights, dims
    )
    threshold_calibration_window_scores = _window_scores(
        threshold_scoring_errors, pred_weights, recon_weights, dims
    )
    test_window_scores = _window_scores(test_scoring_errors, pred_weights, recon_weights, dims)
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

    consistency_calibration = None
    if consistency_max_weight is not None:
        consistency_calibration = _fit_physical_fusion(
            normal_calibration_window_scores,
            normal_calibration_errors["physical_consistency"],
            consistency_max_weight,
        )
        normal_calibration_window_scores = _apply_physical_fusion(
            normal_calibration_window_scores,
            normal_calibration_errors["physical_consistency"],
            consistency_calibration,
        )
        threshold_calibration_window_scores = _apply_physical_fusion(
            threshold_calibration_window_scores,
            threshold_calibration_errors["physical_consistency"],
            consistency_calibration,
        )
        test_window_scores = _apply_physical_fusion(
            test_window_scores,
            test_errors["physical_consistency"],
            consistency_calibration,
        )

    paper_threshold = None
    if threshold_mode.startswith("paper_"):
        paper_threshold = _paper_labelled_snippet_threshold(
            threshold_calibration_window_scores,
            threshold_calibration_errors["labels"],
        )

    aggregation_sensitivity = {}
    ratio_arrays = {}
    # Include the complete 5%-95% grid used by Zhang et al.'s public robust
    # scoring notebook.  Top-1% and the full mean remain useful endpoints.
    for percent in [1, *range(5, 100, 5), 100]:
        ratio = percent / 100.0
        ratio_calibration_scores, ratio_calibration_labels, _ = _aggregate(
            threshold_calibration_errors, threshold_calibration_window_scores, ratio
        )
        ratio_scores, ratio_labels, _ = _aggregate(test_errors, test_window_scores, ratio)
        key = f"top_{percent}pct" if percent < 100 else "mean"
        aggregation_sensitivity[key] = _metrics(
            ratio_scores,
            ratio_labels,
            ratio_calibration_scores,
            ratio_calibration_labels,
            threshold_mode,
            paper_threshold,
        )
        ratio_arrays[percent] = (
            ratio_calibration_scores,
            ratio_calibration_labels,
            ratio_scores,
            ratio_labels,
        )

    if threshold_mode.startswith("paper_"):
        # Select robust Top-p only on the labelled calibration fold, following
        # Zhang et al.'s public five-fold evaluation. Test labels are not used.
        # The public notebook updates on >=, so a tie selects the larger p.
        candidate_percents = list(range(5, 100, 5))
        selected_percent = max(
            candidate_percents,
            key=lambda percent: (
                roc_auc_score(ratio_arrays[percent][1], ratio_arrays[percent][0]),
                percent,
            ),
        )
        selection_source = "labelled_calibration_vehicle_auroc"
    else:
        selected_percent = int(round(top_ratio * 100))
        if selected_percent not in ratio_arrays:
            selected_percent = 5
        selection_source = "predefined_normal_only_protocol"

    selected_ratio = selected_percent / 100.0
    _, _, normal_calibration_cars = _aggregate(
        normal_calibration_errors, normal_calibration_window_scores, selected_ratio
    )
    calibration_scores, calibration_labels, scores, labels = ratio_arrays[selected_percent]
    _, _, calibration_cars = _aggregate(
        threshold_calibration_errors, threshold_calibration_window_scores, selected_ratio
    )
    _, _, cars = _aggregate(test_errors, test_window_scores, selected_ratio)
    return {
        "score_dims": np.asarray(dims).tolist(),
        "physical_fusion": physical_calibration,
        "condition_residual_calibration": (
            None
            if condition_calibration is None
            else _condition_calibration_metadata(condition_calibration)
        ),
        "physical_consistency_fusion": consistency_calibration,
        "metrics": _metrics(
            scores,
            labels,
            calibration_scores,
            calibration_labels,
            threshold_mode,
            paper_threshold,
        ),
        "aggregation_sensitivity": aggregation_sensitivity,
        "selected_top_ratio": selected_ratio,
        "top_ratio_selection": selection_source,
        "calibration_auroc_at_selected_top_ratio": float(
            roc_auc_score(calibration_labels, calibration_scores)
        ),
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
        seed=args.seed if args.battery_fold_seed < 0 else args.battery_fold_seed,
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
    if not args.normalize:
        scaler = None
    elif args.battery_normalization == "paper_channel":
        scaler = PaperChannelNormalizer(splits["train"])
    else:
        scaler = StreamingMinMaxScaler().fit_records(splits["train"])
    if scaler is not None:
        # Physics features must be formed in the original engineering units;
        # per-channel Min-Max values are otherwise incomparable for spreads.
        args.physical_data_min = scaler.offset_.astype(np.float32).tolist()
        args.physical_data_scale = scaler.scale_.astype(np.float32).tolist()
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
    target_dims = list(RESPONSE_DIMS) if args.battery_response_only_training else None
    out_dim = len(target_dims) if target_dims is not None else 7
    model = build_model(args, 7, args.lookback, out_dim, target_dims=target_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    trainer = build_trainer(
        model, optimizer, args, args.lookback, 7, target_dims,
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
        model, _loader(datasets["validation"], args, False), trainer.device,
        args.physical_response_terms, target_dims, args.use_condition_slow_state
    )
    if splits["calibration"] is splits["validation"]:
        calibration_errors = validation_errors
    else:
        calibration_errors = _collect_errors(
            model,
            _loader(datasets["calibration"], args, False),
            trainer.device,
            args.physical_response_terms, target_dims, args.use_condition_slow_state
        )
    test_errors = _collect_errors(
        model, _loader(datasets["test"], args, False), trainer.device,
        args.physical_response_terms, target_dims, args.use_condition_slow_state
    )
    pred_weights, recon_weights = _branch_weights(
        validation_errors["pred"], validation_errors["recon"], args.score_fusion_mode, args.gamma
    )
    condition_calibration = None
    if args.use_condition_residual_calibration:
        condition_calibration = _fit_condition_residual_calibration(
            validation_errors,
            args.condition_calibration_clusters,
            args.seed,
            args.condition_calibration_method,
            args.condition_calibration_temperature,
            [
                "current_mean", "current_std", "current_last", "current_delta_abs_mean",
                "soc_first", "soc_last", "soc_delta",
                *( ["log1p_mileage"] if args.use_condition_slow_state else [] ),
            ],
        )
    all_dims = np.arange(out_dim)
    response_dims = np.arange(out_dim) if target_dims is not None else np.asarray(RESPONSE_DIMS)
    scoring_specs = {}
    if out_dim == 7:
        # Any seven-channel model can be compared under both the original
        # all-channel MTAD-GAT score and the battery response-only score using
        # exactly the same trained weights and inference errors.
        scoring_specs["all_channels"] = (all_dims, None, None, None)
        scoring_specs["response_channels"] = (response_dims, None, None, None)
        primary_variant = (
            "all_channels" if args.battery_score_channels == "all" else "response_channels"
        )
    else:
        scoring_specs["response_channels"] = (response_dims, None, None, None)
        primary_variant = "response_channels"
    if args.use_physical_response_score:
        scoring_specs["response_plus_physics"] = (
            response_dims,
            args.physical_response_max_weight,
            None,
            None,
        )
        primary_variant = "response_plus_physics"
    if condition_calibration is not None:
        scoring_specs["condition_calibrated_response"] = (
            response_dims,
            None,
            condition_calibration,
            None,
        )
        primary_variant = "condition_calibrated_response"
    if args.use_physical_consistency_head:
        consistency_name = (
            "condition_plus_physical_consistency"
            if condition_calibration is not None
            else "response_plus_physical_consistency"
        )
        scoring_specs[consistency_name] = (
            response_dims,
            None,
            condition_calibration,
            args.physical_consistency_score_max_weight,
        )
        primary_variant = consistency_name

    scoring_variants = {}
    variant_arrays = {}
    for variant_name, (
        variant_dims,
        max_physical_weight,
        variant_condition_calibration,
        consistency_max_weight,
    ) in scoring_specs.items():
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
            condition_calibration=variant_condition_calibration,
            consistency_max_weight=consistency_max_weight,
            threshold_mode=(
                "paper_labelled_snippet_rank"
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
        "score_dims": (
            [target_dims[index] for index in primary["score_dims"]]
            if target_dims is not None
            else primary["score_dims"]
        ),
        "model_score_dims": primary["score_dims"],
        "model_output_dims": list(range(7)) if target_dims is None else target_dims,
        "response_only_training": bool(args.battery_response_only_training),
        "vehicle_top_ratio": primary["selected_top_ratio"],
        "configured_vehicle_top_ratio": args.battery_vehicle_top_ratio,
        "top_ratio_selection": primary["top_ratio_selection"],
        "calibration_auroc_at_selected_top_ratio": primary[
            "calibration_auroc_at_selected_top_ratio"
        ],
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
        "condition_residual_calibration": primary["condition_residual_calibration"],
        "physical_consistency_fusion": primary["physical_consistency_fusion"],
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
